import fastapi
import os
import logging
from typing import Optional
from email.message import EmailMessage

# Try to import async SMTP lib; fall back to sync smtplib if not present
try:
    import aiosmtplib
    ASYNC_SMTP_AVAILABLE = True
except Exception:
    import smtplib
    ASYNC_SMTP_AVAILABLE = False

from jinja2 import Environment, FileSystemLoader, select_autoescape, Template

from config import config  # your config object
from logger import get_logger

logger = get_logger(__name__)

# Templates folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "email_templates")

# Create Jinja2 env (safe defaults)
_jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"])
)

# Fallback inline templates in case the templates folder is missing
_VERIFY_TEXT = "Open this link to verify your email: {{ verify_link }}"
_RESET_TEXT = "Reset your password using this link: {{ reset_link }}"

def _load_template(name: str, default_html: Optional[str] = None) -> Template:
    """
    Load a Jinja2 template by name from the templates directory.
    Raises exception if template not found and no default provided.
    """
    try:
        return _jinja_env.get_template(name)
    except Exception:
        if default_html:
            logger.debug(f"Email template '{name}' not found; using built-in fallback.")
            return Template(default_html)
        logger.error(f"Email template '{name}' not found and no default provided.")
        raise

def _build_verify_link(token: str) -> str:
    base = getattr(config, "FRONTEND_BASE_URL", None) or getattr(config, "API_BASE_URL", "")
    path = getattr(config, "EMAIL_VERIFY_PATH", "/verify-email")
    return f"{base.rstrip('/')}{path}?token={token}"

def _build_reset_link(token: str) -> str:
    base = getattr(config, "FRONTEND_BASE_URL", None) or getattr(config, "API_BASE_URL", "")
    path = getattr(config, "EMAIL_RESET_PATH", "/reset-password")
    return f"{base.rstrip('/')}{path}?token={token}"

async def _send_async_smtp(message: EmailMessage) -> None:
    """Send email using aiosmtplib (async). Raises on failure."""
    if not ASYNC_SMTP_AVAILABLE:
        raise RuntimeError("aiosmtplib is not available for async sending.")

    host = config.SMTP_HOST
    port = int(getattr(config, "SMTP_PORT", 587))
    username = getattr(config, "SMTP_USER", None)
    password = getattr(config, "SMTP_PASSWORD", None)
    use_tls = bool(getattr(config, "SMTP_USE_TLS", True))

    try:
        if use_tls:
            await aiosmtplib.send(message, hostname=host, port=port, username=username, password=password, start_tls=True)
        else:
            await aiosmtplib.send(message, hostname=host, port=port, username=username, password=password)
        logger.info("Email sent (async) to %s", message["To"])
    except Exception as exc:
        logger.error("Failed to send email (async) to %s: %s", message["To"], str(exc))
        raise

def _send_sync_smtp(message: EmailMessage) -> None:
    """Send email synchronously using smtplib. Raises on failure."""
    host = config.SMTP_HOST
    port = int(getattr(config, "SMTP_PORT", 587))
    username = getattr(config, "SMTP_USER", None)
    password = getattr(config, "SMTP_PASSWORD", None)
    use_tls = bool(getattr(config, "SMTP_USE_TLS", True))

    try:
        if use_tls:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                smtp.starttls()
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if username and password:
                    smtp.login(username, password)
                smtp.send_message(message)
        logger.info("Email sent (sync) to %s", message["To"])
    except Exception as exc:
        logger.error("Failed to send email (sync) to %s: %s", message["To"], str(exc))
        raise

async def _send_email(
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    *,
    async_send: bool = True,
) -> None:
    """
    Send an email. Use async SMTP if available; otherwise send synchronously.
    """
    message = EmailMessage()
    message["From"] = getattr(config, "EMAIL_FROM", "noreply@example.com")
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    logger.debug("Preparing to send email to %s with subject '%s'", to_email, subject)

    if async_send and ASYNC_SMTP_AVAILABLE:
        await _send_async_smtp(message)
    else:
        _send_sync_smtp(message)

def send_verification_email(
    to_email: str,
    token: str,
    *,
    background_task: Optional["fastapi.BackgroundTasks"] = None,
    subject: Optional[str] = None,
) -> None:
    """
    Build and send an email verification message.
    """
    subject = subject or "Verify your email"
    verify_link = _build_verify_link(token)
    tpl = _load_template("verify_email.html")
    html = tpl.render(verify_link=verify_link)
    text_tpl = Template(_VERIFY_TEXT)
    text = text_tpl.render(verify_link=verify_link)

    if background_task:
        logger.debug("Background sending requested, use BackgroundTasks in route to schedule _send_email.")
        return

    if ASYNC_SMTP_AVAILABLE:
        import asyncio
        asyncio.run(_send_email(to_email, subject, html, text, async_send=True))
    else:
        message = EmailMessage()
        message["From"] = getattr(config, "EMAIL_FROM", "noreply@example.com")
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(text)
        message.add_alternative(html, subtype="html")
        _send_sync_smtp(message)

def send_password_reset_email(
    to_email: str,
    token: str,
    *,
    background_task: Optional["fastapi.BackgroundTasks"] = None,
    subject: Optional[str] = None,
) -> None:
    """
    Build and send a password reset email.
    """
    subject = subject or "Reset your password"
    reset_link = _build_reset_link(token)
    tpl = _load_template("reset_password.html")
    html = tpl.render(reset_link=reset_link)
    text_tpl = Template(_RESET_TEXT)
    text = text_tpl.render(reset_link=reset_link)

    if background_task:
        logger.debug("Background sending requested, use BackgroundTasks in route to schedule _send_email.")
        return

    if ASYNC_SMTP_AVAILABLE:
        import asyncio
        asyncio.run(_send_email(to_email, subject, html, text, async_send=True))
    else:
        message = EmailMessage()
        message["From"] = getattr(config, "EMAIL_FROM", "noreply@example.com")
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(text)
        message.add_alternative(html, subtype="html")
        _send_sync_smtp(message)
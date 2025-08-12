import os

from typing import Optional
from email.message import EmailMessage
try:
    import aiosmtplib
    ASYNC_SMTP_AVAILABLE = True
except Exception:
    import smtplib
    ASYNC_SMTP_AVAILABLE = False

from jinja2 import Environment,FileSystemLoader,select_autoescape,Template

from config import config
from logger import get_logger

base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir,'templates')

logger = get_logger(__name__)
_verify_text = "<p>Open The Link to verify your Email</p>"
_reset_text = "<p>Open The Link to Reset your Password</p>"
_jinja_env = Environment(
    loader=FileSystemLoader(templates_dir),
    autoescape=select_autoescape('html')
)

def load_templates(name:str, default:Optional[str] = None) ->Template:
    try:
        return _jinja_env.get_template(name)
    except Exception:
        if default is None:
            raise 
        else:
            logger.debug('Template %s Not Found using Defaul One',name)
            return Template(default)

def build_link(token:str ,path:str, path_key:str) ->str:
    base = getattr(config,"FRONTEND_BASE_URL","") or getattr(config,"API_BASE_URL","")
    path = getattr(config,path,path_key)
    return f'{base.rstrip("/")}{path}?token={token}'

def send_email(msg:EmailMessage) ->None:
    host = config.SMTP_HOST
    port = int(getattr(config,'SMTP_PORT',587))
    username = getattr(config,'SMTP_USER',None)
    password = getattr(config,'SMTP_PASS',None)
    use_tls = bool(getattr(config,'SMTP_TLS',True))

    if ASYNC_SMTP_AVAILABLE:
        import asyncio
        asyncio.run(
            aiosmtplib.send(
                msg,
                hostname=host,
                port=port,
                username=username,
                password=password,
                use_tls=use_tls
            )
        )
        logger.info('Verification Email (Async) Sent to %s',msg['To'])
        return
    else:
        with smtplib.SMTP(host,port,timeout=10) as smtp:
            if use_tls:
                smtp.starttls()
            
            if username and password:
                smtp.login(username,password)
            
            smtp.send_message(msg)
            logger.info('Verification Email (Sync) Sent To %s',msg['To'])
            return

def build_email(html:str, to_email:str, text:str, subject:str,) ->EmailMessage:
    msg = EmailMessage()
    msg['To'] = to_email
    msg['From'] = getattr(config,'FROM_EMAIL','noreply@example.com')
    msg['subject'] = subject
    msg.set_content(text)
    msg.add_alternative(html)
    return msg

def send_verification_email(token:str, to_email:str, subject:Optional[str] = None ) -> None:
    """Sends a verification email with a link to verify the user's email address
    
    Args:
        token (str): The token to include in the verification link
        to_email (str): The email address to send the verification email to
        subject (Optional[str], optional): The subject of the email. Defaults to 'Verify Your Email'.
    :return: Send Email Verification Request Via Email

    """
    subject = subject or 'Verify Your Email'
    verify_link = build_link(token,'EMAIL_VERIFY_PATH','/verify-email')
    tpl = load_templates('verify_email.html')
    html = tpl.render(verify_link=verify_link)
    text = Template(_verify_text).render(verify_link=verify_link)
    msg = build_email(html,to_email,text,subject)
    return send_email(msg)

def send_reset_password(token:str ,to_email:str, subject:Optional[str] = None) -> None:
    """
    Build and send a password reset email.

    :param token: A verification token issued by the auth service
    :param to_email: The email address to send the verification email to
    :param subject: Optional subject line for the email; defaults to "Reset Your Password"
    :return: Send Reset Password Request Via Email
    """

    subject = subject or 'Reset Your Password'
    reset_link = build_link(token,'PASSWORD_RESET_PATH','/reset-password')
    tpl = load_templates('reset_password.html')
    html = tpl.render(reset_link=reset_link)
    text = Template(_reset_text).render(reset_link=reset_link)
    msg = build_email(html,to_email,text,subject)
    return send_email(msg)

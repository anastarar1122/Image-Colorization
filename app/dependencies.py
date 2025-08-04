import mlflow

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.auth_handler import decode_jwt
from config import config
from logger import get_logger

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_jwt(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload


def get_request_logger(request: Request):
    logger = get_logger("uvicorn.access")
    logger.info(f"[{request.method}] {request.url}")
    return logger

def get_mlflow_tracker():
    try:
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENTAL_NAME)
        return mlflow
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MLflow: {str(e)}")

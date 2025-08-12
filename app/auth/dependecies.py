import redis

from fastapi import Depends,HTTPException
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED,HTTP_403_FORBIDDEN,HTTP_404_NOT_FOUND
from sqlalchemy.orm import Session

from db.db import get_db
from db.models import User
from logger import get_logger
from auth.auth_handler import AuthHandler,TokenType
from db.redis_client import is_blacklisted

logger = get_logger(__name__)


auth_handler = AuthHandler()
auth_scheme = HTTPBearer()

access = TokenType.ACCESS
refresh = TokenType.REFRESH


def _get_token_str(creds:HTTPAuthorizationCredentials) ->str:
    return creds.credentials



def get_current_user(
        creds:HTTPAuthorizationCredentials = Depends(auth_scheme),
        db:Session = Depends(get_db),
) -> User:
    token_str = _get_token_str(creds)
    logger.info('Starting Token Verification')
    payload = auth_handler.verify_token(token_str,False)
    if not payload:
        raise HTTPException(401,'invalid/Missing Token Payload is empty')
    
    jti = payload.get('jti')
    user_id = payload.get('sub')
    check_key = jti or token_str or user_id
    if check_key is None:
        raise HTTPException(HTTP_404_NOT_FOUND,'Token/Subject Not Found')
    logger.info(f'User Extracted From Token Got: {user_id}')
    
    if is_blacklisted(check_key):
        logger.info('Attempt To use BlackListed Token for sub=%s',user_id)
        raise HTTPException(HTTP_403_FORBIDDEN,'Token Revoked')
    
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(HTTP_404_NOT_FOUND,'User Not Found')
    
    logger.info(f'User Found With Name: {user.username}')
    return user

def JWTBearerAdminOnly(user:User = Depends(get_current_user)) -> User:
    logger.info(f'Checking If User:{user.username} is Super User')
    if not getattr(user,'is_superuser'):
        raise HTTPException(403,'Admin Previlages Required')
    return user

def JWTBearerUserOnly(user:User = Depends(get_current_user)) -> User:
    logger.info(f'Checking If User: {user.username} is Active')
    if not getattr(user,'is_active'):
        raise HTTPException(403,'User Previlages Required')
    return user

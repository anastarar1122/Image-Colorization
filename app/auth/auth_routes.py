from fastapi import APIRouter,Depends,HTTPException,BackgroundTasks,Form
from fastapi.security import OAuth2PasswordRequestForm
from starlette.status import HTTP_401_UNAUTHORIZED
from sqlalchemy.orm import Session
from typing import Dict,Any


from db.db import get_db
from db.user_repository import UserRepository
from db.token_repository import TokenRepository
from auth.auth_handler import AuthHandler
from db.schemas import Token as TokenSchema
from logger import get_logger
from config import config

logger = get_logger(__name__)
auth_handler = AuthHandler()
router = APIRouter(prefix='/auth',tags=['auth'])

def login(form_data:OAuth2PasswordRequestForm = Depends(), db:Session = Depends(get_db)) ->Dict[str,Any]:
    user_repo = UserRepository(db)
    user = user_repo.authenticate_user(form_data.username,form_data.password)
    if not user:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Invalid Credentials'
        )

    user_id = user.id.value
    roles = [r.name for r in user.roles] if getattr(user,'roles') else 'user'
    scopes = []
    access_token = auth_handler.create_access_token(
        role=','.join(roles),
        subject=user_id,
        scopes=scopes
    )
    refresh_token = auth_handler.create_refresh_token(
        role=','.join(roles),
        subject= user_id,
        scopes=scopes
    )

    if refresh_token is None or access_token is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Invalid Credentials Token Is Empty'
        )
    
    refresh_payload = auth_handler.decode_token(refresh_token,config.JWT_REFRESH_KEY)
    if refresh_payload is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Missing/Expired Token Payload is Emtpy'
        )
    
    jti = refresh_payload.get('jti')
    tr = TokenRepository(db)
    tr.add_refresh_token(
        token_str=refresh_token,
        user_id=user_id,
        jti=str(jti)
    )
    
    return {'access_token':access_token,'refresh_token':refresh_token,'type':'Bearer'}

@router.post('/refresh',response_model=TokenSchema)
def refresh_token(refresh_token:str = Form(...), db:Session = Depends(get_db)) ->Dict[str,Any]:
    payload = auth_handler.verify_token(refresh_token,True)
    if payload is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Invalid/Expired Token Payload is Empty'
        )
    
    tr = TokenRepository(db)

    if tr.is_revoked(refresh_token):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Refresh Token is Revoked'
        )
    
    new_tokens = auth_handler.refresh_token(refresh_token)
    new_refresh_token = new_tokens['refresh_token']
    new_payload = auth_handler.decode_token(new_refresh_token,config.JWT_REFRESH_KEY)
    
    if new_payload is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='Expired/Invalid Token Payload is Empty'
        )
    user_id = payload.get('subject')
    new_jti = new_payload.get('jti')
    if user_id is None or new_jti is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail='MalFormed Token'
        )
    
    tr.add_refresh_token(new_refresh_token,user_id,new_jti)
    tr.revoke_token(refresh_token)
    return {'access_token':new_tokens['access_token'],'refresh_token':new_refresh_token,'type':'bearer'}

@router.post('logout',response_model=TokenSchema)
def logout(refresh_token:str = Form(...), db:Session = Depends(get_db)) ->Dict[str,str]:
    tr = TokenRepository(db)
    revoked = tr.revoke_token(refresh_token)
    if not revoked:
        logger.info('Logout Attmepted For Token Not Found in DB; marking in redis for safety')
    
    return {'msg':'Logged Out'}
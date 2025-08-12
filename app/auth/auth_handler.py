import jwt
import uuid

from datetime import datetime,timedelta,timezone
from typing import Optional,Union,Dict,List,Any

from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

from config import config
from logger import get_logger

logger = get_logger(__name__)

class TokenType:
    ACCESS = 'access'
    REFRESH = 'refresh'

class AuthHandler:
    def __init__(self) -> None:
        self.access_expire = config.JWT_ACCESS_EXPIRY
        self.refresh_expire = config.JWT_REFRESH_EXPIRY
        self.access_key = config.JWT_SECRET_KEY
        self.refresh_key = config.JWT_REFRESH_KEY
        self.algorithm = config.JWT_ALGORITHM
    
    def _create_token(
        self,
        role:str,
        secret:str,
        token_type:str,
        expire_delta:timedelta,
        subject:Union[str,int],
        scopes:Optional[List[str]],
        extra_data:Optional[Dict[str,Any]],
    ) ->str:

        utc_now = datetime.now(timezone.utc)
        exp = utc_now + expire_delta
        jti = str(uuid.uuid4())
        payload = {
            'sub':str(subject),
            'roles':role,
            'type':token_type,
            'scopes':scopes or [],
            'exp':exp,
            'iat':utc_now,
            'jti':jti
        }
        if extra_data:
            payload.update(extra_data)
        
        logger.info('Creating %s Token for sub=%s, role=%s, jti=%s',
                    token_type,
                    subject,
                    role,
                    jti
                    )

        token = jwt.encode(payload,secret,algorithm=self.algorithm)
        logger.info(f'Token Created')
        return token
    
    def create_access_token(
        self,
        role:str,
        subject:Union[str,int],
        scopes:Optional[List[str]] = None,
        extra_data:Optional[Dict[str,Any]] = None,
    ) ->str:
        tt = TokenType.ACCESS
        exp = timedelta(minutes=self.access_expire)

        return self._create_token(
            role,self.access_key,
            tt,exp,subject,
            scopes,extra_data
        )
    
    def create_refresh_token(
        self,
        role:str,
        subject:Union[str,int],
        scopes:Optional[List[str]] = None,
        extra_data:Optional[Dict[str,Any]] = None,
    ) ->str:
        tt = TokenType.REFRESH
        exp = timedelta(minutes=self.refresh_expire)

        return self._create_token(
            role,self.refresh_key,
            tt,exp,subject,
            scopes,extra_data
        )
    
    def decode_token(self,token:str,secret:str) ->Optional[Dict[str,Any]]:
        token_type = None
        try:
            payload = jwt.decode(token,secret,algorithms=[self.algorithm])
            if payload is None:
                raise HTTPException(
                    HTTP_401_UNAUTHORIZED,
                    'Missing/Invalid Bearer Token'
                )
            
            token_type = payload.get('type')
            sub = payload.get('subject')
            logger.debug('Decoded Token With subject=%s, Type=%s',sub,token_type)
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning('%s Token Expired',token_type or 'Unknown')
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                'Token Epired'
            )
        except jwt.InvalidSignatureError:
            logger.warning('%s Token Has Invalid Signature',token_type or 'Unkonw')
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                'Invalid Signature'
            )
        except jwt.InvalidTokenError:
            logger.warning('%s Token is Invalid',token_type or 'Unkown')
    
    def verify_token(self,token:str,refresh_token:bool) -> Optional[Dict[str,Any]]:
        secret = self.refresh_key if refresh_token else self.access_key

        try:
            payload = self.decode_token(token,secret)
            return payload
        except HTTPException:
            return None
    
    def refresh_token(self,refresh_token:str) ->Dict[str,Any]:
        try:
            payload = self.decode_token(refresh_token,self.refresh_key)
            if not payload:
                raise HTTPException(
                    HTTP_401_UNAUTHORIZED,
                    'Empty Payload Token is Invalid/Expired'
                )
            user_id = payload.get('subject')
            roles = payload.get('roles')
            scopes = payload.get('scopes')

            if not user_id or not roles:
                raise HTTPException(HTTP_401_UNAUTHORIZED,'Invalid Refresh Payload Token')
            
            new_access = self.create_access_token(roles,user_id,scopes)
            new_refresh = self.create_refresh_token(roles,user_id,scopes)
            logger.info('Refreshed Tokens For subject=%s, Roles=%s',user_id,roles)
            return {'access_token':new_access,'refresh_token':new_refresh}
        except Exception as e:
            raise HTTPException(403,f'UnExpected Error while Creating Refreshed Tokens: {e}')
    def has_scope(self,payload:Dict[str,Any], required_scope:str) -> bool:
        scopes = payload.get('scopes',[]) 
        return required_scope in scopes
    
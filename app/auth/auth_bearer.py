from typing import Optional,List,Dict,Any

from fastapi import Request,Depends,HTTPException
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED,HTTP_403_FORBIDDEN

from auth_handler import AuthHandler
from logger import get_logger
from config import config

logger = get_logger(__name__)
auth_handler = AuthHandler()

class AuthBearer(HTTPBearer):
    def __init__(
            self,
            *,
            bearer_format:Optional[str] = 'JWT',
            scheme_name:Optional[str] = 'JWT Auth',
            description:Optional[str] = 'JWT Authentication With Bearer Scheme',
            auto_error:bool,
            required_scopes:Optional[List[str]] = None,
            required_roles:Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            bearerFormat=bearer_format,
            scheme_name=scheme_name,
            description=description,
            auto_error=auto_error)
        
        self.req_scopes = required_scopes or []
        self.req_roles = required_roles or []
    
    async def __call__(self, request: Request) -> Optional[Dict[str,Any]]:
        token = await super().__call__(request)
        if token is None:
            raise HTTPException(HTTP_401_UNAUTHORIZED,'Token Is Missing')
        
        token_str = token.credentials

        try:
            payload = auth_handler.decode_token(token_str,config.JWT_SECRET_KEY)
            if payload is None:
                raise HTTPException(
                    HTTP_401_UNAUTHORIZED,
                    'Invalid/Missing Token Payload is Empty'
                )
            logger.debug(f'[{request.method}, {request.url.path}] Payload Decoded For User: {payload.get("sub")}')

            user_scopes = payload.get('scopes',[])
            if self.req_scopes:
                if not all(scope in user_scopes for scope in self.req_scopes):
                    logger.warning(f'Missing Required Scopes: {self.req_scopes}')
                    raise HTTPException(
                        HTTP_403_FORBIDDEN,
                        'InSufficient Token Scopes'
                    )
            
            user_roles = payload.get('roles',[])
            if self.req_roles:
                if not all(role in user_roles for role in self.req_roles):
                    logger.warning(f'Missing Required Roles: {self.req_roles}')
                    raise HTTPException(
                        HTTP_403_FORBIDDEN,
                        'InSufficient Role Previlages'
                    )
            
            return payload
        
        except HTTPException as e:
            logger.warning(f'[{request.method}, {request.url.path}] Auth Error: {e.detail}')
            raise e
        
        except Exception as e:
            logger.warning(f'[{request.method}, {request.url.path}] UnExpected error accured')
            raise HTTPException(
                HTTP_401_UNAUTHORIZED,
                'Authentication Failed Dur To UnExpected error'
            )


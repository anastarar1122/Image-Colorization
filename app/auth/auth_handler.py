import jwt
from datetime import datetime,timedelta,timezone
from typing import Optional,Dict,Any,Union

from fastapi import HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED

from config import config
from logger import get_logger

logger = get_logger(__name__)

class TokenType:
    ACCESS = 'access'
    REFRESH = 'refresh'



class AuthHandler:
    def __init__(self):
        self.jwt_secret_key = config.JWT_SECRET_KEY
        self.jwt_refresh_key = config.JWT_REFRESH_KEY
        self.algorithm = config.JWT_ALGORITHM
        self.access_expire = config.JWT_ACCESS_EXPIRY
        self.refresh_expire = config.JWT_REFRESH_EXPIRY
    

    def create_token(
        self,
        subject:Union[str,int],
        expire_delta:timedelta,
        token_type:str,
        secret:str,
        extra_data:Optional[Dict[str,Any]] = None,
    ) ->str:
        """
        Creates a JWT token for the given subject (user ID or username).

        Args:
            subject (Union[str,int]): The user ID or username.
            expire_delta (timedelta): The time delta after which the token expires.
            token_type (str): The type of the token, either 'access' or 'refresh'.
            secret (str): The secret key used to sign the token.
            extra_data (Optional[Dict[str,Any]], optional): Additional data to include in the payload. Defaults to None.
    
        Returns:
            str: The created JWT token.
        """

        utc_now = datetime.now(timezone.utc)
        logger.info(f'With {subject}/{token_type} Starting To Create Token...')


        payload = {
            'sub':str(subject),
            'exp':utc_now + expire_delta,
            'type':token_type
        }

        if extra_data:
            payload.update(extra_data)
        
        token = jwt.encode(payload,secret,algorithm=self.algorithm)
        logger.debug(f'With {subject}/{token_type} Token Created Successfully')
        return token
    
    def create_access_token(
        self,
        subject:Union[str,int],
        extra_data:Optional[Dict[str,Any]] = None,
    ) ->str:
        exp = timedelta(minutes=self.access_expire)
        return self.create_token(
            subject,
            exp,
            TokenType.ACCESS,
            self.jwt_secret_key,
            extra_data
        )
    
    def create_refresh_token(
        self,
        subject:Union[str,int],
        extra_data:Optional[Dict[str,int]] = None,
    ) -> str:
        exp = timedelta(minutes=self.refresh_expire)
        return self.create_token(
            subject,
            exp,
            TokenType.REFRESH,
            self.jwt_refresh_key,
            extra_data
        )
    


    
    def decode_token(self,token:str, secret:str,algorithm:str, expected_type:str ) ->Optional[Dict[str,Any]]:
        """       
        
        Decodes a JWT token and checks if the type matches the expected type.
        Args:
            token (str): The JWT token to decode.
            secret (str): The secret key used to sign the token.
            algorithm (str): The algorithm used to sign the token.
            expected_type (str): The expected type of the token.

        Returns:
            Optional[Dict[str,Any]]: The decoded payload if the token is valid, otherwise None.

        Raises:
            HTTPException: If the token is invalid, expired, or has an invalid type.
        """

        try:
            payload = jwt.decode(token,secret,algorithms=[algorithm])
            token_type = payload.get('type')

            if token_type != expected_type:
                logger.warning(f'Invalid Token Type Expected:{expected_type} Got: {token_type}')
                raise HTTPException(HTTP_401_UNAUTHORIZED,'Invalid Token Type')
            
            return payload
        
        except jwt.ExpiredSignatureError as e:
            logger.warning(f'{expected_type.capitalize()} Token Expired')
            raise HTTPException(HTTP_401_UNAUTHORIZED,'Expired Token')
        
        except jwt.InvalidTokenError as i:
            logger.warning(f'Invalid {expected_type.capitalize()} Token')
            raise HTTPException(HTTP_401_UNAUTHORIZED,f'Invalid {expected_type.capitalize()} Token')
    
    def verify_token(self,token:str, refresh:bool = False) ->bool:
        
        """
        Verifies the validity of a JWT token.

        Args:
            token (str): The JWT token to verify.
            refresh (bool, optional): Indicates whether the token is a refresh token. Defaults to False.

        Returns:
            bool: True if the token is valid, False otherwise.
        """

        secret = self.jwt_refresh_key if refresh else self.jwt_secret_key
        try:
            jwt.decode(token,secret,algorithms=[self.algorithm])
            return True
        
        except jwt.PyJWTError:
            return False
    
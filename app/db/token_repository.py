from typing import Optional
from datetime import datetime
from sqlalchemy.orm import Session

from db.models import RefreshToken
from logger import get_logger

logger = get_logger(__name__)

class TokenRepository:
    def __init__(self, db:Session) -> None:
        self.db = db
    

    def add_refresh_token(
        self,
        token_str:str,
        user_id:int,
        jti:str,
        expires_at:Optional[datetime] = None,
    ) -> RefreshToken:
        
        """
        Add a new refresh token to the database.

        Args:
            token_str (str): The refresh token string.
            user_id (int): The ID of the user associated with the refresh token.
            jti (str): The unique identifier for the token.
            expires_at (Optional[datetime]): The expiration datetime of the token. Defaults to None.

        Returns:
            RefreshToken: The newly created RefreshToken object.

        Raises:
            Exception: If the token could not be persisted in the database.
        """

        try:
            rt = RefreshToken(
                token=token_str,
                user_id=user_id,
                is_revoked=False,
                jti=jti,
                expires_at=expires_at
            )

            self.db.add(rt)
            self.db.commit()
            self.db.refresh(rt)
            logger.info('Presisted Refresh Token for user_id=%s, jti=%s, expires_at',user_id,jti,expires_at)
            return rt

        except Exception as e:
            self.db.rollback()
            logger.exception('Failed To Presist Refresh Token: %s',user_id)
            raise e
    
    def revoke_token(self, token_or_jti:str)->bool:
        """
        Revoke a refresh token by marking it as revoked in the database

        Args:
            token_or_jti (str): The token string or JTI of the token to revoke

        Returns:
            bool: True if the token was revoked, False if it was not found
        """
        try:
            try:
                logger.info('Trying To Get Token using Token String')
                rt = self.db.query(RefreshToken).filter(RefreshToken.token == token_or_jti).first()
                logger.info('Token Found Using Token String Skipping The JTI Token Step')
            
            except:
                logger.warning('Token String Method Failed Using JTI Method To Fetch Token')
                rt = self.db.query(RefreshToken).filter(RefreshToken.jti == token_or_jti).first()
                logger.info('Token Found Using JTI')
            
            if not rt:
                logger.warning('Token Not Found by Token_str/JTI')
                return False
            
            rt.update({'is_revoked':True})
            self.db.commit()
            logger.info('Revoked Refresh Token for user_id=%s',rt.user_id)
            return True
        except Exception as e:
            logger.exception('Failed To Revoke Token: %s',e)
            raise e
    
    def is_revoked(self, token_or_jti):
        """
        Check if a refresh token is revoked or not

        Args:
            token_or_jti (str): The token string or JTI of the token to check

        Returns:
            bool: True if the token is revoked, False if it is not revoked
        """
        
        try:
            try:
                logger.info('Trying To Get Token using Token String')
                rt = self.db.query(RefreshToken).filter(RefreshToken.token == token_or_jti).first()
                logger.info('Token Found Using Token String Skipping The JTI Token Step')
            
            except:
                logger.warning('Token String Method Failed Using JTI Method To Fetch Token')
                rt = self.db.query(RefreshToken).filter(RefreshToken.jti == token_or_jti).first()
                logger.info('Token Found Using JTI')
            
            if not rt:
                logger.exception('Token Not Found for %s',token_or_jti)
                return False
            
            return bool(rt.is_revoked)
        except Exception as e:
            logger.exception('Token Revocation Failed; %s',e)
            return True
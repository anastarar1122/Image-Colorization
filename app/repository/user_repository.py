from pydantic import EmailStr
from typing import Optional,List
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from sqlalchemy.exc import SQLAlchemyError

from config import config
from models.orm_models import User
from logger import get_logger
from db.schemas import UserCreate,UserUpdate

logger = get_logger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRepository:
    def __init__(self,db:Session):
        self.db = db
    
    def  get_user_by_id(self,id:int) -> Optional[User]:
        return self.db.query(User).filter(User.id == id).first()
    
    def get_user_by_name(self,username:str) -> Optional[User]:
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self,email:EmailStr) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def get_users(self,skip:int=0, limit:int=10) -> List[User]:
        return self.db.query(User).offset(skip).limit(limit).all()
    
    def verify_password(self,plain_password:str, hashed_password:str) ->bool:
        return pwd_context.verify(plain_password,hashed_password)
    
    def hash_password(self,password:str) ->str:
        return pwd_context.hash(password)
    
    def authenticate_user(self,username_or_email:str, password:str)-> Optional[User]:
        user = self.get_user_by_email(username_or_email) or self.get_user_by_name(username_or_email)
        if not user:
            logger.warning(f'Authentication Faield: User {username_or_email}')
            return None
        if not self.verify_password(password, str(user.hashed_password)):
            return None
        return user


    def create_user(self,username:str, email:str, password:str) -> Optional[User]:
        hashed_password = self.hash_password(password)

        db_user = User(
            username=username,
            email=email,
            password=hashed_password
        )
        try:
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            logger.info(f'User Created: {db_user.username}')
            return db_user
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f'Failed To Create User: {db_user.username}')
            raise
    
    def update_user(self,user_id:int, user_update:UserUpdate) -> Optional[User]:
        user = self.get_user_by_id(user_id)
        if not user:
            logger.error(f'User Not Found: user_id={user_id}')
            raise 

        for field,value in user_update.model_dump(exclude_unset=True).items():
            setattr(user,field,value)
        
        try:
            self.db.commit()
            self.db.refresh(user)
            logger.info(f'Updated User: {user.username}')
            return user
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f'Update Failed User: {user.username}: {e}')
            raise
    
    def delete_user(self,user_id:int) ->bool:
        user = self.get_user_by_id(user_id)
        if not user:
            logger.error(f'No User Found with {user_id}')
        try:
            self.db.delete(user)
            self.db.commit()
            logger.info(f'{user_id}: User Deleted')
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f'Failed to Delete User: {user_id}')
            return False
        
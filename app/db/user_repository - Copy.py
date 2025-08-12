from typing import List,Optional
from pydantic import BaseModel,EmailStr,Field



class UserBase(BaseModel):
    username:str = Field(description="User's Name",examples=['waleed@123'])
    email: EmailStr = Field(description="User'Email",examples=['Waleed@example.com'])

class UserCreate(UserBase):
    password:str = Field(description='Account Password',examples=['StrongPass@123'],min_length=8) 


class UserLoggin(BaseModel):
    email:EmailStr
    password:str 

class UserOut(UserBase):
    id:int
    is_active: bool
    is_superuser: bool

    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    username:Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class Token(BaseModel):
    id:int
    access_token: Optional[bool]
    refresh_token:Optional[bool]

class TokenData(BaseModel):
    user_id:str
    email: EmailStr

class ChangePassword(BaseModel):
    current_password: str 
    new_password:str = Field(description='New Password',min_length=8)

class ResetPasswordRequest(BaseModel):
    email:EmailStr

class ResetPasswordConfirm(BaseModel):
    token:str
    new_password:str = Field(min_length=8)

class VarifyRequest(BaseModel):
    email:EmailStr

class PaginatedUserList(BaseModel):
    total:int
    page:int
    per_page:int
    users:List[UserOut]

    class Config:
        orm_mode = True
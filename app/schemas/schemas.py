from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List


class UserBase(BaseModel):
    username: str = Field(..., examples=["waleed123"])
    email: EmailStr = Field(..., examples=["waleed@example.com"])


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, examples=["StrongPass@123"])


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(UserBase):
    id: int
    is_active: bool
    is_superuser: bool

    class Config:
        orm_mode = True


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, examples=["new_username"])
    email: Optional[EmailStr] = Field(None, examples=["newemail@example.com"])
    is_active: Optional[bool]
    is_superuser: Optional[bool]


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[EmailStr] = None


class ChangePassword(BaseModel):
    current_password: str
    new_password:str =  Field(min_length=8)


class ResetPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordConfirm(BaseModel):
    token: str
    new_password:str =  Field(min_length=8)


class VerifyEmailRequest(BaseModel):
    token: str


class PaginatedUserList(BaseModel):
    total: int
    page: int
    per_page: int
    users: List[UserOut]

    class Config:
        orm_mode = True

from sqlalchemy import Boolean,Integer,String,LargeBinary,ForeignKey,DateTime,Column,Text,func,Table
from sqlalchemy.orm import relationship
from db.db import Base

class TimeStampMixin:
    created_on      = Column(DateTime(timezone=True),default=func.now(),nullable=False)
    updated_on      = Column(DateTime(timezone=True),default=func.now(),nullable=False,onupdate=func.now())
    deleted_on      = Column(DateTime(timezone=True),nullable=True)
    

user_roles = Table(
    'user_role',
    Base.metadata,
    Column('user_id',Integer,ForeignKey('users.id',ondelete='CASCADE'),primary_key=True),
    Column('role_id',Integer,ForeignKey('role.id',ondelete='CASCADE'),primary_key=True)
)


class User(Base,TimeStampMixin):
    __tablename__ = 'users'

    id              = Column(Integer,nullable=False,index=True,autoincrement=True,primary_key=True)
    username        = Column(String(50),nullable=False,index=True,unique=True)
    email           = Column(String(120),nullable=False,index=True,unique=True)
    hashed_password = Column(String(128), nullable=False)
    is_active       = Column(Boolean,nullable=False,default=True)
    is_superuser    = Column(Boolean,nullable=False,default=False)

    posts = relationship('Post',back_populates='owner',cascade='all,delete-orphan')
    roles = relationship('Role',secondary=user_roles,back_populates='users')
    refresh_token = relationship('RefreshToken',back_populates='user',cascade='all,delete-orphan')
    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', email='{self.email}')>"
    

class Post(Base,TimeStampMixin):
    __tablename__ = 'posts'

    id              = Column(Integer,primary_key=True,index=True,autoincrement=True)
    title           = Column(String(100),nullable=False)
    content         = Column(Text,nullable=False)
    user_id         = Column(Integer,ForeignKey('users.id',ondelete='CASCADE'))

    owner = relationship('User',back_populates='posts')

    def __repr__(self):
        return f'<Post(id={self.id}, title={self.title[:15]}..., user_id={self.user_id} )>'
    

class Image(Base,TimeStampMixin):
    __tablename__ = 'images'

    id              = Column(Integer,primary_key=True,index=True,autoincrement=True)
    L               = Column(LargeBinary,nullable=False,doc='GrayScale Channel Blob')
    AB              = Column(LargeBinary,nullable=False,doc='Color Channel Blob')

    def __repr__(self):
        return f'< Images(id={self.id}) >'

class RefreshToken(Base,TimeStampMixin):
    __tablename__ = 'refresh_tokens'

    id = Column(Integer,primary_key=True,autoincrement=True)
    token = Column(String(512), nullable=False,unique=True,index=True)
    jti = Column(String(36),nullable=False,index=True,unique=True)
    is_revoked = Column(Boolean,default=False)
    user_id = Column(Integer,ForeignKey('users.id',ondelete='CASCADE'))
    user = relationship('User',back_populates='refresh_tokens')
    expires_at = Column(DateTime,nullable=True)

    def __repr__(self) -> str:
        return f'< RefreshTokens(id={self.id},user_id={self.user_id} ,is_revoked{self.is_revoked}) >'


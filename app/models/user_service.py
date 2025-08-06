from sqlalchemy import Boolean,Integer,String,LargeBinary,ForeignKey,DateTime,Column,Text,func
from sqlalchemy.orm import relationship
from db.db import Base

class TimeStampMixin:
    created_on      = Column(DateTime(timezone=True),default=func.now(),nullable=False)
    updated_on      = Column(DateTime(timezone=True),default=func.now(),nullable=False,onupdate=func.now())
    deleted_on      = Column(DateTime(timezone=True),nullable=True)

class User(Base,TimeStampMixin):
    __tablename__ = 'users'

    id              = Column(Integer,nullable=False,index=True,autoincrement=True,primary_key=True)
    username        = Column(String(50),nullable=False,index=True,unique=True)
    email           = Column(String(120),nullable=False,index=True,unique=True)
    hashed_password = Column(String(128), nullable=False)
    is_active       = Column(Boolean,nullable=False,default=True)
    is_superuser    = Column(Boolean,nullable=False,default=False)

    posts = relationship('Post',back_populates='owner',cascade='all,delete-orphan')

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

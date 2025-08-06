from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from config import config
from logger import get_logger

logger = get_logger(__name__)

db_pass = config.DB_PASSWORD
db_user = config.DB_USER
db_host = config.DB_HOST
db_port = config.DB_PORT
db_name = config.DB_NAME
db_url = f'{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'

engine = create_engine(db_url)
SessionLocal = sessionmaker(bind=engine,autoflush=False)
Base = declarative_base()

def init_db():
    from models.orm_models import Post,User,Image
    logger.info("Initializing the database")
    Base.metadata.create_all(bind=engine)
    logger.info("All Tables Created if not Exists")
    logger.info('ALl Tables Created if not Exists')

def get_db():
    db = SessionLocal()
    logger.info("Getting a database session")
    try:
        logger.info("Yielding the database session")
        yield db
    except Exception as e:
        logger.info(f"Error occurred: {e}")
        logger.info("Rolling back database session")
        db.rollback()
    finally:
        logger.info("Closing the database session")
        db.close()

try:
    init_db()
    logger.info('DataBase Initialized And Connected SuccessFully')
except Exception as e:
    logger.error('DataBase Initialization Failed')
    raise e
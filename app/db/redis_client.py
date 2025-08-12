import redis

from typing import Optional

from config import config
from logger import get_logger

logger = get_logger(__name__)

_redis = redis.Redis(
    host=getattr(config,'REDIS_HOST','localhost'),
    port=getattr(config,'REDIS_PORT',6379),
    db=getattr(config,'REDIS_DB',0),
    decode_responses=True
)

def blacklist_token(jti_or_token:str, expires_seconds:Optional[int] = None) -> None:

    try:
        if expires_seconds:
            _redis.setex(jti_or_token,expires_seconds,'blacklisted')
        else:
            _redis.set(jti_or_token,'blacklisted')
    except Exception as e:
        logger.exception('Failed To BlackList Token')

def is_blacklisted(jti_or_token:str) ->bool:
    try:
        return bool(_redis.get(jti_or_token))
    except:
        logger.exception('Failed To Check BlackList Status')
        return False

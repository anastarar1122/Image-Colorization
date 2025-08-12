from sqlalchemy.orm import Session
from pathlib import Path
from typing import List

from config import config
from db.models import Image
from logger import get_logger
from utils.image_utils import ImageUtils

logger = get_logger(__name__)
img_utils = ImageUtils()

class ImageRepository:
    def __init__(self,db:Session) -> None:
        self.db = db
    
    def get_blob_data(self,l_path:Path,ab_paths:List[Path],max_samples:int):
        l,ab = img_utils.load_data(l_path,ab_paths,max_samples)
        logger.info(f'Loaded {l.shape[0]} Image Pairs For SQL DataBase')
        l_blob = [img.tobytes() for img in l]
        ab_blob = [img.tobytes() for img in ab]
        return list(zip(l_blob,ab_blob))
    
    def insert_image_pairs(self,l_path:Path,ab_paths:List[Path],max_samples:int=1000):
        try:
            blob = self.get_blob_data(l_path,ab_paths,max_samples)
            for l,ab in blob:
                image = Image(l=l,ab=ab)
                self.db.add(image)
            
            self.db.commit()
            logger.info(f'Inserted {len(blob)} Image Pairs into DataBase')
        except Exception as e:
            logger.error(f'Failed To insert Data: {e}')
            self.db.rollback()
            raise
    
    def get_image_by_id(self,image_id:int):
        return self.db.query(Image).filter(Image.id == image_id).first()
    
    def get_all_images(self,limit:int = 100):
        return self.db.query(Image).limit(limit).all()

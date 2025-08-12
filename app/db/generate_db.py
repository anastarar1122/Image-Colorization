import sqlite3
import numpy as np

from pathlib import Path

from config import config
from logger import get_logger
from utils.image_utils import ImageUtils

logger = get_logger(__name__)
img_utils = ImageUtils()


L_path = Path(config.DATASET_NPY_GRAY_PATH) / 'gray_scale.npy'
ab_paths = [
    Path(config.DATASET_NPY_RGB_PATH) / 'ab1.npy',
    Path(config.DATASET_NPY_RGB_PATH) / 'ab2.npy'
]
db_path = Path(config.DB_PATH)

conn = sqlite3.connect(db_path)

conn.execute(
"""
    CREATE TABLE IF NOT EXISTS IMAGES(
    ID INTEGER PRIMARY KEY,
    L BLOB NOT NULL,
    AB BLOB NOT NULL
    )
"""
)
conn.commit()

L,AB = img_utils.load_data(L_path,ab_paths,1000)

logger.info(f'Loaded {L.shape[0]} Image Pairs')

for idx in range(L.shape[0]):
    l_blob = L[idx].tobytes()
    ab_blob = AB[idx].tobytes()

conn.execute("Insert into IMAGES(L,AB) VALUES(?,?)",(l_blob,ab_blob))
conn.commit()
conn.close()
logger.info(f'Inserted {L.shape[0]} Image Pairs into {db_path}')
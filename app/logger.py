import os
import logging
from config import config

def ensure_log_file():
    log_path = config.LOG_PATH

    log_dir = os.path.dirname(log_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)
    
    if not os.path.isfile(log_path):
        with open(log_path,'w') as f:
            f.write('')
    
    return log_path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_path = ensure_log_file()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.propagate = False
    return logger
import os
import mlflow
import logging
import requests
from pathlib import Path
from typing import Literal, Tuple
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    EMBED_DIM: int = 64
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    OUTPUT_CHANNELS: int = 2
    FREEZE_LAYERS: int = 10
    INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 1)

    MODEL_VERSION: str = 'V1'
    MODEL_SAVE_FORMAT: str = 'onnx'
    MODEL_TYPES: Literal['onnx', 'tflite', 'tf'] = 'onnx'
    ARCHITECTURE: Literal['u_net', 'efficient_netB0'] = 'efficient_netB0'

    DATASET_NAME: str = 'IMAGE_COLORISATION_DS'
    DATASET_NPY_RGB_PATH: str = 'F:/.vscode/Projects/p4/data/'
    DATASET_NPY_GRAY_PATH: str = 'F:/.vscode/Projects/p4/data/'
    SAVE_DATASET_PATH: str = 'F:/.vscode/Projects/p4/data/saved_dataset'

    MODEL_SAVE_DIR: str = 'F:/.vscode/Projects/p4/models/'
    MODEL_SAVE_DIR_PB: str = 'F:/.vscode/Projects/p4/models/pb'
    MODEL_SAVE_DIR_H5: str = 'F:/.vscode/Projects/p4/models/h5'
    MODEL_SAVE_DIR_ONNX: str = 'F:/.vscode/Projects/p4/models/onnx'
    MODEL_SAVE_DIR_TFLITE: str = 'F:/.vscode/Projects/p4/models/tflite'

    LOG_PATH: str = 'F:/.vscode/Projects/p4/logs/app.log'
    TENSORBOARD_LOG_DIR: str = 'F:/.vscode/Projects/p4/models/runs'
    SAVE_INPUT_PATH: str = 'F:/.vscode/Projects/p4/saves/input'
    SAVE_OUTPUT_PATH: str = 'F:/.vscode/Projects/p4/saves/output'

    MLFLOW_URI: str = 'http://127.0.0.1:5000'
    MLFLOW_EXPERIMENTAL_NAME:str = 'image_colorizor'
    MLFLOW_ARTIFACT_DIR: str = 'F:/.vscode/Projects/p4/mlflow/artifacts'

    PORT: int = 8000
    HOST: str = '0.0.0.0'
    JWT_SECRET: str = 'default'

    model_config = {
        'env_file': '.env'
    }


    def _is_mlflow_server_healthy(self, endpoint: str = 'health') -> bool:
        try:
            response = requests.get(f'{self.MLFLOW_URI}/api/2.0/{endpoint}')
            if response.status_code == 200:
                mlflow.set_tracking_uri(self.MLFLOW_URI)
                reg_uri = mlflow.get_registry_uri()
                logging.info(f'MLflow server healthy at /{endpoint}')
                logging.info(f'MLflow registry URI: {reg_uri}')
                logging.info('MLflow logging initialized.')
                return True
        except requests.RequestException as e:
            logging.warning(f'MLflow server check failed at /{endpoint}: {e}')
        return False

    def check_mlflow_server(self) -> bool:
        return self._is_mlflow_server_healthy('health') or self._is_mlflow_server_healthy('experiments/list')

    def get_last_run_name(self) -> str:
        run_file = Path(self.MLFLOW_ARTIFACT_DIR) / 'last_run_name.txt'
        if run_file.exists():
            return run_file.read_text().strip()
        else:
            logging.warning('Last run file not found, using default run name: train_dual_models')
            return 'train_dual_models'


config = Settings()

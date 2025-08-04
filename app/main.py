import os
import typer
from fastapi import FastAPI

# Set TensorFlow environment variables globally
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import project components
from app.routers.predict import ModelPredictor
from app.config import config
from pathlib import Path
from app.utils.image_utils import ImageUtils
from app.logger import get_logger

logger = get_logger(__name__)
cli = typer.Typer()
utils = ImageUtils()

app = FastAPI(
    title='Image Colorizaion APi',
    version='1.0.0',
    description='FastApi App for Grayscale To RGB Images Colorization',
    contact={'Name':'Waleed','Email':'waleed.anas.tarar@gmail.com'},
    license_info={'name':'MIT'},
)

@cli.command('predict')
def run_prediction(
    model_filename:str = 'demo.keras',
    architecture:str = 'demo_unet',
    save_results:bool = False,
    use_saved_models:bool = False,
):
    logger.info(f'Typer Command Initiated With Model: {model_filename} Architecture: {architecture}')
    L_path = Path(config.DATASET_NPY_GRAY_PATH) / 'gray_scale.npy'
    logger.info(f'L_Path Loaded')
    ab_paths = [Path(config.DATASET_NPY_RGB_PATH) / 'ab1.npy',Path(config.DATASET_NPY_RGB_PATH) / 'ab2.npy']
    logger.info(f'AB_Paths Loaded')
    

    model_path = Path(config.MODEL_SAVE_DIR_H5) / model_filename
    logger.info(f'Model Path Loaded: {model_filename}')
    if use_saved_models:
        logger.info(f'Using Saved Models For Predictions')
    else:
        logger.info('Training New Models')
    

    predictor = ModelPredictor(
        L_path,ab_paths,model_path,use_saved_models,False,
        False,False,True,False,128,
        500,16,0,0,
        architecture=architecture
    )
    
    
    test_ds,preds,model = predictor.predict(
        True,False,None,None,True,0,0
    )

    results = predictor.evaluate(
        model,test_ds,2,True,True,True,True,True
    )

    for name,val in results.items():
        logger.info(f'{name}:{val}')

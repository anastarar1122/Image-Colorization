import os
import re
import keras
import mlflow
import tensorflow as tf

from io import StringIO
from pathlib import Path
from typing import Tuple, List,Union,cast,Optional

from keras import models
from keras.optimizers import Adam
from keras.saving import save_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from utils.image_utils import ImageUtils
from utils.model_registry import ModelBuilder
from config import config
from logger import get_logger

logger = get_logger(__name__)
img_utils = ImageUtils()
builder = ModelBuilder()

class ModelTrainer:
    def __init__(self):
        self.mlflow_enable = config.check_mlflow_server()

    def train(
            self,
            L_path:Path,
            ab_path:Union[List[Path],Path],
            use_saved_dataset:bool = False,
            save_data:bool = False,
            augment:bool = False,
            shuffle:bool = True,
            save_model:bool = True,
            shuffle_size:int = 512,
            export_format:Union[str,List[str]] = ['h5','pb','tflite','onnx'],
            dataset_size:int = 0,
            batch_size:int = 32,
            epochs:int = 30,
            fine_tune_layers:int=10,
            architecture:str = 'unet',
    ) -> Tuple[keras.callbacks.History,keras.models.Model,tf.data.Dataset]:
        
        train_ds,test_ds,val_ds = img_utils.load_dataset(
            L_path,ab_path,shuffle_size,batch_size,
            dataset_size,augment,shuffle,use_saved_dataset,save_data
            )
        if architecture in ['unet','demo_unet']:
            train_ds = train_ds.map(img_utils.to_multi_output,num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(img_utils.to_multi_output,num_parallel_calls=tf.data.AUTOTUNE)
        
        model,loss,metrics = self._build_and_compile(architecture,fine_tune_layers) 
        model.compile(optimizer=Adam(learning_rate=0.1),loss=loss,metrics=metrics)#type:ignore
        
        cb_early    = EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True,verbose=1)
        cb_reduce   = ReduceLROnPlateau(monitor='val_loss',patience=4,factor=0.5,verbose=1) 
        tb_architecture = re.sub(r'[^\w\-\.]', '_', architecture)
        tb_dir      = Path(config.TENSORBOARD_LOG_DIR) / tb_architecture
        cb_tb       = TensorBoard(log_dir=str(tb_dir),histogram_freq=1,write_graph=False)

        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[cb_early,cb_reduce,cb_tb],
            verbose=1   #type:ignore
        )

        if save_model:
            self.save_trained_models(model,export_format,architecture)
        return hist,model,test_ds

    
    def _build_and_compile(self,architecture,fine_tune_layers:int = 10):

        if architecture in ['simple','unet']:

            model = builder.build_multihead_model(architecture)
            loss = {'decoder_L_out':'mse','decoder_AB_out':'mse'}
            metrics = {'decoder_L_out':['mse','accuracy'],'decoder_AB_out':['mse','accuracy']}
        
        elif architecture == 'mlp':
            model = builder.build_mlp_style()
            loss = 'mse';metrics = ['mse','accuracy']
        
        elif architecture == 'pretrained':
            model = builder.build_pretrained(fine_tune_layers=fine_tune_layers)
            loss = 'mse'; metrics = ['mse','accuracy']

        elif architecture == 'demo_unet':
            model = builder.demo_unet()
            loss = 'mse';metrics = ['mse','accuracy']

        elif architecture == 'demo_multihead':
            model = builder.build_demo_multihead_model()
            metrics = {'decoder_L_out':['mse','accuracy'],'decoder_AB_out':['mse','accuracy']}
            loss = {'decoder_L_out':'mse','decoder_AB_out':'mse'}
        
        
        elif architecture == 'demo_pretrained':
            model = builder.build_demo_pretrained()
            loss = 'mse';metrics = ['mse','accuracy']
        
        else:
            raise ValueError(f'unknow architecture: {architecture}')
        
        model.compile(optimizer=Adam(1e-4),loss=loss,metrics=metrics) #type:ignore
        if config.check_mlflow_server():
            stream = StringIO()
            model.summary(print_fn=lambda x:stream.write(x + '\n'))
            mlflow.log_text(stream.getvalue(),'model_summary.txt')
            logger.info('Model Summary Logged TO Mlflow Successfully')
        else:
            model.summary()
        return model,loss,metrics


    def save_trained_models(
            self,
            model:models.Model,
            save_format:Union[str,List[str]],
            architecture:str,
            retrainable:bool=True
            ) -> None:

        if self.mlflow_enable:
            if 'keras' in save_format:
                if retrainable:
                    try:
                        keras_path = Path(config.MODEL_SAVE_DIR_H5) / f'{architecture}.keras'
                        save_model(model,str(keras_path))
                        logger.info(f'RetrainAble Model Saved in Format Keras at {str(keras_path)}')
                        mlflow.log_artifact(str(keras_path))
                    except Exception as e:
                        logger.warning(f'Model Saving Failed in Format Keras/H5 : {e}')
            

            elif 'pb' in save_format:
                pb_path = Path(config.MODEL_SAVE_DIR_PB) / f'{architecture}.pb'
                if retrainable:
                    try:
                        model.save(str(pb_path))
                        logger.info(f'RetrainAble Model Saved in Format PB at {str(pb_path)}')
                        mlflow.log_artifact(str(pb_path))
                    except Exception as e:
                        logger.warning(f'Model Saving Failed in Format PB : {e}')
                else:
                    try:
                        tf.saved_model.save(model,str(pb_path))
                        mlflow.log_artifact(str(pb_path))
                        logger.info(f'Model Saved in Format PB at {str(pb_path)}')
                        
                    except Exception as e:
                        logger.warning(f'Model Saving Failed in Format PB : {e}')
                    


            elif 'tflite' in save_format:
                if retrainable:
                    logger.warning(f'TFLite Model Can Not Be Retrain: Saving in untrainable Method')
                try:
                    tflite_path = Path(config.MODEL_SAVE_DIR_TFLITE) / f'{architecture}.tflite'
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()
                    tflite_model = cast(bytes,tflite_model)
                    with open(tflite_path,'wb') as f:
                        f.write(tflite_model)
                    logger.info(f'Model Saved in  Format TFLite at {str(tflite_path)}')
                    mlflow.log_artifact(str(tflite_path))
                except Exception as e:
                    logger.warning(f'Model Saving In Format TFLite : {e} ')
    

    def load_saved(self,
                model_path:Path,
                train_ds:Optional[tf.data.Dataset],
                val_ds:Optional[tf.data.Dataset],
                retrain_model:bool = False,
                freeze_layers:int = 0,
                epochs:int=0,
                ):
            if model_path.suffix in ['.keras','.h5']:
                logger.info(f'Loading of  Keras/H5 Model Started From: {model_path}')
                if retrain_model:
                    model = models.load_model(str(model_path))
                    return self.retrain_saved_models(model_path,model,train_ds,val_ds,freeze_layers,epochs)
                else: 
                    return models.load_model(str(model_path))

            elif model_path.suffix == '.tflite':
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
                logger.info(f'Loading of  TFLite Model Started From: {model_path}')
                return interpreter.allocate_tensors()

            
            elif model_path.suffix == '.pb':
                logger.info(f'Loading of  PB Model Started From : {model_path}')
                if retrain_model and 'assets' in os.listdir(model_path):
                    model = models.load_model(str(model_path))
                    return self.retrain_saved_models(model_path,model,train_ds,val_ds,freeze_layers,epochs)
                else:
                    logger.error('Invalid Model File For Loading .PB ')
            else:
                raise ValueError(f'No Model Found From {model_path}')
        
    

    def retrain_saved_models(self,model_path:Path,model,new_train_ds,new_val_ds,freeze_layers,epochs:int):
        if not isinstance(model,models.Model) or epochs <= 0 or freeze_layers <= 0:
            raise ValueError(f'error Due To Model:{isinstance(model,models.Model)} Epochs: {epochs} Freeze_Layers:{freeze_layers}')
        
        if model_path.suffix in ['.h5','.keras']:
            if model is None:
                raise ValueError(f'Got No/None Model at Path: {model_path}')
            
            for layer in model.layers[:-freeze_layers]:
                layer.trainable = False
            logger.info(f'{model_path.name}: {model_path.suffix} Retraining Started with Trainable Layers: {freeze_layers}')

            model.compile(optimizer=Adam(learning_rate=1e-3),loss='mse',metrics=['mse','accuracy'])#type:ignore
            model.fit(new_train_ds,validation_data=new_val_ds,epochs=epochs)
        
        if model_path.suffix == '.pb':
            if model is None:
                raise ValueError(f'Got No/None Model At Path: {model_path}')
            
            for layer in model.layers[:-freeze_layers]:
                layer.trainable = False
            
            logger.info(f'{model_path.name}: {model_path.suffix} Retraining Started with Trainable Layers: {freeze_layers}')
            
            model.compile(optimizer=Adam(learning_rate=1e-3),loss='mse',metrics=['mse','accuracy']) #type:ignore
            model.fit(new_train_ds,new_val_ds,epochs=epochs)
        else:
            logger.info('Untrainable Model Detected Returning UnTrained Model')
        
        return model
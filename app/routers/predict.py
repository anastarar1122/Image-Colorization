import mlflow
import numpy as np
import mlflow.tensorflow as mltf
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from keras.models import Model

from typing import Union,List,Optional,Tuple,Dict
from skimage.color import deltaE_cie76, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from utils.image_utils import ImageUtils
from trainer.model_trainer import ModelTrainer
from config import config
from logger import get_logger

img_utils = ImageUtils()
trainer = ModelTrainer()
logger = get_logger(__name__)

class ModelPredictor:
    allowed_architecture =['simple', 'unet', 'mlp', 'pretrained', 'demo_unet', 'demo_pretrained', 'demo_multihead'] 

    def __init__(
            self,
            L_path:Path,
            ab_paths:Union[List[Path],Path],
            model_path:Path,
            use_saved_model:bool,
            use_saved_dataset:bool,
            save_data:bool,
            augment:bool,
            shuffle:bool,
            save_model:bool,
            shuffle_size:int,
            dataset_size:int,
            batch_size:int,
            epochs:int,
            fine_tune_layers:int,
            export_format:Union[str,List[str]] = ['h5','pb','tflite','onnx'],
            architecture: str = 'unet'
            ) -> None:
        if architecture not in  self.allowed_architecture:
            raise ValueError(f'InValid Architecture: {architecture} Expected: \n\n{self.allowed_architecture}')

        self.L_path          = L_path
        self.epochs          = epochs
        self.shuffle         = shuffle
        self.augment         = augment
        self.ab_paths        = ab_paths
        self.save_data       = save_data
        self.save_model      = save_model
        self.model_path      = model_path
        self.batch_size      = batch_size
        self.shuffle_size    = shuffle_size
        self.dataset_size    = dataset_size
        self.architecture    = architecture
        self.export_format   = export_format
        self.use_saved_model = use_saved_model
        self.fine_tune_layers= fine_tune_layers
        self.use_saved_dataset= use_saved_dataset

    def __repr__(self) -> str:
        return (
            f'ModelPredictor(Architecture={self.architecture!r}),'
            f'Export_Formats=({self.export_format!r}),'
            f'Model_Path=({str(self.model_path)!r})'
            f"epochs={self.epochs}, batch_size={self.batch_size}, "
            f"use_saved_model={self.use_saved_model})"
        )
    
    
    def __str__(self) -> str:
        return f'Model Predictor For {self.architecture} (Format: {self.export_format})'

    def predict(
            self,
            use_saved_model:bool = True,
            retrain_model:bool = False,
            new_train_ds:Optional[tf.data.Dataset] = None,
            new_val_ds:Optional[tf.data.Dataset] = None,
            plot_history:bool = False,
            freeze_layers:int = 0,
            epochs:int = 0,
    ) -> Tuple[tf.data.Dataset,List,Model]:
        train_ds,val_ds,test_ds = img_utils.load_dataset(
            self.L_path,self.ab_paths,
            self.shuffle_size,self.batch_size,
            self.dataset_size,self.augment,
            self.shuffle,self.use_saved_dataset,
            self.save_data
        )
        if train_ds is None or val_ds is None or test_ds is None:
            raise ValueError('Dataset is Emppty')
        
        if use_saved_model:
            if self.model_path.suffix in ['.keras','.h5','.pb']:
                model = trainer.load_saved(self.model_path,new_train_ds,new_val_ds,retrain_model,freeze_layers,epochs)
                y_true,preds = self.get_preds_and_truth(model,test_ds)
                return test_ds,preds,model

            elif self.model_path.suffix == '.tflite':
                interpreter = trainer.load_saved(self.model_path,new_train_ds,new_val_ds,retrain_model,freeze_layers,epochs)
                all_preds,all_gts = [],[]
                for X_batch,y_batch in test_ds:#type:ignore
                    if isinstance(interpreter,tf.lite.Interpreter):
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()

                        input_data = X_batch.numpy().astype(input_details[0]['dtype'])

                        interpreter.set_tensor(input_details[0]['index'],input_data)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        all_preds.append(output_data)
                        all_gts.append(y_batch.numpy())

                all_preds = np.concatenate(all_preds,axis=0)
                all_gts = np.concatenate(all_gts,axis=0)
                return test_ds,all_preds.tolist(),interpreter
                
            
        if config.check_mlflow_server:
            run_name = f'{self.architecture}_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            with mlflow.start_run(run_name=run_name):
                mltf.autolog()

                hist,model,test_ds = trainer.train(
                    self.L_path,self.ab_paths,
                    self.use_saved_dataset,self.save_data,
                    self.augment,self.shuffle,
                    self.save_model,self.shuffle_size,
                    self.export_format,self.dataset_size,
                    self.batch_size,self.epochs,
                    self.fine_tune_layers,self.architecture
                )
                y_true,preds = self.get_preds_and_truth(model,test_ds)


                mlflow.set_tags({
                    'architecture':self.architecture,
                    'Stage':'Training',
                    'Predictor_Repr':repr(self),
                    'developer':'Waleed',
                    'Task':'Image Colorization'
                })

                mlflow.log_params({
                    'L_Path':str(self.L_path),
                    'AB_Paths':str(self.ab_paths),
                    'Batch_Size':self.batch_size,
                    'Epochs':self.epochs,
                    "dataset_size": self.dataset_size,
                    "shuffle_size": self.shuffle_size,
                    "augment": self.augment,
                    "shuffle": self.shuffle,
                    "save_model": self.save_model,
                    "use_saved_dataset": self.use_saved_dataset,
                    "fine_tune_layers": self.fine_tune_layers,
                    "architecture": self.architecture,
                    "export_format": self.export_format,
                })

                if self.save_model:
                    for i in self.export_format:
                        self.model_path = Path(config.MLFLOW_ARTIFACT_DIR)/ f'{self.architecture}.{i}'
                        if self.model_path.exists():
                            mlflow.log_artifact(str(self.model_path),artifact_path='models')


                
                eval_results = self.evaluate(
                    model,test_ds,
                    samples_to_show=0,
                    find_psnr=True,
                    find_ssim=True,
                    find_delta_e_76=True,
                    find_delta_e_2000=True,
                    show_lab_prediction=False
                    
                )
                mlflow.log_dict(eval_results,'eval_metric.json')
                mlflow.log_dict(hist.history,'Training_history.json')
                logger.info(f"MLflow run '{run_name}' logged successfully.")



        if plot_history:
            def plot_losses(h1):
                plt.plot(h1.history['val_loss'],color='red',label=f'Val_Loss')
                plt.plot(h1.history['loss'],color='yellow',label=f'Train_Loss')
                plt.title('Val/Train Losses')
                plt.tight_layout()
                plt.legend()
                plt.show()
            
            def plot_accuracy(h1):
                plt.plot(h1.history['val_accuracy'],color='blue',label=f'Val_accuracy')
                plt.plot(h1.history['accuracy'],color='green',label=f'Train_accuracy')
                plt.title('Val/Train Accuracy')
                plt.tight_layout()
                plt.legend()
                plt.show()

            plot_losses(hist)
            plot_accuracy(hist)

        return test_ds,preds,model
    
    def get_preds_and_truth(self,model,test_dataset) ->Tuple[List,List]:
        y_true,preds = [],[]

        for batch in test_dataset:
            l,ab = batch
            ab_preds = model.predict(l,verbose=1)
        if (ab is not None) or (ab_preds is not None):
            y_true.append(ab if tf.executing_eagerly() else tf.convert_to_tensor(ab))
            preds.append(ab_preds if isinstance(ab_preds, np.ndarray) else ab_preds.numpy())


        y_true = np.concatenate(y_true,axis=0)
        preds = np.concatenate(preds,axis=0)
        return y_true.tolist(),preds.tolist()

    def evaluate(
            self,
            model:Model,
            test_ds,
            samples_to_show:int = 0,
            find_psnr:bool = True,
            find_ssim:bool = True,
            find_delta_e_76:bool = True,
            find_delta_e_2000:bool = True,
            show_lab_prediction:bool = True,
            ):
        
        print("Starting evaluation...")
        l_channels,y_preds_ab,y_true_ab = [],[],[]
        
        print("Iterating over test dataset...")
        for batch in test_ds:
            l,ab = batch
            preds = model.predict(l)
            if ab is None or preds is None or l is None:
                raise ValueError('AB/Preds/L is Empty')
            l_channels.append(l.numpy())
            y_true_ab.append(ab.numpy())
            y_preds_ab.append(preds)
        
        l_channels = np.concatenate(l_channels,axis=0)
        y_true_ab = np.concatenate(y_true_ab,axis=0)
        y_preds_ab = np.concatenate(y_preds_ab,axis=0)

        print("Converted predictions and ground truth to RGB...")
        results:Dict[str,float] = {}
        img_true = img_utils.lab_to_rgb(l_channels,y_true_ab)
        img_preds = img_utils.lab_to_rgb(l_channels,y_preds_ab)

        print("Predicted image:", img_preds.shape, img_preds.min(), img_preds.max())
        print("Ground truth:", img_true.shape, img_true.min(), img_true.max())

        if find_psnr:
            print("Calculating PSNR...")
            psnr = tf.image.psnr(img_true/255.0,img_preds/255.0,max_val=1.0)
            results['PSNR'] = tf.reduce_mean(psnr).numpy()

            psnr1 = [sk_psnr(y_true_ab[i], y_preds_ab[i], data_range=1.0) for i in range(len(y_true_ab))]
            results['SK_PSNR'] = float(np.mean(psnr1))

        if find_ssim:
            print("Calculating SSIM...")
            ssim = tf.image.ssim(img_true / 255.0, img_preds / 255.0, max_val=1.0)
            results['SSIM'] = tf.reduce_mean(ssim).numpy()

            ssim1 = [sk_ssim(y_true_ab[i], y_preds_ab[i], data_range=255, channel_axis=-1) for i in range(len(y_true_ab))]
            results['SK_SSIM'] = float(np.mean(ssim1))
        
        if find_delta_e_76 or find_delta_e_2000:
            print("Calculating Delta E...")
            delta_true_e =  np.concatenate([l_channels,y_preds_ab])
            delta_preds_e = np.concatenate([l_channels,y_true_ab])

            if find_delta_e_76:
                delta_e_76 = np.mean([deltaE_cie76(t,p) for t,p in zip(delta_true_e,delta_preds_e)])
                results['Delta_E76'] = float(delta_e_76)
            
            if find_delta_e_2000:
                delta_e2000 = np.mean([deltaE_ciede2000(t,p) for t,p in zip(img_true,img_preds)])
                results['Delta_E2000'] = float(delta_e2000)
            
            
            if samples_to_show > 0:
                print(f"Showing {samples_to_show} sample predictions...")
                indices = np.random.choice(len(l_channels), size=samples_to_show, replace=False)
                for idx in indices:
                    l_img = np.squeeze(l_channels[idx])
                    pred_ab_img = y_preds_ab[idx]
                    true_ab_img = y_true_ab[idx]

                    img_utils.show_prediction(l_img,pred_ab_img,true_ab_img,(8,4))

                    if show_lab_prediction:
                        img_utils.show_lab_predictions(l_img,pred_ab_img,true_ab_img)

        print("Evaluation complete.")
        return results
    

import io
import cv2
import keras
import mlflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from skimage import io,color
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import gray2rgb, rgba2rgb, rgb2lab,rgb2lab, lab2rgb
from skimage.transform import resize
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional,Union
from sklearn.model_selection import train_test_split
from keras.layers import RandomRotation,RandomBrightness,RandomFlip,RandomContrast,RandomCrop

from config import config
from logger import get_logger

logger = get_logger(__name__)

class Dataspliter:
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    @classmethod
    def split(cls,X,y,test_size=0.1,shuffle=True):
        if cls.X_train is None:
            cls.X_train,cls.X_test,cls.y_train,cls.y_test = train_test_split(X,y,test_size=test_size,shuffle=shuffle)
            return cls.X_train,cls.X_test,cls.y_train,cls.y_test
        else:
            return cls.X_train,cls.X_test,cls.y_train,cls.y_test

class ImageUtils:
    def __init__(self,target_shape:Tuple[int,int] = (224,224)):
        self.enable_mlflow = config.check_mlflow_server()
        self.target_shape = target_shape

    def load_data(
        self,
        L_path: Path,
        ab_paths: List[Path],
        dataset_size: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not L_path.exists():
            raise FileNotFoundError(f"No gray file at {L_path}")
        L = np.load(L_path, mmap_mode="r").astype(np.float32) / 100.0
        logger.info(f"Loaded raw L shape: {L.shape}")

        if L.ndim == 3:
            L = L[..., None]
        elif L.ndim == 4 and L.shape[-1] == 1:
            L = np.squeeze(L, axis=-1)[..., None]
        else:
            raise ValueError(f"Unexpected L shape: {L.shape}")
        if dataset_size > 0 and L.shape[0] > dataset_size:
            logger.info(f"Trimming L from {L.shape[0]} → {dataset_size}")
            L = L[:dataset_size]


        ab_list = []
        for idx, p in enumerate(ab_paths):
            if not Path(p).exists():
                raise FileNotFoundError(f"AB file not found: {p}")
            ab = np.load(p, mmap_mode="r").astype(np.float32)
            logger.info(f"Loaded raw AB[{idx}] shape: {ab.shape}")

            if ab.ndim == 3:
                ab = ab[..., None]
            elif ab.ndim == 4 and ab.shape[-1] >= 2:
                ab = ab[...,1:3]
            else:
                raise ValueError(f"Unexpected AB[{idx}] shape: {ab.shape}")
            if dataset_size > 0 and ab.shape[0] > dataset_size:
                logger.info(f"Trimming AB[{idx}] from {ab.shape[0]} → {dataset_size}")
                ab = ab[:dataset_size]
            ab_list.append(ab)
        AB = np.concatenate(ab_list, axis=-1)
        logger.info(f"Concatenated AB shape: {AB.shape}")

        if AB.shape[-1] != 2:
            AB = AB[...,:2]
            logger.info(f"Reduced AB to 2 channels: {AB.shape}")
        logger.info(f"AB range before normalization: [{AB.min()}, {AB.max()}]")
        AB = (AB - 128.0) / 128.0
        logger.info(f"AB range after normalization: [{AB.min()}, {AB.max()}]")

        if self.enable_mlflow:
            mlflow.log_param("L_shape", L.shape)
            mlflow.log_param("AB_shape", AB.shape)
        return L, AB
    def get_dataset(self,L,ab_images,shuffle,buffer_size,batch_size):
        logger.info('Preparing For The Creation Dataset')
        logger.info(f'Shape of Received L: {L.shape}')
        logger.info(f'Shape of Received ab_images: {ab_images.shape}')
        
        
        df = tf.data.Dataset.from_tensor_slices((L,ab_images))
        if shuffle:
            df = df.shuffle(buffer_size,reshuffle_each_iteration=True)
            logger.info('Dataset shuffled')
        ds = df.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info('Dataset batched and prefetch added')
        return ds
    
    def augment(
            self,
            dataset:tf.data.Dataset,
            img_size:Tuple[int,int] = (224,224),
            flip:str = 'horizontal_and_vertical',
    )->tf.data.Dataset:
        
        
        aug_layers = keras.Sequential(name='aug_pipeline',layers=[
            RandomBrightness(0.1),
            RandomContrast(0.1),
            RandomFlip(flip),
            RandomRotation(0.1),
            RandomCrop(img_size[0],img_size[1])
        ])
        def _augment(L,AB):
            logger.info('Augmenting Training Data')
            X = tf.concat([L,AB],axis=-1)
            logger.info(f'Concatenated Shape: {tf.shape(X)}')
            X = aug_layers(X)
            logger.info(f'Augmented Shape: {X.shape}')
            L2 = X[...,:1]
            logger.info(f'Augmented L2 Shape: {L2.shape}')
            AB2 = X[...,1:3]
            logger.info(f'Augmented AB2 Shape: {AB2.shape}')
            return L2,AB2
        
        return dataset.map(_augment,num_parallel_calls=tf.data.AUTOTUNE)
    
    def load_dataset(self,
                L_path: Path,
                AB_path: Union[List[Path], Path],
                shuffle_size: int = 0,
                batch_size: int = 32,
                dataset_size: int = 0,
                augment: bool = True,
                shuffle: bool = True,
                use_saved_dataset: bool = False,
                save_data: bool = False,
                ):
        logger.info(f'Loading Dataset From:\nGray: {L_path}\nRGB: {AB_path}')
        if use_saved_dataset and save_data:
            raise ValueError('Both use_saved_dataset and save_data cannot be True simultaneously')
        L = AB = None

        if use_saved_dataset and isinstance(AB_path,Path):
            if not (L_path.exists() and AB_path.exists()):
                raise FileNotFoundError(f'Missing saved L or AB .npy files')
            logger.info(f'Loading saved .npy files...')
            L = np.load(L_path).astype(np.float32) / 100.0
            AB = np.load(AB_path).astype(np.float32)
            AB = (AB - 128.0) / 128.0
        elif isinstance(AB_path, list) and all(isinstance(p, Path) for p in AB_path):
            L, AB = self.load_data(L_path, AB_path, dataset_size)
            if save_data and dataset_size > 0:
                logger.info(f'Saving dataset to: {config.SAVE_DATASET_PATH}')
                np.save(Path(config.SAVE_DATASET_PATH) / 'L_data.npy', L)
                np.save(Path(config.SAVE_DATASET_PATH) / 'ab_data.npy', AB)
        else:
            raise ValueError(f'Invalid path input: AB_path must be Path or List[Path]')
        if L is None or AB is None:
            raise RuntimeError("L and AB must be loaded before splitting")

        logger.info('Splitting data into training, test, and validation sets...')
        X_train, X_test, y_train, y_test = Dataspliter.split(L, AB, shuffle=shuffle)
        X_train, X_val, y_train, y_val = Dataspliter.split(X_train, y_train)
        logger.info('Creating training dataset')
        train_ds = self.get_dataset(X_train, y_train, True, shuffle_size, batch_size)
        if augment:
            logger.info('Applying augmentation to training data')
            train_ds = self.augment(train_ds)
        logger.info('Creating test dataset')
        test_ds = self.get_dataset(X_test, y_test, False, shuffle_size, batch_size)
        logger.info('Creating validation dataset')
        val_ds = self.get_dataset(X_val, y_val, False, shuffle_size, batch_size)
        logger.info('Datasets ready')
        return train_ds, test_ds, val_ds


    def to_multi_output(self, L, AB):
        return L, {
            'decoder_L_out': tf.cast(L, tf.float32),
            'decoder_AB_out': tf.cast(AB, tf.float32)
        }

    def pad_image(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
    
        h, w = image.shape[:2]
        th, tw = target_shape

        if h > th:
            start_h = (h - th) // 2
            image = image[start_h : start_h + th, :, ...]
        if w > tw:
            start_w = (w - tw) // 2
            image = image[:, start_w : start_w + tw, ...]

        pad_h = max(0, th - image.shape[0])
        pad_w = max(0, tw - image.shape[1])
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h or pad_w:
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )

        return image

    def lab_to_rgb(
        self,
        l: np.ndarray,
        ab: np.ndarray,
        debug: bool = False
    ) -> Union[np.ndarray, np.ndarray]:
        l = np.asarray(l, dtype=np.float32)
        ab = np.asarray(ab, dtype=np.float32)

        # Ensure dimensions are 4D for processing
        if l.ndim == 2:  # (H, W)
            l = l[..., np.newaxis]
        if l.ndim == 3 and l.shape[-1] == 1:  # (H, W, 1)
            l = l[np.newaxis, ...]  # (1, H, W, 1)
        if ab.ndim == 3 and ab.shape[-1] == 2:
            ab = ab[np.newaxis, ...]  # (1, H, W, 2)

        if l.ndim != 4 or ab.ndim != 4:
            raise ValueError(f"Expected l and ab to be 4D, got l.shape={l.shape}, ab.shape={ab.shape}")

        if l.shape[0] != ab.shape[0]:
            raise ValueError(f"Batch size mismatch: l={l.shape[0]}, ab={ab.shape[0]}")

        batch_size = l.shape[0]
        rgb_images = []

        for i in range(batch_size):
            l_single = l[i, :, :, 0] * 100.0            # Rescale L to [0,100]
            ab_single = ab[i] * 128.0                   # Rescale AB to [-128,128]

            lab = np.stack([l_single, ab_single[:, :, 0], ab_single[:, :, 1]], axis=-1)  # (H,W,3)

            rgb = lab2rgb(lab)                          # float32 in [0,1]
            rgb = np.clip(rgb, 0, 1)
            rgb_uint8 = (rgb * 255).astype(np.uint8)    # uint8 in [0,255]

            if debug:
                print(f"[Image {i}] L range: {l_single.min():.2f} to {l_single.max():.2f}")
                print(f"[Image {i}] a range: {ab_single[:,:,0].min():.2f} to {ab_single[:,:,0].max():.2f}")
                print(f"[Image {i}] b range: {ab_single[:,:,1].min():.2f} to {ab_single[:,:,1].max():.2f}")

            rgb_images.append(rgb_uint8)

        return rgb_images[0] if batch_size == 1 else np.stack(rgb_images)


    def show_prediction(
            self,
            l:np.ndarray,
            ab_preds:np.ndarray,
            ab_true:np.ndarray,
            fig_size:Tuple[int,int]
    ) ->None:
        
        logger.info(f'Visualizing Predictions L shape: {l.shape}, Predicted_AB shape: {ab_preds.shape}, True_AB shape: {ab_true.shape}')

        # if ab_preds.ndim == 4:
        #     logger.info(f"AB_preds has batch dimension. Original shape: {ab_preds.shape}")
        #     ab_preds = ab_preds[0]
        #     logger.info(f"Removed batch dimension. New shape: {ab_preds.shape}")

        preds_rgb = self.lab_to_rgb(l,ab_preds)
        true_rgb = self.lab_to_rgb(l,ab_true)

        n_cols = 2 + (1 if ab_true is not None else 0)
        plt.figure(figsize=fig_size)
        
        
        if ab_true is not None:
            plt.subplot(1,n_cols,1)
            plt.imshow(true_rgb)
            plt.title('Actual Image')
            plt.axis('off')
        
        plt.subplot(1,n_cols,2)
        plt.imshow(preds_rgb)
        plt.title('Predicted Image')
        plt.axis('off')
        
        plt.subplot(1,n_cols,3)
        plt.imshow(np.squeeze(l),cmap='gray')
        plt.title('GrayScale Input')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def show_lab_predictions(self,l: np.ndarray, ab_preds: np.ndarray, ab_true: Optional[np.ndarray] = None):
        if l.ndim == 2:
            l = l[..., np.newaxis]
        elif l.ndim == 3 and l.shape[-1] != 1:
            raise ValueError("L channel must be single channel (H, W) or (H, W, 1)")

        if ab_preds.ndim != 3 or ab_preds.shape[-1] != 2:
            raise ValueError("ab_preds must be (H, W, 2)")

        # Denormalize Lab components
        L_scaled = l[..., 0] * 100.0  # L channel in [0, 100]
        ab_scaled = ab_preds * 127.0  # ab channels in [-128, 128]
        lab_pred = np.concatenate([L_scaled[..., np.newaxis], ab_scaled], axis=-1)

        rgb_pred = lab2rgb(lab_pred)

        # Show grayscale and prediction
        images = [np.squeeze(l), rgb_pred]
        titles = ["Grayscale (L)", "Predicted RGB"]

        if ab_true is not None:
            ab_true_scaled = ab_true * 127.0
            lab_true = np.concatenate([L_scaled[..., np.newaxis], ab_true_scaled], axis=-1)
            rgb_true = lab2rgb(lab_true)
            images.append(rgb_true)
            titles.append("Ground Truth RGB")

        # Plotting
        plt.figure(figsize=(8, 5))
        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(np.clip(img, 0, 1))
            plt.axis('off')
            plt.title(title)
        plt.tight_layout()
        plt.show()
    

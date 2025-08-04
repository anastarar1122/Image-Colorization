import os
import keras
import tensorflow as tf


from keras import layers, models
from typing import Literal,Tuple,List,cast
from keras.applications import VGG16,EfficientNetB0,MobileNetV2,ResNet50

from utils.model_utils import ModelUtils
from utils.image_utils import ImageUtils
from logger import get_logger
from config import config
utils = ImageUtils()

logger = get_logger(__name__)

class ModelBuilder:
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 1),
        output_size:Tuple[int,int] = (256,256),
        output_channels: int = 2,
        mlp_dim: int = 512,
        dropout_rate: float = 0.1,
        
        
    ):
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_channels = output_channels
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.model_utils = ModelUtils(
            self.input_shape,
            self.dropout_rate,
            self.mlp_dim
        )
        self.validate_input()

    def validate_input(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError(f"Expected 3D input_shape, got {self.input_shape}")
        if self.input_shape[2] != 1:
            tf.get_logger().warning(f"Expected grayscale input with 1 channel, got {self.input_shape[2]}")
        if self.output_channels < 1:
            tf.get_logger().warning(f"Output channels must be positive, got {self.output_channels}")

    

    def build_mlp_style(
            self,
            use_vit:bool = True,
            patch_size:int = 16,
            num_heads:int = 8,
            transformer_blocks:int=6,
    ):
        
        inputs = tf.convert_to_tensor(layers.Input(shape=self.input_shape,name='input_grayscale'))
        X,num_patches = self.model_utils.patch_embedding(
            patch_size=patch_size,
            embed_dim=config.EMBED_DIM,
            inputs=inputs
        )
        pos_encoding = self.model_utils.position_encoding(num_patches,config.EMBED_DIM)
        X = X + pos_encoding

        for _ in range(transformer_blocks):
            X = self.model_utils.transformer_encoder(X,num_heads)
        

        X = layers.Dense(
            self.input_shape[0] * self.input_shape[1] * self.output_channels,
            name='output_dense'
        )(X)

        X = layers.Reshape((self.input_shape[0],self.input_shape[1],self.output_channels))(X)

        if not use_vit:
            X = layers.Conv2D(128,3,padding='same',activation='swish')(X)
            X = layers.Dropout(self.dropout_rate)(X)
            X = layers.Conv2DTranspose(64,4,strides=2,padding='same',activation='swish')(X)
            outputs = layers.Conv2D(self.output_channels,1,activation='sigmoid',name='hybrid_output')(X)
            model_name = 'HybridCNNTransformerColorizer'
        else:
            outputs = layers.Activation('sigmoid',name='vit_output')(X)
            model_name = 'ViTColorizer'
        
        model = models.Model(inputs,outputs,name=model_name)
        return model
    def build_multihead_model(
            self,
            architecture:Literal['simple','unet'] = 'unet'
    ) -> keras.Model:
        
        inputs = layers.Input(shape=self.input_shape,name='L_input')
        b,c = self.model_utils.build_encoder(inputs)
        decode_filters = [128,32]

        l_recon = self.model_utils.build_decoder(b,c,decode_filters,'decoder_L',architecture,final_activation='sigmoid')
        L = layers.Lambda(lambda t: tf.image.resize(t,(224,224)),name='decoder_L_out')(l_recon)

        ab_preds = self.model_utils.build_decoder(b,c,decode_filters,'decoder_AB',architecture,final_activation='tanh')
        AB = layers.Lambda(lambda t: tf.image.resize(t,(224,224)),name='decoder_AB_out')(ab_preds)
        
        return models.Model(inputs,outputs={'decoder_L_out':L,'decoder_AB_out':AB},name=f'Multihead_{architecture}_{self.output_size[0]}x{self.output_size[1]}')
    
    def build_pretrained(
            self,
            base:Literal['vgg16', 'resnet50', 'mobilenetv2', 'efficientnetb0'] = 'efficientnetb0',
            fine_tune_layers:int = 0,
            use_unet:bool = True
    ) ->models.Model:
        inputs = layers.Input(shape=self.input_shape,name=f'L_input')
        X = layers.Concatenate()([inputs,inputs,inputs])

        backbone_dict = {
            'vgg16':VGG16,
            'resnet50':ResNet50,
            'mobilenetv2':MobileNetV2,
            'efficientnetb0':EfficientNetB0
        }

        if base not in backbone_dict:
            raise ValueError(f'UnSupported Base Model: {base}')
        
        backbone = backbone_dict[base](include_top=False,weights='imagenet',input_shape=[self.input_shape[0],self.input_shape[1],3])
        if fine_tune_layers > 0:
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False
        
        X = backbone(X)

        if use_unet:
            u1 = self.model_utils._conv_block(X,512,0.3)
            c1 = layers.UpSampling2D()(u1) 

            u2 = self.model_utils._conv_block(c1,256,0.2)
            c2 = layers.UpSampling2D()(u2)

            u3 = self.model_utils._conv_block(c2,128,0.1)
            c3 = layers.UpSampling2D()(u3)

            u4 = self.model_utils._conv_block(c3,64,0.1)
            c4 = layers.UpSampling2D()(u4)

            u5 = self.model_utils._conv_block(c4,32)
            c5 = layers.UpSampling2D()(u5)

            u6 = self.model_utils._conv_block(c5,32)
            final_feat = u6

        else:
            c1 = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='swish')(X)
            u1 = self.model_utils._conv_block(c1, 512)

            c2 = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='swish')(u1)
            u2 = self.model_utils._conv_block(c2,256)

            c3 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='swish')(u2)
            u3 = self.model_utils._conv_block(c3, 128)

            c4 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='swish')(u3)
            u4 = self.model_utils._conv_block(c4, 64)
            
            c5 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='swish')(u4)
            u5 = self.model_utils._conv_block(c5, 32)
            final_feat = u5
        
        outputs = layers.Conv2D(
            filters=2,
            kernel_size=1,
            padding='same',
            activation='tanh'
            
        )(final_feat)

        return models.Model(inputs, outputs, name=f'pretrained_{base}')

    def build_demo_multihead_model(self, architecture: Literal['simple', 'unet'] = 'unet') -> keras.Model:
        inputs = layers.Input(shape=self.input_shape, name='L_input')
        b, c = self.model_utils.build_demo_encoder(inputs)
        decode_filters = [64, 32]  # Fewer filters for demo model

        l_recon = self.model_utils.build_demo_decoder(b, c, decode_filters, 'decoder_L', architecture, final_activation='sigmoid')
        L = layers.Lambda(lambda t: tf.image.resize(t, (224, 224)), name='decoder_L_out')(l_recon)

        ab_preds = self.model_utils.build_demo_decoder(b, c, decode_filters, 'decoder_AB', architecture, final_activation='tanh')
        AB = layers.Lambda(lambda t: tf.image.resize(t, (224, 224)), name='decoder_AB_out')(ab_preds)

        return models.Model(inputs, outputs={'decoder_L_out': L, 'decoder_AB_out': AB}, name=f'Multihead_{architecture}')

    def build_demo_pretrained(self, 
                        base: str = 'efficientnetb0',
                        fine_tune_layers: int = 0,
                        use_unet: bool = True) -> models.Model:
        
        inputs = layers.Input(shape=self.input_shape, name='L_input')  # Input layer (L channel)
        X = inputs  # No concatenation, directly use the L channel input

        backbone_dict = {
            'vgg16': VGG16,
            'resnet50': ResNet50,
            'mobilenetv2': MobileNetV2,
            'efficientnetb0': EfficientNetB0
        }

        if base not in backbone_dict:
            raise ValueError(f'Unsupported Base Model: {base}')
        
        backbone = backbone_dict[base](include_top=False, weights='imagenet', 
                                    input_shape=[self.input_shape[0], self.input_shape[1], 3])  # Pretrained backbone
        if fine_tune_layers > 0:
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False
        
        X = backbone(X)

        if use_unet:
            u1 = self.model_utils._conv_block(X, 256, 0.3)
            c1 = layers.UpSampling2D()(u1)
            u2 = self.model_utils._conv_block(c1, 128, 0.2)
            c2 = layers.UpSampling2D()(u2)
            u3 = self.model_utils._conv_block(c2, 64, 0.1)
            c3 = layers.UpSampling2D()(u3)
            u4 = self.model_utils._conv_block(c3, 32, 0.1)
            c4 = layers.UpSampling2D()(u4)
            final_feat = self.model_utils._conv_block(c4, 32)

        else:
            c1 = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='swish')(X)
            u1 = self.model_utils._conv_block(c1, 256)
            c2 = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='swish')(u1)
            u2 = self.model_utils._conv_block(c2, 128)
            c3 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='swish')(u2)
            u3 = self.model_utils._conv_block(c3, 64)
            final_feat = u3
        
        
        outputs = layers.Conv2D(filters=2, kernel_size=1, padding='same', activation='tanh')(final_feat)

        return models.Model(inputs, outputs, name=f'pretrained_{base}')
    

    def demo_unet(self) -> models.Model:


        inputs = layers.Input(shape=self.input_shape,name='grayscale_input')
        inputs = cast(keras.KerasTensor,inputs)

        c1 = self.model_utils._conv_block(inputs,64)
        p1 = layers.MaxPooling2D()(c1)

        c2 = self.model_utils._conv_block(p1, 128)
        p2 = layers.MaxPooling2D()(c2)

        c3 = self.model_utils._conv_block(p2, 256)
        p3 = layers.MaxPooling2D()(c3)

        c4 = self.model_utils._conv_block(p3, 512)
        p4 = layers.MaxPooling2D()(c4)

        bn = self.model_utils._conv_block(p4, 1024)

        u1 = self.model_utils.upsample_decoder(bn, c4, 512)
        d1 = self.model_utils._conv_block(u1,512,0.1)

        u2 = self.model_utils.upsample_decoder(d1, c3, 256)
        d2 = self.model_utils._conv_block(u2,256,0.1)
        
        u3 = self.model_utils.upsample_decoder(d2, c2, 128)
        d3 = self.model_utils._conv_block(u3,128,0.1)
        
        u4 = self.model_utils.upsample_decoder(d3, c1, 64)
        d4 = self.model_utils._conv_block(u4,64,0.1)

        outputs = layers.Conv2D(2, 1, activation='tanh')(d4)

        model = models.Model(inputs, outputs, name='U-Net-Colorizer')
        return model



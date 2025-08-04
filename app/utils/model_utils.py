import keras
import tensorflow as tf

from keras import layers
from typing import Tuple,Any,List,Literal

from logger import get_logger
logger = get_logger(__name__)



class ModelUtils:
    def __init__(
            self,
            input_shape:Tuple[int,int,int] = (224,224,1),
            dropout_rate:float=0.1,
            mlp_dim:int = 2048,
    ) -> None:
        
        self.input_shape = input_shape
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
    
    def patch_embedding(
            self,
            patch_size:int,
            embed_dim:int,
            inputs:tf.Tensor
    ) ->Tuple[keras.KerasTensor,int]:
        patch = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='input_embedding'
        )(inputs)
        num_patches = (self.input_shape[0] // patch_size) * (self.input_shape[1] // patch_size)
        X = layers.Reshape((num_patches,embed_dim))(patch)
        return X,num_patches
    
    def position_encoding(
            self,
            num_patches:int,
            embed_dim:int,
    ) ->keras.KerasTensor:
        position = tf.range(start=0,limit=num_patches,delta=1)
        return layers.Embedding(input_dim=num_patches,output_dim=embed_dim)(position)
    
    def transformer_encoder(
            self,
            X:Any,
            head_nums:int,
    ) ->keras.KerasTensor:
        attn_output = layers.MultiHeadAttention(
            num_heads=head_nums,
            key_dim=self.input_shape[-1],
            dropout=self.dropout_rate
        )(X,X)

        X = layers.Add()([X,attn_output])
        X = layers.LayerNormalization(epsilon=1e-6)(X)

        mlp = layers.Dense(self.mlp_dim,'gelu')(X)
        mlp = layers.Dropout(self.dropout_rate)(mlp)
        mlp = layers.Dense(X.shape[-1])(mlp)

        X = layers.Add()([X,mlp])
        X = layers.LayerNormalization(epsilon=1e-6)(X)
        return X
    
    def _conv_block(
        self,
        x:keras.KerasTensor,
        filters: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
        strides: int = 1,
        activation: str = 'swish',
        use_bn: bool = True
    ) -> keras.KerasTensor:
        x = layers.Conv2D(filters, kernel_size, strides, padding='same')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        return x
    
    def upsample_decoder(self, X, skip, filters):
        X = layers.Conv2DTranspose(filters, 2, 2, padding='same')(X)
        return layers.Concatenate()([X,skip])
    
    def build_encoder(self,inputs) -> Tuple[keras.KerasTensor,List[keras.KerasTensor]]:

        c1 = self._conv_block(inputs,32,0.1)
        c1 = self._conv_block(c1,32)
        p1 = layers.MaxPooling2D(2)(c1) #112

        c2 = self._conv_block(p1,64,0.1)
        c2 = self._conv_block(c2,64)
        p2 = layers.MaxPooling2D(2)(c2) #56

        b = self._conv_block(p1,128,0.3)
        b = self._conv_block(b,128)

        c3 = self._conv_block(p2,256,0.2)
        c3 = self._conv_block(c3,256)
        p3 = layers.MaxPooling2D(2)(c3) #28

        c4 = self._conv_block(p3,512,0.2)
        c4 = self._conv_block(c4,512)
        p4 = layers.MaxPooling2D(2)(c4) #14

        b = self._conv_block(p4,1024,0.3)
        b = self._conv_block(b,1024)
        return b,[c1,c2,c3,c4]


    
    def build_decoder(
            self,
            bottel_neck:keras.KerasTensor,
            skips:List[keras.KerasTensor],
            filter_list:List[int],
            head_name:Literal['decoder_L','decoder_AB'] = 'decoder_L',
            architecture:Literal['simple','unet'] = 'unet',
            final_activation:Literal['sigmoid','tanh'] = 'sigmoid'
    ) ->keras.KerasTensor:
        
        X = bottel_neck

        if architecture == 'unet':
            for i,(f, skip) in enumerate(zip(filter_list,reversed(skips))):
                if i < len(skips):
                    X = self.upsample_decoder(X,skips[-(i + 1)],f)
                    X = self._conv_block(X,f)
                    # X = self._conv_block(X,f)
                else:
                    X = self.upsample_decoder(X,skip,f)
                    X = self._conv_block(X,f)
                    # X = self._conv_block(X,f)
        elif architecture == 'simple':
            for i,(f,skip) in enumerate(zip(filter_list,reversed(skips))):
                if i < len(skips):
                    X = self.upsample_decoder(X,skips[-(i + 1)],f)
                X = self.upsample_decoder(X,skip,f)
        
        X = layers.Conv2D(
            filters=1 if head_name == 'decoder_L' else 2,
            kernel_size=1,
            padding='same',
            activation='sigmoid' if head_name == 'decoder_L' else 'tanh',
            name=head_name
        )(X)
        return X




    def build_demo_encoder(self, inputs) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        c1 = self._conv_block(inputs, 16, 0.1)
        c1 = self._conv_block(c1, 16)
        p1 = layers.MaxPooling2D(2)(c1)  # 112

        c2 = self._conv_block(p1, 32, 0.1)
        c2 = self._conv_block(c2, 32)
        p2 = layers.MaxPooling2D(2)(c2)  # 56

        b = self._conv_block(p1, 64, 0.3)
        b = self._conv_block(b, 64)

        c3 = self._conv_block(p2, 128, 0.2)
        c3 = self._conv_block(c3, 128)
        p3 = layers.MaxPooling2D(2)(c3)  # 28

        c4 = self._conv_block(p3, 256, 0.2)
        c4 = self._conv_block(c4, 256)
        p4 = layers.MaxPooling2D(2)(c4)  # 14

        b = self._conv_block(p4, 512, 0.3)
        b = self._conv_block(b, 512)
        return b, [c1, c2, c3, c4]

    def build_demo_decoder(
        self,
        bottleneck: keras.KerasTensor,
        skips: List[keras.KerasTensor],
        filter_list: List[int],
        head_name: Literal['decoder_L', 'decoder_AB'] = 'decoder_L',
        architecture: Literal['simple', 'unet'] = 'unet',
        final_activation: Literal['sigmoid', 'tanh'] = 'sigmoid'
    ) -> keras.KerasTensor:
        X = bottleneck

        if architecture == 'unet':
            for i, (f, skip) in enumerate(zip(filter_list, reversed(skips))):
                X = self.upsample_decoder(X, skips[-(i + 1)], f)
                X = self._conv_block(X, f)
        elif architecture == 'simple':
            for i, (f, skip) in enumerate(zip(filter_list, reversed(skips))):
                X = self.upsample_decoder(X, skip, f)

        X = layers.Conv2D(
            filters=1 if head_name == 'decoder_L' else 2,
            kernel_size=1,
            padding='same',
            activation='sigmoid' if head_name == 'decoder_L' else 'tanh',
            name=head_name
        )(X)
        return X





import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras import backend as K


def abs_backend(inputs):
    return K.abs(inputs)

def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs, 1), 1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    if in_channels >= out_channels:
        # No padding needed, or consider reducing channels via 1x1 Conv
        return inputs  # or apply a 1x1 convolution here to adjust channels
    else:
        pad_dim = (out_channels - in_channels) // 2
        padded = tf.pad(inputs, paddings=[[0, 0], [0, 0], [0, 0], [pad_dim, pad_dim]], mode='CONSTANT')
        return padded

# Residual Shrinkage Block
def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
    residual = incoming
    in_channels = incoming.shape[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides), 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = tf.keras.layers.Multiply()([abs_mean, scales])
        
        # Soft thresholding
        sub = tf.keras.layers.Subtract()([residual_abs, thres])
        zeros = tf.keras.layers.Subtract()([sub, sub])
        n_sub = tf.keras.layers.Maximum()([sub, zeros])
        residual = tf.keras.layers.Multiply()([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pool-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1, 1), strides=(2, 2))(identity)
            
        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(identity)
        
        residual = tf.keras.layers.Add()([residual, identity])
    
    return residual

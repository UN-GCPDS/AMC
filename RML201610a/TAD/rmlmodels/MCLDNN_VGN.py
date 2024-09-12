import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Add, Activation, BatchNormalization, ZeroPadding2D, Reshape, GlobalAveragePooling2D, Lambda,Layer, GaussianDropout,GRU
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import tensorflow as tf

import tensorflow as tf
from RFFfunctions import *
from Denoiser import *

class ThresholdDenoisingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ThresholdDenoisingLayer, self).__init__()
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(16, activation='sigmoid')

    def call(self, inputs):
        # Extraer características promedio de cada mapa de características
        beta = self.global_average_pooling(inputs)

        # Procesamiento adicional para calcular el umbral
        beta = self.dense1(beta)
        beta = self.bn(beta)
        beta = self.dense2(beta)

        # Expansión de beta para coincidir con las dimensiones espaciales de inputs
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)

        # Cálculo de la media de beta a lo largo de todas las dimensiones, resultando en un escalar
        beta_mean = tf.reduce_mean(beta)  # Elimina keepdims para obtener un escalar

        # Cálculo de tau como el doble del valor escalar de beta_mean
        tau = 2 * beta_mean

        # Aplicación de la función de desruido
        outputs = self.denoise_function(inputs, tau)
        return outputs

    def denoise_function(self, inputs, tau):
        # Cálculo del signo de inputs
        sign_x = tf.sign(inputs)

        # Cálculo de inputs ajustados
        adjusted_inputs = tf.abs(inputs) - tau

        # Aplicación del umbral usando tf.where, elemento por elemento
        return tf.where(tf.abs(inputs) > tau, sign_x * adjusted_inputs, tf.zeros_like(inputs))



    
def MCLDNN(input_shape=[2, 128], classes=11):
    inputs = Input(shape=(input_shape[0], input_shape[1], 1))
    
    # Apply batch normalization
    x = tf.keras.layers.BatchNormalization()(inputs)

    # Apply first convolutional layer with kernel regularizer
    x = ConvRFF_SinCos(8, (2, 3), activation='relu', padding='SAME', trainable_W=False,trainable_scale=True)(x)
   

    # Apply auto threshold denoising layer if necessary
    # x = ThresholdDenoisingLayer()(x)

    # This could be your custom layer or method as defined elsewhere
    net = residual_shrinkage_block(x, 1, 16, downsample=False)
    
    # Apply second convolutional layer with kernel regularizer
    x = Conv2D(16, (2, 3), activation='relu', padding='SAME')(net)
   
   
    # Apply third convolutional layer with kernel regularizer
    x = Conv2D(32, (2, 3), activation='relu', padding='VALID')(x)
   
   
    # Reshape the data to fit GRU layer
    x = Reshape((126, 32))(x)
    x = GaussianDropout(0.3)(x)  # Apply Gaussian Dropout
    # Apply GRU layer
    x = GRU(64)(x)

    # Final fully connected layer with softmax activation for classification
    outputs = Dense(11, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

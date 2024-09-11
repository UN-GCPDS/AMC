import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax, BatchNormalization, TimeDistributed
from tensorflow.keras.layers import LSTM, Bidirectional, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model

def DAE(weights=None, input_shape=[128, 2], classes=11, **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    # Input layer
    inputs = Input(shape=input_shape, name='input')
    x = inputs
    dr = 0.0  # Dropout rate
    
    # LSTM Unit
    x, state_h, state_c = LSTM(units=32, return_state=True, return_sequences=True)(x)
    x = Dropout(dr)(x)
    x, state_h1, state_c1 = LSTM(units=32, return_state=True, return_sequences=True)(x)

    # Classifier
    xc = Dense(32, activation='relu')(state_h1)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(16, activation='relu')(xc)
    xc = BatchNormalization()(xc)
    xc = Dropout(dr)(xc)
    xc = Dense(classes, activation='softmax', name='xc')(xc)

    # Decoder
    xd = TimeDistributed(Dense(2), name='xd')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=[xc, xd])

    # Load weights, if provided
    if weights is not None:
        model.load_weights(weights)

    return model

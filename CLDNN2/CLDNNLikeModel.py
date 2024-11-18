import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, LSTM, Reshape, Activation
from tensorflow.keras.optimizers import Adam

def CLDNNLikeModel(weights=None,
                   input_shape1=[2, 128],
                   classes=11,
                   **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5
    input_x = Input(shape=input_shape1 + [1], name='input')

    # Convolutional layers
    x = Conv2D(256, (1, 3), activation="relu", name="conv1", kernel_initializer='glorot_uniform')(input_x)
    x = Dropout(dr)(x)
    x = Conv2D(256, (2, 3), activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv3", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    x = Dropout(dr)(x)

    # Reshaping the convolution output for the LSTM layer
    x1 = Reshape((80, 120))(x)

    # LSTM layer
    lstm_out = LSTM(units=50, name="lstm")(x1)

    # Fully connected layers
    x = Dense(128, activation='relu', name="dense1")(lstm_out)
    x = Dropout(dr)(x)
    output = Dense(classes, activation='softmax', name="dense2")(x)

    # Model definition
    model = Model(inputs=input_x, outputs=output)

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = CLDNNLikeModel(weights=None, input_shape1=(2, 128), classes=11)

    # Optimizer
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Model compilation
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # Model summary and configuration
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    model.summary()

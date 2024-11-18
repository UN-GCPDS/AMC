# Import all the necessary libraries
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

def LSTMModel(weights=None,
              input_shape=(128, 2),
              classes=11,
              **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input = Input(shape=input_shape, name='input')
    # LSTM Unit
    x = LSTM(units=128, return_sequences=True)(input)
    x = LSTM(units=128)(x)

    # DNN
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input, outputs=x)

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = LSTMModel(None, input_shape=(128, 2), classes=11)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:', model.summary())

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape, LSTM, Conv2D
from tensorflow.keras.optimizers import Adam

def MCLDNN(weights=None,
           input_shape1=[2, 128],
           input_shape2=[128, 1],
           classes=11,
           **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    tap = 8
    input1 = Input(input_shape1 + [1], name='input1')
    input2 = Input(input_shape2, name='input2')
    input3 = Input(input_shape2, name='input3')

    # Separate Channel Combined Convolutional Neural Networks
    x1 = Conv2D(50, (2, tap), padding='same', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(input1)
    x2 = Conv1D(50, tap, padding='causal', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(input2)
    x2_reshape = Reshape([-1, 128, 50])(x2)
    x3 = Conv1D(50, tap, padding='causal', activation="relu", name="conv1_3", kernel_initializer='glorot_uniform')(input3)
    x3_reshape = Reshape([-1, 128, 50], name='reshap2')(x3)
    
    x = concatenate([x2_reshape, x3_reshape], axis=1)
    x = Conv2D(50, (1, tap), padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = concatenate([x1, x])
    x = Conv2D(100, (2, 5), padding='valid', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)

    # LSTM Unit
    x = Reshape(target_shape=(124, 100), name='reshape2')(x)
    x = LSTM(units=128, return_sequences=True)(x)
    x = LSTM(units=128)(x)

    # DNN
    x = Dense(128, activation='selu', name='fc1')(x)
    x = Dropout(dr)(x)
    x = Dense(128, activation='selu', name='fc2')(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=[input1, input2, input3], outputs=x)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = MCLDNN(None, classes=10)

    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    model.summary()

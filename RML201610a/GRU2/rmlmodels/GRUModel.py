import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPool1D, ReLU, Dropout, Softmax
from tensorflow.keras.layers import Bidirectional, Flatten, GRU
from tensorflow.keras.utils import plot_model

def GRUModel(weights=None,
             input_shape=[128, 2],
             classes=11,
             **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    input_layer = Input(shape=input_shape, name='input')
    x = input_layer

    # GRU Layer
    x = GRU(units=128, return_sequences=True)(x)
    x = GRU(units=128)(x)

    # Fully connected layer
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    # Create the model
    model = GRUModel(None, input_shape=(128, 2), classes=11)

    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # Plot model
    plot_model(model, to_file='model.png', show_shapes=True)

    # Display model details
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:', model.summary())

"""
CLDNNLike model for RadioML.

# Reference:
- [CONVOLUTIONAL,LONG SHORT-TERM MEMORY, FULLY CONNECTED DEEP NEURAL NETWORKS]
Adapted from code contributed by Mika.
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU, Dropout, Softmax, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

def CNN2(weights=None, input_shape=[2, 128], classes=11, **kwargs):
    # Check if the weights file exists
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    # Define the input layer
    input_layer = Input(input_shape + [1], name='input')
    
    # Convolutional Layers
    x = Conv2D(256, (2, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')(input_layer)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(128, (2, 8), padding='same', activation="relu", name="conv2", kernel_initializer='glorot_uniform')(x)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64, (2, 8), padding='same', activation="relu", name="conv3", kernel_initializer='glorot_uniform')(x)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(64, (2, 8), padding='same', activation="relu", name="conv4", kernel_initializer='glorot_uniform')(x)
    x = MaxPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.5)(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='dense1')(x)
    output = Dense(classes, activation='softmax', name='dense2')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model

# Main execution block
if __name__ == '__main__':
    model = CNN2(weights=None, input_shape=[2, 128], classes=11)

    # Compile the model with Adam optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # Print model details
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:')
    model.summary()

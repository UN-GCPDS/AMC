import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Conv2D, Activation
from tensorflow.keras.initializers import glorot_uniform, he_normal

# Build VT-CNN2 Neural Net model using Keras primitives
#  - Reshape [N,2,128] to [N,2,128,1] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross-entropy optimization

def CNN2Model(weights=None, input_shape=[2, 128], classes=11, **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    dr = 0.5  # dropout rate (%)
    model = Sequential()
    
    # Reshape layer
    model.add(Reshape(input_shape + [1], input_shape=input_shape))
    
    # Convolutional layers
    model.add(Conv2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer=glorot_uniform()))
    model.add(Dropout(dr))
    model.add(Conv2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer=glorot_uniform()))
    model.add(Dropout(dr))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer=he_normal(), name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(classes, kernel_initializer=he_normal(), name="dense2"))
    model.add(Activation('softmax'))
    
    # Load weights
    if weights is not None:
        model.load_weights(weights)
    
    return model

if __name__ == '__main__':
    print(CNN2Model().summary())

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, concatenate, Activation, Flatten

def DenseNet(weights=None,
             input_shape=[2, 128],
             classes=11,
             **kwargs):
    if weights is not None and not (os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.6  # Dropout rate
    input = Input(input_shape + [1], name='input')  # Adding a channel dimension
    x = Conv2D(256, (1, 3), activation="relu", name="conv1", padding='same')(input)
    x1 = Conv2D(256, (2, 3), name="conv2", padding='same')(x)
    x2 = concatenate([x, x1])
    x2 = Activation('relu')(x2)
    x3 = Conv2D(80, (1, 3), name="conv3", padding='same')(x2)
    x4 = concatenate([x, x1, x3])
    x4 = Activation('relu')(x4)
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", padding='same')(x4)
    x = Dropout(dr)(x)
    x = Flatten()(x)

    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dr)(x)
    x = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input, outputs=x)

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = DenseNet(None, input_shape=[2, 128], classes=11)

    # Optimizer
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    # Display model information
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:')
    model.summary()

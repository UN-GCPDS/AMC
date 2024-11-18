import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Conv2D, Flatten, Add

def ResNet(weights=None,
            input_shape=[2, 128],
            classes=11,
            **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.6
    input = Input(shape=input_shape + [1], name='input')

    x = Conv2D(256, (1, 3), name="conv1", kernel_initializer='glorot_uniform', padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(256, (2, 3), name="conv2", kernel_initializer='glorot_uniform', padding='same')(x)
    x1 = Add()([input, x])
    x1 = Activation('relu')(x1)
    x = Conv2D(80, (1, 3), activation="relu", name="conv3", kernel_initializer='glorot_uniform', padding='same')(x1)
    x = Conv2D(80, (1, 3), activation="relu", name="conv4", kernel_initializer='glorot_uniform', padding='same')(x)
    x = Dropout(dr)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dr)(x)
    output = Dense(classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input, outputs=output)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = ResNet(None, input_shape=[2, 128], classes=11)

    # Update optimizer creation
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:', model.summary())

import os
import tensorflow as tf
import math

def cal1(x):
    return tf.keras.backend.cos(x)

def cal2(x):
    return tf.keras.backend.sin(x)

def PETCGDNN(weights=None,
             input_shape=[128, 2],
             input_shape2=[128],
             classes=11,
             **kwargs):
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')

    dr = 0.5  # dropout rate (%)
    
    input1 = tf.keras.Input(shape=input_shape + [1], name='input1')
    input2 = tf.keras.Input(shape=input_shape2, name='input2')
    input3 = tf.keras.Input(shape=input_shape2, name='input3')

    x1 = tf.keras.layers.Flatten()(input1)
    x1 = tf.keras.layers.Dense(1, name='fc2')(x1)
    x1 = tf.keras.layers.Activation('linear')(x1)

    cos1 = tf.keras.layers.Lambda(cal1)(x1)
    sin1 = tf.keras.layers.Lambda(cal2)(x1)
    
    x11 = tf.keras.layers.Multiply()([input2, cos1])
    x12 = tf.keras.layers.Multiply()([input3, sin1])
    x21 = tf.keras.layers.Multiply()([input3, cos1])
    x22 = tf.keras.layers.Multiply()([input2, sin1])
    
    y1 = tf.keras.layers.Add()([x11, x12])
    y2 = tf.keras.layers.Subtract()([x21, x22])
    
    y1 = tf.keras.layers.Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = tf.keras.layers.Reshape(target_shape=(128, 1), name='reshape2')(y2)
    
    x11 = tf.keras.layers.Concatenate(axis=-1)([y1, y2])
    x3 = tf.keras.layers.Reshape(target_shape=(128, 2, 1), name='reshape3')(x11)

    # Spatial feature
    x3 = tf.keras.layers.Conv2D(75, (8, 2), padding='valid', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(x3)
    x3 = tf.keras.layers.Conv2D(25, (5, 1), padding='valid', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(x3)

    # Temporal feature
    x4 = tf.keras.layers.Reshape(target_shape=(117, 25), name='reshape4')(x3)
    x4 = tf.keras.layers.GRU(units=128)(x4)

    x = tf.keras.layers.Dense(classes, activation='softmax', name='softmax')(x4)

    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=x)

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    model = PETCGDNN(None, classes=10)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    
    # Plot model architecture
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    
    print('Model layers:', model.layers)
    print('Model config:', model.get_config())
    print('Model summary:', model.summary())

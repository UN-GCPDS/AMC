import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import sys
import h5py
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.utils import to_categorical
import mltools, dataset2016
import rmlmodels.MCLDNN_VGN as mcl
import csv

# Set up TensorFlow backend and CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Helper functions
def rotate_matrix(theta):
    m = np.zeros((2, 2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    N, L, C = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3 * np.pi / 2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))

    y_DA = np.tile(y, (1, 4)).T.reshape(-1)
    return x_DA, y_DA

def output_layer(model, flag=None):
    if flag == 'conv2':
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv2').output)
    elif flag == 'conv4':
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv4').output)
    elif flag == 'cu_dnnlstm_2':
        model = Model(inputs=model.inputs, outputs=model.get_layer('cu_dnnlstm_2').output)
    return model

# Load and preprocess data
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = dataset2016.load_data()

# Data augmentation with rotation
X_train = X_train.transpose((0, 2, 1))
Y_train = np.argmax(Y_train, axis=1)
X_train, Y_train = Rotate_DA(X_train, Y_train)
X_train = X_train.transpose((0, 2, 1))
Y_train = to_categorical(Y_train)

X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)

# Save test and label data
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)
np.save("lbl.npy", lbl)
np.save("test_idx.npy", test_idx)
np.save("snrs.npy", snrs)

# Model setup
model = mcl.MCLDNN()
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

model.summary()

# Training parameters
nb_epoch = 500
batch_size = 1024
filepath = 'weights/weights.keras'

# Train the model
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2,
                    validation_data=(X_val, Y_val),
                    callbacks=[
                        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                    ])

# Save training history
with open('weights/model_history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

# Show performance
try:
    mltools.show_history(history)
except:
    print("Error showing result")

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

# Prediction function
def predict(model):
    model.load_weights(filepath)
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, mods)
    mltools.plot_confusion_matrix(confnorm, labels=mods, save_filename='figure/mclstm_total_confusion.pdf')
    
    acc = {}
    acc_mod_snr = np.zeros((len(mods), len(snrs)))

    for i, snr in enumerate(snrs):
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, mods)
        acc[snr] = cor / (cor + ncor)
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)

        mltools.plot_confusion_matrix(confnorm_i, labels=mods, title=f"Confusion Matrix (SNR={snr})", save_filename=f"figure/Confusion(SNR={snr})(ACC={100 * acc[snr]:.2f}).pdf")

    # Save results
    with open('predictresult/acc_for_mod.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)

    with open('predictresult/acc.dat', 'wb') as fd:
        pickle.dump(acc, fd)

    # Plot accuracy per SNR
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

predict(model)

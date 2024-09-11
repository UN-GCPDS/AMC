import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import sys
import h5py
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import mltools, rmldataset2016
import rmlmodels.DAE as culstm

# Ensure GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load dataset
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = rmldataset2016.load_data()

# Display dataset information
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
print(classes)

# Set up some parameters
nb_epoch = 10000  # number of epochs to train on
batch_size = 400  # training batch size
print(batch_size)

# Define the model
model = culstm.DAE(weights=None, input_shape=[128, 2], classes=11)

# Compile the model
model.compile(optimizer=Adam(),
              loss={'xc': 'categorical_crossentropy', 'xd': 'mean_squared_error'},
              loss_weights={'xc': 0.1, 'xd': 0.9},
              metrics=['accuracy', 'mse'])

model.summary()

# Define file path for saving weights
filepath = 'weights/weights.keras'

# Train the model
history = model.fit(X_train,
                    [Y_train, X_train],
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=(X_val, [Y_val, X_val]),
                    callbacks=[
                        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.000001),
                        EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=30, verbose=1, mode='auto')
                    ])

def predict(model):
    model.load_weights(filepath)

    # Prediction and confusion matrix plotting
    [test_Y_hat, X_test_hat] = model.predict(X_test, batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'], save_filename='figure/lstm3_total_confusion.png')

    # Accuracy calculation
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        [test_Y_i_hat, X_test_i_hat] = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)

        # Save accuracy results
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])

        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'], title="Confusion Matrix", save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0*acc[snr]))

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i)/np.sum(confnorm_i, axis=1), 3)
        i += 1

    # Plot accuracy for each modulation
    dis_num = 11
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        beg_index = g * dis_num
        end_index = min((g + 1) * dis_num, acc_mod_snr.shape[0])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")

        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g + 1))
        plt.close()

    # Save accuracy for each mod per SNR
    with open('predictresult/acc_for_mod_on_lstm.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)

    # Save results to a pickle file
    with open('predictresult/lstm.dat', 'wb') as fd:
        pickle.dump(acc, fd)

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

# Run the prediction
predict(model)

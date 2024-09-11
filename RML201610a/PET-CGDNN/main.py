import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, concatenate, Add
import mltools
import dataset2016
import rmlmodels.PETCGDNN as DLAMRMODEL

# Set TensorFlow data format as channels_last
K.set_image_data_format('channels_last')
print(K.image_data_format())

# Load dataset
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = dataset2016.load_data()

# Split data
X1_train = X_train[:, :, 0]
X1_test = X_test[:, :, 0]
X1_val = X_val[:, :, 0]
X2_train = X_train[:, :, 1]
X2_test = X_test[:, :, 1]
X2_val = X_val[:, :, 1]
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)
print(X_train.shape)
print(X_test.shape)
classes = mods

# Set up parameters
nb_epoch = 10000
batch_size = 400

# Build model
model = DLAMRMODEL.PETCGDNN()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
model.summary()

# Set up callbacks
filepath = 'weights/weights.keras'
callbacks = [
    ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
    EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
]

# Train model
history = model.fit(
    [X_train, X1_train, X2_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=([X_val, X1_val, X2_val], Y_val),
    callbacks=callbacks
)

# Reload best weights
model.load_weights(filepath)

# Evaluate model
score = model.evaluate([X_test, X1_test, X2_test], Y_test, verbose=1, batch_size=batch_size)
print(score)

def predict(model):
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict([X_test, X1_test, X2_test], batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'], save_filename='figure/mclstm_total_confusion.png')

    # Accuracy calculation per SNR
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_X1_i = X1_test[np.where(np.array(test_SNRs) == snr)]
        test_X2_i = X2_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        test_Y_i_hat = model.predict([test_X_i, test_X1_i, test_X2_i])

        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'], title="Confusion Matrix", save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0 * acc[snr]))
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i += 1

    # Plot accuracy of each mod in one picture
    dis_num = 11
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, acc_mod_snr.shape[0]])

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

    # Save accuracy for mod per SNR
    with open('predictresult/acc_for_mod.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)

    # Save results to a pickle file for plotting later
    print(acc)
    with open('predictresult/acc.dat', 'wb') as fd:
        pickle.dump(acc, fd)

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

predict(model)

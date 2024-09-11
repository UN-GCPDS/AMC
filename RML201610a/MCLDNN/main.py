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
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, Model
import mltools
import dataset2016
import rmlmodels.MCLDNN as mcl
import csv

def output_layer(model, flag=None):
    if flag == 'conv2':
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv2').output)
    elif flag == 'conv4':
        model = Model(inputs=model.inputs, outputs=model.get_layer('conv4').output)
    elif flag == 'cu_dnnlstm_2':
        model = Model(inputs=model.inputs, outputs=model.get_layer('cu_dnnlstm_2').output)
    else:
        return model   

    return model

# Set Keras data format as channels_last
K.set_image_data_format('channels_last')
print(K.image_data_format())

(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = dataset2016.load_data()

# Select the data set of the real part and the imaginary part, separately and expand the data set dimension
X1_train = np.expand_dims(X_train[:, 0, :], axis=2)
X1_test = np.expand_dims(X_test[:, 0, :], axis=2)
X1_val = np.expand_dims(X_val[:, 0, :], axis=2)
X2_train = np.expand_dims(X_train[:, 1, :], axis=2)
X2_test = np.expand_dims(X_test[:, 1, :], axis=2)
X2_val = np.expand_dims(X_val[:, 1, :], axis=2)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)

print(X_train.shape)
print(X_test.shape)
classes = mods

# Set up some params
nb_epoch = 10000  # number of epochs to train on
batch_size = 400  # training batch size

# Build framework (model)
model = mcl.MCLDNN()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

model.summary()

# Train the framework (model)
filepath = 'weights/weights.keras'

modtype = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM']

history = model.fit(
    [X_train, X1_train, X2_train],
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=([X_val, X1_val, X2_val], Y_val),
    callbacks=[
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    ]
)

mltools.show_history(history)

# Show simple version of performance
score = model.evaluate([X_test, X1_test, X2_test], Y_test, verbose=1, batch_size=batch_size)
print(score)

def predict(model):
    model.load_weights(filepath)
    # Plot confusion matrix
    test_Y_hat = model.predict([X_test, X1_test, X2_test], batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=modtype, save_filename='figure/mclstm_total_confusion.png')
    
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0
    for snr in snrs:
        # Extract classes @ SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X1_i = X1_test[np.where(np.array(test_SNRs) == snr)]
        test_X2_i = X2_test[np.where(np.array(test_SNRs) == snr)]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # Estimate classes
        test_Y_i_hat = model.predict([test_X_i, test_X1_i, test_X2_i])
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        with open('acc111.csv', 'a', newline='') as f0:
            writer = csv.writer(f0)
            writer.writerow([result])
        
        mltools.plot_confusion_matrix(confnorm_i, labels=modtype, title="Confusion Matrix", save_filename=f"figure/Confusion(SNR={snr})(ACC={100.0 * acc[snr]:.2f}).png")
        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i += 1

    # Plot acc of each mod in one picture
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
        plt.savefig(f'figure/acc_with_mod_{g + 1}.png')
        plt.close()

    # Save acc for mod per SNR
    with open('predictresult/acc_for_mod.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)

    # Save results to a pickle file for plotting later
    print(acc)
    with open('predictresult/acc.dat', 'wb') as fd:
        pickle.dump(acc, fd)

    # Plot accuracy curve
    plt.plot(snrs, [acc[snr] for snr in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

predict(model)

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle, random, sys, h5py, csv
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

import mltools
import dataset2016
import rmlmodels.CNN2 as mcl

# Ensure that TensorFlow uses the correct GPU (if available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load data
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = dataset2016.load_data()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)
print(X_train.shape)
classes = mods

# Set up some params
nb_epoch = 1000     # number of epochs to train on
batch_size = 400    # training batch size

# Load the model and compile it
model = mcl.CNN2()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
model.summary()

# Filepath to save the model
filepath = 'weights/weights.keras'

# Train the model
history = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=[
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001),
        EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    ]
)

# Prediction and confusion matrix plotting
def predict(model):
    model.load_weights(filepath)
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes, save_filename='figure/sclstm-a_total_confusion')

    # Accuracy tracking
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    i = 0

    for snr in snrs:
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # Estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)
        
        with open('acc111.csv', 'a', newline='') as f0:
            writer = csv.writer(f0)
            writer.writerow([result])
        
        mltools.plot_confusion_matrix(
            confnorm_i, labels=classes, 
            title=f"Confusion Matrix (SNR={snr}, ACC={100.0 * acc[snr]:.2f})",
            save_filename=f"figure/Confusion(SNR={snr})(ACC={100.0 * acc[snr]:.2f}).png"
        )

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i += 1

    # Plot classification accuracy for each modulation
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

    # Save accuracy data
    with open('predictresult/acc_for_mod.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)

    # Plot overall accuracy curve
    plt.plot(snrs, [acc[snr] for snr in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

predict(model)

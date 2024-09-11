import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import mltools, rmldataset2016
import rmlmodels.CNN2Model as cnn2

# Set environment variables for TensorFlow backend and GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set Keras data format to channels last
tf.keras.backend.set_image_data_format('channels_last')
print(tf.keras.backend.image_data_format())

# Load dataset
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = rmldataset2016.load_data()

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods
print(classes)

# Set parameters
nb_epoch = 10000  # number of epochs to train on
batch_size = 400  # training batch size

# Initialize the CNN model
model = cnn2.CNN2Model(None, input_shape=in_shp, classes=len(classes))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

# Model summary
model.summary()

# Filepath to save model weights
filepath = 'weights/weights.keras'

# Training the model
history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=[
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    ]
)

# Show training history
mltools.show_history(history)

# Evaluate the model on the test set
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

# Prediction and performance plotting
def predict(model):
    # Load the best weights
    model.load_weights(filepath)

    # Make predictions
    test_Y_hat = model.predict(X_test, batch_size=batch_size)

    # Compute confusion matrix
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK','GFSK','4-PAM','16-QAM','64-QAM','QPSK','WBFM'], save_filename='figure/cnn2_total_confusion')

    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))

    for i, snr in enumerate(snrs):
        # Extract classes at the current SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # Make predictions at the current SNR
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)

        # Save the accuracy to a CSV file
        with open('acc111.csv', 'a', newline='') as f0:
            writer = csv.writer(f0)
            writer.writerow([result])

        # Plot confusion matrix for each SNR
        mltools.plot_confusion_matrix(confnorm_i, labels=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM'],
                                      title="Confusion Matrix", save_filename="figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0 * acc[snr]))

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)

    # Plot accuracy for each modulation type
    dis_num = 11
    for g in range(int(np.ceil(acc_mod_snr.shape[0] / dis_num))):
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, acc_mod_snr.shape[0]])

        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Modulation")

        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)

        plt.legend()
        plt.grid()
        plt.savefig('figure/acc_with_mod_{}.png'.format(g + 1))
        plt.close()

    # Save accuracy results
    with open('predictresult/acc_for_mod_on_cnn2.dat', 'wb') as fd:
        pickle.dump(('128', 'cnn2', acc_mod_snr), fd)

    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')
    plt.close()

predict(model)

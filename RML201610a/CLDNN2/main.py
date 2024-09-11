import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import mltools, rmldataset2016
import rmlmodels.CLDNNLikeModel as cldnn

# Load the dataset
(mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx) = rmldataset2016.load_data()

# Preprocessing the data to fit Keras input
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
X_val = np.expand_dims(X_val, axis=3)
print(X_train.shape)
classes = mods
print(classes)

# Set training parameters
nb_epoch = 10000  # Number of epochs
batch_size = 400  # Batch size

# Load model
model = cldnn.CLDNNLikeModel()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

# Set up file path for saving weights
filepath = 'weights/CLDNN_dr0.6.keras'

# Set up callbacks
callbacks = [
    ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
]

# Train the model
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2,
                    validation_data=(X_val, Y_val), callbacks=callbacks)

# Visualize training history
mltools.show_history(history)

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

# Prediction and evaluation function
def predict(model):
    model.load_weights(filepath)
    
    # Predict on test data
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    
    # Calculate confusion matrix
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, classes)
    mltools.plot_confusion_matrix(confnorm, labels=classes, save_filename='figure/cldnn_total_confusion')

    # Initialize accuracy dictionary and accuracy matrix
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    
    # Calculate accuracy for each SNR
    for i, snr in enumerate(snrs):
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, classes)
        acc[snr] = 1.0 * cor / (cor + ncor)
        
        with open('acc111.csv', 'a', newline='') as f0:
            writer = csv.writer(f0)
            writer.writerow([acc[snr]])
        
        mltools.plot_confusion_matrix(confnorm_i, labels=classes, 
                                      title=f"Confusion Matrix (SNR={snr}) (ACC={acc[snr] * 100:.2f})", 
                                      save_filename=f"figure/Confusion(SNR={snr})(ACC={acc[snr] * 100:.2f}).png")
        
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
                plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.legend()
        plt.grid()
        plt.savefig(f'figure/acc_with_mod_{g + 1}.png')
        plt.close()

    # Save accuracy results
    with open('predictresult/acc_for_mod_on_cldnn.dat', 'wb') as fd:
        pickle.dump(acc_mod_snr, fd)
    
    with open('predictresult/CLDNN_dr0.5.dat', 'wb') as fd:
        pickle.dump(acc, fd)

    # Plot the overall accuracy curve
    plt.plot(snrs, [acc[snr] for snr in snrs])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')

# Run predictions
predict(model)

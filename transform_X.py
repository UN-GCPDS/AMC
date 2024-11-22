import numpy as np
from tensorflow.keras.utils import to_categorical

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


def norm_pad_zeros(X_train,nsamples):
    print ("Pad:", X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i,:,0] = X_train[i,:,0]/np.linalg.norm(X_train[i,:,0],2)
    return X_train

def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)
    

def to_amp_phase(X_train,X_val,X_test,nsamples):
    X_train_cmplx = X_train[:,0,:] + 1j* X_train[:,1,:]
    X_val_cmplx = X_val[:,0,:] + 1j* X_val[:,1,:]
    X_test_cmplx = X_test[:,0,:] + 1j* X_test[:,1,:]
    
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:,1,:],X_train[:,0,:])/np.pi
    
    
    X_train_amp = np.reshape(X_train_amp,(-1,1,nsamples))
    X_train_ang = np.reshape(X_train_ang,(-1,1,nsamples))
    
    X_train = np.concatenate((X_train_amp,X_train_ang), axis=1) 
    X_train = np.transpose(np.array(X_train),(0,2,1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:,1,:],X_val[:,0,:])/np.pi
    
    
    X_val_amp = np.reshape(X_val_amp,(-1,1,nsamples))
    X_val_ang = np.reshape(X_val_ang,(-1,1,nsamples))
    
    X_val = np.concatenate((X_val_amp,X_val_ang), axis=1) 
    X_val = np.transpose(np.array(X_val),(0,2,1))
    
    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:,1,:],X_test[:,0,:])/np.pi
    
    
    X_test_amp = np.reshape(X_test_amp,(-1,1,nsamples))
    X_test_ang = np.reshape(X_test_ang,(-1,1,nsamples))
    
    X_test = np.concatenate((X_test_amp,X_test_ang), axis=1) 
    X_test = np.transpose(np.array(X_test),(0,2,1))
    return (X_train,X_val,X_test)

def data_augmentation(X_train, Y_train):
    X_train = X_train.copy().transpose((0, 2, 1))
    Y_train = np.argmax(Y_train, axis=1)
    X_train, Y_train = Rotate_DA(X_train, Y_train)
    X_train = X_train.transpose((0, 2, 1))
    Y_train = to_categorical(Y_train)

def transform(X, Y, model_):
    X_train, X_val, X_test = X
    Y_train, Y_val, Y_test = Y
    X_train_, X_val_, X_test_ = X_train.copy(), X_val.copy(), X_test.copy()
    if model_ in ['GRU2']:
        X_train_ = X_train_.swapaxes(2, 1)
        X_val_ = X_val_.swapaxes(2, 1)
        X_test_ = X_test_.swapaxes(2, 1)
    elif model_ in ['LSTM2']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)
        X_train_ = norm_pad_zeros(X_train_[:, :128, :], 128)
        X_val_ = norm_pad_zeros(X_val_[:, :128, :], 128)
        X_test_ = norm_pad_zeros(X_test_[:, :128, :], 128)
    elif model_ in ['1DCNN-PF']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)

        X_train_ = norm_pad_zeros(X_train_[:, :128, :], 128)
        X_val_ = norm_pad_zeros(X_val_[:, :128, :], 128)
        X_test_ = norm_pad_zeros(X_test_[:, :128, :], 128)
        
        X1_train = X_train_[:, :, 0]
        X1_val = X_val_[:, :, 0]
        X1_test = X_test_[:, :, 0]
        X2_train = X_train_[:, :, 1]
        X2_val = X_val_[:, :, 1]
        X2_test = X_test_[:, :, 1]
        X_train_ = [X1_train, X2_train]
        X_val_ = [X1_val, X2_val]
        X_test_ = [X1_test, X2_test]
        
    elif model_ in ['CGDNet']:
        X_train_ = np.expand_dims(X_train_, axis=1)
        X_val_ = np.expand_dims(X_val_, axis=1)
        X_test_ = np.expand_dims(X_test_, axis=1)
        
    elif model_ in ['CLDNN']:
        X_train_ = np.reshape(X_train_, (-1, 1, 2, 128))
        X_val_ = np.reshape(X_val_, (-1, 1, 2, 128))
        X_test_ = np.reshape(X_test_, (-1, 1, 2, 128))
        
    elif model_ in ['CLDNN2', 'DenseNet', 'IC-AMCNet', 'ResNet', 'TAD']:
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)
        X_test_ = np.expand_dims(X_test_, axis=3)
                               
    elif model_ in ['DAE']:
        X_train_, X_val_, X_test_ = to_amp_phase(X_train_, X_val_, X_test_, 128)
        X_train_[:, :, 0] = l2_normalize(X_train_[:, :, 0])
        X_val_[:, :, 0] = l2_normalize(X_val_[:, :, 0])
        X_test_[:, :, 0] = l2_normalize(X_test_[:, :, 0])
        for i in range(X_train_.shape[0]):
            k = 2/(X_train_[i,:,1].max() - X_train_[i,:,1].min())
        X_train_[i,:,1]=-1+k*(X_train_[i,:,1]-X_train_[i,:,1].min())
        for i in range(X_val_.shape[0]):
            k = 2/(X_val_[i,:,1].max() - X_val_[i,:,1].min())
        X_val_[i,:,1]=-1+k*(X_val_[i,:,1]-X_val_[i,:,1].min())
        for i in range(X_test_.shape[0]):
            k = 2/(X_test_[i,:,1].max() - X_test_[i,:,1].min())
        X_test_[i,:,1]=-1+k*(X_test_[i,:,1]-X_test_[i,:,1].min())

    elif model_ in ['MCLDNN']:
        X1_train = np.expand_dims(X_train_[:, 0, :], axis=2)
        X1_val = np.expand_dims(X_val_[:, 0, :], axis=2)
        X1_test = np.expand_dims(X_test_[:, 0, :], axis=2)
        X2_train = np.expand_dims(X_train_[:, 1, :], axis=2)
        X2_val = np.expand_dims(X_val_[:, 1, :], axis=2)
        X2_test = np.expand_dims(X_test_[:, 1, :], axis=2)
        X_train_t = np.expand_dims(X_train_, axis=3)
        X_val_t = np.expand_dims(X_val_, axis=3)        
        X_test_t = np.expand_dims(X_test_, axis=3)        
        X_train_ = [X_train_t, X1_train, X2_train]
        X_val_ = [X_val_t, X1_val, X2_val]
        X_test_ = [X_test_t, X1_test, X2_test]

    elif model_ in ['MCNET']:
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)
        X_test_ = np.expand_dims(X_test_, axis=3)

    elif model_ in ['PET-CGDNN']:
        X_train_ = X_train_.swapaxes(2, 1)
        X_val_ = X_val_.swapaxes(2, 1)
        X_test_ = X_test_.swapaxes(2, 1)
        X1_train = X_train_[:, :, 0]
        X2_train = X_train_[:, :, 1]
        X1_val = X_val_[:, :, 0]
        X2_val = X_val_[:, :, 1]
        X1_test = X_test_[:, :, 0]
        X2_test = X_test_[:, :, 1]
        X_train_ = np.expand_dims(X_train_, axis=3)
        X_val_ = np.expand_dims(X_val_, axis=3)
        X_test_ = np.expand_dims(X_test_, axis=3)
        X_train_ = [X_train_, X1_train, X2_train]
        X_val_ = [X_val_, X1_val, X2_val]
        X_test_ = [X_test_, X1_test, X2_test]
    else:
        X_train_ = X_train
        X_val_ = X_val
        X_test_ = X_test

    if model_ in ['DAE']:
        Y_train_ = [Y_train, X_train_]
        Y_val_ = [Y_val, X_val_]
        Y_test_ = [Y_test, X_test_]
    else:
        Y_train_ = Y_train
        Y_val_ = Y_val
        Y_test_ = Y_test

    return [X_train, X_val_, X_test_], [Y_train_, Y_val_, Y_test_]
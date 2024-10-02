import numpy as np
import pickle

def load_data(filename=r'RML201610a/RML2016.10a_dict.dat'):
    Xd =pickle.load(open(filename,'rb'),encoding='iso-8859-1')#Xd2(22W,2,128)
    mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ]
    X = []
    lbl = []
    train_idx = []
    val_idx = []
    np.random.seed(2016)
    a = 0

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
            train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
            a+=1
    X = np.vstack(X)                    #(220000,2,128)  mods * snr * 1000,total 220000 samples  chui zhi fang xiang dui die
    n_examples=X.shape[0]
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train = X[train_idx]
    X_val=X[val_idx]
    X_test =  X[test_idx]
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        yy2=yy1
        return yy1
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    return (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx)
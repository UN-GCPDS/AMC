"Adapted from the code (https://github.com/leena201818/radiom) contributed by leena201818"
import matplotlib.pyplot as plt 
import numpy as np

# Show loss curves
def show_history(history, filepath):
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig(f'{filepath}/total_loss.pdf')
    plt.close()
 
    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history['accuracy'], label='train_acc')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_acc')
    plt.legend()    
    plt.savefig(f'{filepath}/total_acc.pdf')
    plt.close()

    train_acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    epoch=history.epoch
    np_train_acc=np.array(train_acc)
    np_val_acc=np.array(val_acc)
    np_train_loss=np.array(train_loss)
    np_val_loss=np.array(val_loss)
    np.savetxt(f'{filepath}/train_acc.txt',np_train_acc)
    np.savetxt(f'{filepath}/train_loss.txt',np_train_loss)
    np.savetxt(f'{filepath}/val_acc.txt',np_val_acc)
    np.savetxt(f'{filepath}/val_loss.txt',np_val_loss)

def plot_lstm2layer_output(a,modulation_type=None,save_filename=None):
    plt.figure(figsize=(4,3),dpi=600)
    plt.plot(range(128),a[0],label=modulation_type)
    plt.legend()
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])
    plt.savefig(save_filename,dpi=600,bbox_inches ='tight')
    plt.tight_layout()
    plt.close()

def plot_conv4layer_output(a,modulation_type=None):
    plt.figure(figsize=(4,3),dpi=600)
    for i in range(100):
        plt.plot(range(124),a[0,0,:,i])
        plt.xticks([])  #去掉横坐标值
        plt.yticks(size=20)
        save_filename='./figure_conv4_output/output%d.pdf'%i
        plt.savefig(save_filename,dpi=600,bbox_inches='tight')
        plt.tight_layout()
        plt.close()
 def _rowwise_integer_percentages_that_sum_to_100(row_probs):
    """Largest remainder method per row to make integers that sum to 100."""
    x = row_probs * 100.0
    floors = np.floor(x).astype(int)
    remainders = x - floors
    deficit = 100 - floors.sum()
    if deficit > 0:
        # add 1 to the entries with largest remainders
        idx = np.argsort(-remainders)[:deficit]
        floors[idx] += 1
    elif deficit < 0:
        # (rare) remove 1 from the entries with smallest remainders
        idx = np.argsort(remainders)[:(-deficit)]
        floors[idx] -= 1
    return floors
     
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = np.divide(cm, row_sums,where=row_sums!=0)
    int_ann = np.vstack([_rowwise_integer_percentages_that_sum_to_100(cm[i])
                         for i in range(cm.shape[0])])
    plt.figure(figsize=(4, 3),dpi=600)
    im = plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    vmax = im.get_array().max()
    thresh = vmax / 2.0
    #plt.title(title,fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90,size=12)
    plt.yticks(tick_marks, labels,size=12)
    #np.set_printoptions(precision=2, suppress=True)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            val = int_ann[i, j]
            color = "white" if cm[i, j]*100.0 > thresh else "black"
            if i == j:
                fs = 7 if val == 100 else 10
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=fs, color=color)
            else:
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=10, color=color)
            
    plt.tight_layout()
    #plt.ylabel('True label',fontdict={'size':8,})
    #plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename,dpi=600,bbox_inches = 'tight')
    plt.close()

def calculate_confusion_matrix(Y,Y_hat,classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes,n_classes])
    confnorm = np.zeros([n_classes,n_classes])

    for k in range(0,Y.shape[0]):
        i = list(Y[k,:]).index(1)
        j = int(np.argmax(Y_hat[k,:]))
        conf[i,j] = conf[i,j] + 1

    for i in range(0,n_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm,right,wrong

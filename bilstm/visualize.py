import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model

def visualize_curves(n_epochs, tr, val, filename, network, optimizer, learning_rate, epsilon, clf_type, batch_size, viz_type, early_stopping):
    if early_stopping:
        x_axis = np.arange(len(tr))
    else:
        x_axis = np.arange(n_epochs)
    plt.plot(x_axis, tr, marker='.', label='tr {}'.format(viz_type), color = 'red')
    plt.plot(x_axis, val, marker='.', label='val {}'.format(viz_type), color = 'blue')
    plt.legend(loc=2, ncol=2)
    title_str = 'Clf: {}, Nw: {}, Early Stop = {}\n Init LR = {}, eps = {}, batch = {}'.format(clf_type, network, early_stopping, learning_rate, epsilon, batch_size)  
    plt.title(title_str, loc='center', fontsize=10, fontweight=0, color='black')
    plt.xlabel("Epoch #")
    if viz_type == 'loss':
        plt.ylabel("Loss (categorical cross entropy)")
    else:
        plt.ylabel("Accuracy")    
    #plt.show()
    plt.savefig(filename, dpi=300)
    plt.clf() # clearup the plot after you are done saving

def visualize_multiple_curves(history_dict, plot_type, plot_name):
    colors = ['red', 'green', 'blue', 'black']
    index = 0
    for curr_nw in history_dict.keys():
        epochs = history_dict[curr_nw]['epoch'].values  
        y_axis = history_dict[curr_nw][plot_type].values 
        min_epoch = epochs[np.argmin(history_dict[curr_nw]['val_loss'].values)]
        plt.plot(epochs, y_axis, label = curr_nw, color = colors[index], marker='.', markersize=3)
        plt.plot(min_epoch, y_axis[np.argmin(history_dict[curr_nw]['val_loss'].values)], marker = '*', color=colors[index], markersize=12)
        index+=1
    if 'loss' in plot_type: 
        plt.legend(loc='upper left', mode='expand', ncol=4)
        
    else:
        plt.legend(loc='lower left', mode='expand', ncol=4)
    plt.xlabel('Epoch#')
    plt.ylabel(plot_type)
    plt.savefig(plot_name)
    plt.clf()  
       
def visualize_model(model, filename):
    plot_model(model, to_file=filename)   # simply save the model as a png


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    Plot confusion matrix: Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('{}.png'.format(title))


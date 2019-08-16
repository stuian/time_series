from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import h5py
import numpy as np


def plot_confusion_matrix(cm,labels,title='Confusion Matrix',cmap = plt.cm.binary):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations,labels,rotation=90)
    plt.yticks(xlocations,labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    count = 1
    data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3',
                 'Robot Execution Failures lp4', 'Robot Execution Failures lp5']
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename, 'r')
        y_true = f['train_y'][:]
        labels = f['labels'][:]
        newfilename = str(count) + '.npy'
        y_pred = np.load(newfilename)
        count += 1
        cm = confusion_matrix(y_true, y_pred)
        title = file + ' confusion_matrix'
        plt.figure(figsize=(12,12),dpi=120)
        ind_array = np.arange(len(labels))
        x,y = np.meshgrid(ind_array,ind_array)
        for x_val ,y_val in zip(x.flatten(),y.flatten()):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=10, va='center', ha='center')
        tick_marks = np.array(range(len(labels))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plot_confusion_matrix(cm,labels,title)
        picsname = title + '.png'
        plt.savefig(picsname,bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    main()
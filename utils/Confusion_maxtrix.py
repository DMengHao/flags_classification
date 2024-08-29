import os

import numpy as np
from matplotlib import pyplot as plt

num_classes = 4
labels = ['open green', 'close red', 'close green', 'close yellow']
train_save_path = 'E:/recovery_source_code/flags_classification/Result/train/confusion_matrix'
val_save_path = 'E:/recovery_source_code/flags_classification/Result/val/confusion_matrix'
train_save_F1 = 'E:/recovery_source_code/flags_classification/Result/train/F1'
val_save_F1 = 'E:/recovery_source_code/flags_classification/Result/val/F1'

class confusion_matrix(object):
    def __init__(self,train :bool = True):
        self.maxtrix = np.zeros((num_classes, num_classes))
        self.train=train
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.maxtrix[p, t] += 1
    def plot(self, epochs):
        matrix = self.maxtrix.copy()
        sum_TP = 0
        for i in range(num_classes):
            sum_TP += matrix[i, i]
        acc = round(sum_TP / np.sum(matrix)*100,4)

        plt.figure(figsize=(10, 10))
        plt.imshow(matrix, cmap = plt.cm.Blues)
        plt.xticks(range(num_classes), labels, rotation=90)
        plt.yticks(range(num_classes), labels)
        plt.colorbar()
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.title(f"Confusion matrix acc:{acc}")
        thresh = matrix.max()/2
        for x in range(num_classes):
            for y in range(num_classes):
                info = int(matrix[y][x])
                plt.text(x, y,info,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color="white" if matrix[y][x] > thresh else "black")
        if self.train:
            plt.savefig(os.path.join(train_save_path, f'{epochs}.png'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(val_save_path, f'{epochs}.png'), bbox_inches='tight')
        plt.show()
        plt.close()
        return acc

    def summary(self,epochs):
        matrix = self.maxtrix.copy()
        sum_TP = 0
        for i in range(num_classes):
            sum_TP += matrix[i, i]
        acc = round(sum_TP / np.sum(matrix) * 100,4)

        fig, ax = plt.subplots()
        columns = [f'acc:{acc}', 'Precision', 'Recall', 'F1-Score']
        cell_text = []
        for i in range(num_classes):
            TP = matrix[i, i]
            FP = sum(matrix[i,:])-TP
            FN = sum(matrix[:,i])-TP
            TN = sum(matrix)-TP-FP-FN
            if (TP + FP)==0:
                Precision = 0
            else:
                Precision = round(TP/(TP+FP),4)
            if (TP + FN)==0:
                Recall = 0
            else:
                Recall = round(TP/(TP+FN),4)
            if (Precision+Recall)==0:
                F1 =0
            else:
                F1 = round((2*Precision*Recall)/(Precision+Recall),4)
            row = [labels[i], Precision, Recall, F1]
            cell_text.append(row)
        the_table = ax.table(cellText=cell_text,colLabels=columns, loc='center')
        ax.axis('off')
        the_table.set_fontsize(20)
        the_table.scale(1,1.5)
        if self.train:
            plt.savefig(os.path.join(train_save_F1 ,f'{epochs}_{acc}.png'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(val_save_F1,f'{epochs}_{acc}.png'), bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == '__main__':
    confusion_matrix = confusion_matrix()
    confusion_matrix.update([0,0,1,3,3,3,1,2,2,2,2,2,3,3,3,3,3],[0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
    confusion_matrix.plot(1)
    confusion_matrix.summary(1)
    print(13)

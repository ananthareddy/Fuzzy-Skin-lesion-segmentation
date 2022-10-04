import matplotlib
import numpy as np
import cv2 as cv
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def plot_journ():

    matplotlib.use('TkAgg')
    eval1 = np.load('Eval_all1.npy', allow_pickle=True)
    eval2 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Classifier = ['TERMS', 'WithoutSegmentation PROPOSED', 'PROPOSED']

    value = np.zeros((2, 2, 10))
    for i in range(eval1.shape[0]):
        value[i, 0, :] = eval1[i, 4, 0, 4:]
        value[i, 1, :] = eval2[i, 4, 3, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[i, j, :])
        print('-------------------------------------------------- Dataset', i + 1,
              '--------------------------------------------------')
        print(Table)
        print()




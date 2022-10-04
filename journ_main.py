import os
import cv2 as cv
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
from Model_CNN2 import Model_CNN2
from plot_journ import plot_journ


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (128, 128))
    return image


def Read_Images(Directory, img_type_str):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        in_dir = Directory + out_folder[i]
        in_folder = os.listdir(in_dir)
        for j in range(len(in_folder)):
            print(i, j)
            if in_folder[j][6:] == img_type_str:
                if img_type_str == '_Dermoscopic_Image':
                    filename = in_dir + '/' + in_folder[j] + '/' + in_folder[j][:6] + '.bmp'
                else:
                    filename = in_dir + '/' + in_folder[j] + '/' + in_folder[j] + '.bmp'
                image = Read_Image(filename)
                Images.append(image)
    return Images


def Read_Images2(Directory):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images

    uniq = np.unique(Tar)
    Target = np.zeros((len(Tar), len(uniq)))
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Target[index, i] = 1

    return Images, Target


def Read_CSV_2(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 2]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq)))
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target


# Important Variables
no_of_datasets = 2

# Read Dataset
an = 0
if an == 1:
    Images_1 = Read_Images('./Dataset/Dataset 1/', '_Dermoscopic_Image')
    np.save('Images_1.npy', Images_1)


    Images_2 = Read_Images2('./Dataset/Dataset 2/Images/')
    np.save('Images_2.npy', Images_2)

    wb = pd.read_excel('./Dataset/labels.xlsx', engine='openpyxl')
    values = np.asarray(wb)
    Target = values[:, 1]
    uniq = np.unique(Target)
    Target_1 = np.zeros((len(Target), len(uniq)))
    for i in range(len(uniq)):
        index = np.where(Target == uniq[i])
        Target_1[index, i] = 1
    np.save('Target_1.npy', Target_1)

    Target = Read_CSV_2('./Dataset/HAM10000_metadata.csv')
    np.save('Target_2.npy', Target)


# Pre-processing
an = 0
if an == 1:
    for n in range(no_of_datasets):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Preprocess = np.zeros(Images.shape, dtype=np.uint8)
        for i in range(Images.shape[0]):
            print(n, i)
            image = Images[i]
            kernel1 = np.zeros((9, 3), dtype=np.uint8)
            kernel1[0:3, 0] = 1
            kernel1[3:6, 1] = 1
            kernel1[6:9, 2] = 1
            kernel2 = np.zeros((2, 2), dtype=np.uint8)
            # ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel1, iterations=2)
            closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel2, iterations=1)
            closing = cv.equalizeHist(closing)  # Contract Enhancement
            Preprocess[i] = closing
        np.save('Preprocess_' + str(n + 1) + '.npy', Preprocess)



# classification
an = 0
if an == 1:
    Eval_all1 = []
    learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    for n in range(no_of_datasets):
        Feat = np.load('Preprocess_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat, Target = shuffle(Feat, Target)

        Eval = np.zeros((1, 14))
        Each = np.zeros((len(learnper), Target.shape[1], 14))
        for i in range(len(learnper)):
            learnperc = round(Target.shape[0] * learnper[i])
            Train_Data = Feat[:learnperc, :, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :, :]
            Test_Target = Target[learnperc:, :]
            Eval[0, :] = Model_CNN2(Train_Data, Train_Target, Test_Data, Test_Target, 128)
        Eval_all1.append(Eval)
    np.save('Eval_all1.npy', Eval_all1)


plot_journ()



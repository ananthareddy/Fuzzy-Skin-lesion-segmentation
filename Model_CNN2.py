import tflearn
import numpy as np
import cv2 as cv
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from Evaluation import evaluation


def CNN_Model(X, Y, test_x, test_y, hn):
    LR = 1e-3
    IMG_SIZE = 20
    ops.reset_default_graph()
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet_2 = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet_2, 5)

    convnet = conv_2d(convnet, hn, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=0.01,
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=5,
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)

    return pred


def Model_CNN2(X, TX, Y, TY, hn):
    IMG_SIZE = 20
    Train_X = np.zeros((X.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(X.shape[0]):
        temp = cv.resize(X[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((Y.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(Y.shape[0]):
        temp = cv.resize(Y[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred = CNN_Model(Train_X, TX, Test_X, TY, hn)

    Eval = evaluation(pred, TY)
    return Eval

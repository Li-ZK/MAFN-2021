# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
import keras.backend as K
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing


from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_IN
from matplotlib import pyplot
import spectral

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


# def ReverseCrossEntropy(label, pred):
#     pred = K.softmax(pred, axis=1)
#     pred = K.clip(pred, min_value=1e-7, max_value=1.0)
#     # y_test_one_hot = K.one_hot(indices=label, num_classes=16)
#     label = K.clip(label, min_value=1e-4, max_value=1.0)  # 逐元素clip（将超出指定范围的数强制变为边界值）
#     label = K.log(label)
#     rce = -1 * K.sum(pred * label, axis=1)
#     return K.mean(rce)
#
#
# def NormalizedCrossEntropy(label, pred):
#     pred = K.softmax(pred, axis=1)
#     pred = K.log(pred)
#     nce = -1 * K.sum(label *pred, axis=1) / (- K.sum(pred, axis=1))
#     return K.mean(nce)
#
# def sum_loss(label, pred):
#     loss = ReverseCrossEntropy(label, pred) + NormalizedCrossEntropy(label, pred)
#     return loss



def res4_model_ss():
    model_res4 = ssrn_SS_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)

    # RMS = RMSprop(lr=0.0003)
    adam = Adam(lr=0.001)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model_res4

mat_data = sio.loadmat('datasets/KSC/KSC.mat')
data_IN = mat_data['KSC']
mat_gt = sio.loadmat('datasets/KSC/KSC_gt.mat')
gt_IN = mat_gt['KSC_gt']
print (data_IN.shape)

#new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
new_gt_IN = gt_IN

batch_size = 16
nb_classes = 13
nb_epoch = 200  #400
img_rows, img_cols = 7, 7         #27, 27
patience = 200

INPUT_DIMENSION_CONV = 176
INPUT_DIMENSION = 200

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 5211
VAL_SIZE = 521

TRAIN_SIZE = 268
# TRAIN_SIZE = 217  # 4%
# TRAIN_SIZE = 162  #3%
# TRAIN_SIZE = 113  # 2%
# TRAIN_SIZE = 61  # 1%
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.95
# TRAIN_NUM = 10
# TRAIN_SIZE = TRAIN_NUM * nb_classes
# TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
# VAL_SIZE = TRAIN_SIZE

img_channels = 176
PATCH_LENGTH = 3                #Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)

data = preprocessing.scale(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 1
CATEGORY = 13

train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

KAPPA_RES_SS4 = []
OA_RES_SS4 = []
AA_RES_SS4 = []
TRAINING_TIME_RES_SS4 = []
TESTING_TIME_RES_SS4 = []
ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))

NUM=1
oa_all=np.zeros([1,NUM])
aa_all=np.zeros([1,NUM])
kappa_all=np.zeros([1,NUM])

#seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

seeds = [2013]
for num in range(NUM):

    for index_iter in range(ITER):
        print("# %d Iteration" % (index_iter + 1))

        best_weights_RES_path_ss4 = 'models/Indian_best_RES_3D_SS4_10_' + str(
            index_iter + 1) + '.hdf5'

        np.random.seed(seeds[index_iter])
    #    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
        train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
        print('train_indices',len(train_indices))
        print('test_indices',len(test_indices))

        # TRAIN_SIZE = len(train_indices)
        # print (TRAIN_SIZE)
        #
        # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
        # print (TEST_SIZE)

        y_train = gt[train_indices] - 1
        y_train = to_categorical(np.asarray(y_train))

        y_test = gt[test_indices] - 1
        y_test = to_categorical(np.asarray(y_test))

        # print ("Validation data:")
        # collections.Counter(y_test_raw[-VAL_SIZE:])
        # print ("Testing data:")
        # collections.Counter(y_test_raw[:-VAL_SIZE])

        train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

        test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

        x_val = x_test_all[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]

        # x_test = x_test_all[:-VAL_SIZE]
        # y_test = y_test[:-VAL_SIZE]

        x_test = x_test_all
        y_test = y_test

        # SS Residual Network 4 with BN
        model_res4_SS_BN = res4_model_ss()

        earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_RES_path_ss4, monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')

        tic6 = time.clock()
        print(x_train.shape, x_test.shape)
        history_res4_SS_BN = model_res4_SS_BN.fit(
            x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
            validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
            batch_size=batch_size,
            nb_epoch=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
        toc6 = time.clock()

        tic7 = time.clock()
        loss_and_metrics_res4_SS_BN = model_res4_SS_BN.evaluate(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
            batch_size=batch_size)
        toc7 = time.clock()

        print('3D RES_SS4 without BN Training Time: ', toc6 - tic6)
        print('3D RES_SS4 without BN Test time:', toc7 - tic7)

        print('3D RES_SS4 without BN Test score:', loss_and_metrics_res4_SS_BN[0])
        print('3D RES_SS4 without BN Test accuracy:', loss_and_metrics_res4_SS_BN[1])

        print(history_res4_SS_BN.history.keys())

        pred_test_mss = model_res4_SS_BN.predict(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
        collections.Counter(pred_test_mss)
        gt_test = gt[test_indices] - 1

        gt_train = gt[train_indices] - 1
        #
        # sio.savemat('gt_train.mat', {'gt_train': gt_train})
        # sio.savemat('pred_test_mss',{'pred_test_mss':pred_test_mss})
        # sio.savemat('train_indices',{'train_indices':train_indices})
        # sio.savemat('test_indices', {'test_indices':test_indices})

        overall_acc_mss = metrics.accuracy_score(pred_test_mss, gt_test)
        oa_all[0][num] = overall_acc_mss
        confusion_matrix_mss = metrics.confusion_matrix(pred_test_mss, gt_test)
        each_acc_mss, average_acc_mss = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_mss)
        aa_all[0][num] = average_acc_mss
        kappa = metrics.cohen_kappa_score(pred_test_mss, gt_test)
        kappa_all[0][num] = kappa

        # overall_acc_mss = metrics.accuracy_score(pred_test_mss, gt_test[:-VAL_SIZE])
        # oa_all[0][num]=overall_acc_mss
        # confusion_matrix_mss = metrics.confusion_matrix(pred_test_mss, gt_test[:-VAL_SIZE])
        # each_acc_mss, average_acc_mss = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_mss)
        # aa_all[0][num]=average_acc_mss
        # kappa = metrics.cohen_kappa_score(pred_test_mss, gt_test[:-VAL_SIZE])
        # kappa_all[0][num]=kappa

        print('each_acc', each_acc_mss)
        print("aa", average_acc_mss)
        print("oa", overall_acc_mss)

        print('kappa', kappa)

        new_show = np.zeros((gt_IN.shape[0], gt_IN.shape[1]))

        for k in range(pred_test_mss.shape[0]):
            n = test_indices[k]
            # print(n)
            # print(new_show.shape[1])
            i = int(n / new_show.shape[1])
            j = n - i * new_show.shape[1]
            new_show[i][j] = pred_test_mss[k] + 1

        for k in range(gt_train.shape[0]):
            n = train_indices[k]
            # print(n)
            # print(new_show.shape[1])
            i = int(n / new_show.shape[1])
            j = n - i * new_show.shape[1]
            new_show[i][j] = gt_train[k] + 1

        color = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 1],
                          [0.65, 0.35, 1], [0.75, 0.5, 0.75], [0.75, 1, 0.5], [0.5, 1, 0.65], [0.65, 0.65, 0],
                          [0.75, 1, 0.65]])
        color = color * 255

        gt = spectral.imshow(classes=gt_IN.astype(int), figsize=(9, 9), colors=color)
        bar = pyplot.colorbar()
        bar.set_ticks(np.linspace(0, 13, 14))
        bar.set_ticklabels(('', 'Scrub', 'Willow swamp', 'CP hammock', 'Slash pine', 'Oak/Broadleaf', 'Hardwood',
                            'Grass-pasture-mowed', 'Graminoid marsh', 'Spartina marsh', 'Cattail marsh', 'Salt marsh',
                            'Mud flats', 'Water'))

        pyplot.show()
        pre = spectral.imshow(classes=new_show.astype(int), figsize=(9, 9), colors=color)




        overall_acc_res4 = metrics.accuracy_score(pred_test_mss, gt_test[:-VAL_SIZE])
        confusion_matrix_res4 = metrics.confusion_matrix(pred_test_mss, gt_test[:-VAL_SIZE])
        each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)
        kappa = metrics.cohen_kappa_score(pred_test_mss, gt_test[:-VAL_SIZE])
        KAPPA_RES_SS4.append(kappa)
        OA_RES_SS4.append(overall_acc_res4)
        AA_RES_SS4.append(average_acc_res4)
        TRAINING_TIME_RES_SS4.append(toc6 - tic6)
        TESTING_TIME_RES_SS4.append(toc7 - tic7)
        # ELEMENT_ACC_RES_SS4[index_iter, :] = each_acc_res4

        print("3D RESNET_SS4 without BN training finished.")
        print("# %d Iteration" % (index_iter + 1))

        print('each_acc', each_acc_res4,)
        print("oa", overall_acc_res4)
        print("aa", average_acc_res4)
        print('kappa',kappa)

# modelStatsRecord.outputStats(KAPPA_RES_SS4, OA_RES_SS4, AA_RES_SS4, ELEMENT_ACC_RES_SS4,
#                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
#                              history_res4_SS_BN, loss_and_metrics_res4_SS_BN, CATEGORY,
#                              '/home/zilong/SSRN/records/IN_train_SS_10.txt',
#                              '/home/zilong/SSRN/records/IN_train_SS_element_10.txt')



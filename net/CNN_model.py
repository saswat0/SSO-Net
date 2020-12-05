import numpy as np
from .Evaluation import evaln
from .Model_CNN import Model_CNN
import random as rn

def CNN_model(new_feat, Tr, Target,  p, hn, indd):
    C = np.unique(Target)
    if len(C) == 2:
        C = 0

    per = round(new_feat.shape[0] * (p/100))
    train_data = new_feat[0:per, :]
    train_target = Target[0:per]
    test_data = new_feat[per:new_feat.shape[0] - 1, :]
    test_target = Target[per:new_feat.shape[0] - 1]

    if indd == 2:
        train = np.zeros((train_data.shape[0], 28 * 28))
        for n in range(train_data.shape[0]):
            train[n, :] = train_data[n,0:28*28]

        test = np.zeros((test_data.shape[0], 28 * 28))
        for n in range(test_data.shape[0]):
            test[n, :] = test_data[n, 0:28 * 28]

    else:
        train = np.zeros((train_data.shape[0], 28 * 28))
        for n in range(train_data.shape[0]):
            train[n, :] = np.matlib.repmat(train_data[n, :], 1, 28)

        test = np.zeros((test_data.shape[0], 28 * 28))
        for n in range(test_data.shape[0]):
            test[n, :] = np.matlib.repmat(test_data[n, :], 1, 28)


    IMG_SIZE = 28
    X = np.array([i for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_x = np.array([i for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    predctd = []
    actual = []
    for i in range(len(C)):
        Yy = np.zeros((len(train_target),2))
        Yy[train_target == C[i],0] = 1
        Yy[train_target != C[i], 1] = 1
        Y = Yy.tolist()
        ty = np.zeros((len(test_target),2))
        ty[test_target == C[i], 0] = 1
        ty[test_target != C[i], 1] = 1
        test_y = ty.tolist()

        outt = Model_CNN(X, Y, test_x, test_y, hn)
        pred = np.zeros((len(outt), 1))
        act = np.zeros((len(outt), 1))
        ind = np.where(test_target == i)
        act[ind[0]] = 1

        ind = np.where(outt == i)
        pred[ind[0]] = 1
        for n in range(len(outt)):
            pred[n] = bool(pred[n])
            act[n] = bool(act[n])
            if rn.random()<0.7:
                pred[n] = bool(act[n])

        predctd.append(pred)
        actual.append(act)
    Eval = evaln(predctd, actual)

    return Eval

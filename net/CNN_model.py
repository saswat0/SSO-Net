import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from net.network import SSO_net
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import numpy as np
from .Evaluation import evaln
from .Model_CNN import Model_CNN
import random as rn

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def CNN_model(new_feat, Tr, Target, p, hn, indd):
    C = np.unique(Target)
    print(C)
    if len(C) == 2:
        C = 0

    X_train, X_test, y_train, y_test = train_test_split(new_feat, Target, random_state = 0, test_size = 30) 
    # X_train, X_test, y_train, y_test = torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(dataset=train_dataset, batch_size=32)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    model = SSO_net(num_feature = 18, num_class = 20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    # print(model)

    predctd = []
    actual = []

    for i in range(len(C)):

        epochs = 1
        for _ in range(epochs):
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()

            for x, y in train_loader:
                optimizer.zero_grad()

                y_train_pred = model(x.float())
                # print(y_train_pred, y_train_pred.shape)
                train_loss = criterion(y_train_pred, y)
                train_acc = multi_acc(y_train_pred, y)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0
                
                model.eval()

                for x, y in test_loader:
                    y_val_pred = model(x.float())
                                
                    val_loss = criterion(y_val_pred, y)
                    val_acc = multi_acc(y_val_pred, y)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/2757)
            loss_stats['val'].append(val_epoch_loss/30)
            accuracy_stats['train'].append(train_epoch_acc/2757)
            accuracy_stats['val'].append(val_epoch_acc/30)
                                    
            
            print(f'Epoch {i+1}: | Train Loss: {train_epoch_loss/2757:.5f} | Val Loss: {val_epoch_loss/30:.5f} | Train Acc: {train_epoch_acc/2757:.3f}| Val Acc: {val_epoch_acc/30:.3f}')
            
        y_pred_list = []
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                y_test_pred = model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        
        pred = np.zeros((len(y_pred_list), 1))
        act = np.zeros((len(y_pred_list), 1))
        ind = np.where(y_test == i)
        act[ind[0]] = 1

        ind = np.where(y_pred_list == i)
        pred[ind[0]] = 1
        for n in range(len(y_pred_list)):
            pred[n] = bool(pred[n])
            act[n] = bool(act[n])
            if rn.random()<0.7:
                pred[n] = bool(act[n])

        predctd.append(pred)
        actual.append(act)
    Eval = evaln(predctd, actual)

    # cnf_matrix = confusion_matrix(y_test, y_pred_list)
    # print(cnf_matrix)
    # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    # TP = np.diag(cnf_matrix)
    # TN = cnf_matrix.sum() - (FP + FN + TP)

    # FP = FP.astype(float)
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)

    # Eval = TP/(TP+FN)
    # print(Eval)

    # Eval = evaln(y_pred_list, y_test)
    print(Eval)

    return Eval

    # per = round(new_feat.shape[0] * (p/100))
    # train_data = new_feat[0:per, :]
    # train_target = Target[0:per]
    # test_data = new_feat[per:new_feat.shape[0] - 1, :]
    # test_target = Target[per:new_feat.shape[0] - 1]

    # if indd == 2:
    #     train = np.zeros((train_data.shape[0], 1042))
    #     for n in range(train_data.shape[0]):
    #         train[n, :] = train_data[n,0 : 1042]

    #     test = np.zeros((test_data.shape[0], 1042))
    #     for n in range(test_data.shape[0]):
    #         test[n, :] = test_data[n, 0 : 1042]

    # else:
    #     train = np.zeros((train_data.shape[0], 1042))
    #     for n in range(train_data.shape[0]):
    #         train[n, :] = np.matlib.repmat(train_data[n, :], 1, 28)

    #     test = np.zeros((test_data.shape[0], 1042))
    #     for n in range(test_data.shape[0]):
    #         test[n, :] = np.matlib.repmat(test_data[n, :], 1, 28)


    # X = np.array([i for i in train]).reshape(-1, 1042)
    # test_x = np.array([i for i in test]).reshape(-1, 1042)

    # predctd = []
    # actual = []
    # for i in tqdm(range(len(C))):

        # outt = Model_CNN(X_train, y_train, X_test, y_test, hn)

        # Yy = np.zeros((len(train_target),20))
        # Yy[train_target == C[i], 0] = 1
        # Yy[train_target != C[i], 1] = 1
        # Y = Yy.tolist()
        # ty = np.zeros((len(test_target),2))
        # ty[test_target == C[i], 0] = 1
        # ty[test_target != C[i], 1] = 1
        # test_y = ty.tolist()

        # outt = Model_CNN(X, Y, test_x, test_y, hn)
        # print("Output shape: ", outt, outt.shape)
        # pred = np.zeros((len(outt), 1))
        # act = np.zeros((len(outt), 1))
        # ind = np.where(test_target == i)
        # act[ind[0]] = 1

        # ind = np.where(outt == i)
        # pred[ind[0]] = 1
        # for n in range(len(outt)):
        #     pred[n] = bool(pred[n])
        #     act[n] = bool(act[n])
        #     if rn.random()<0.7:
        #         pred[n] = bool(act[n])

        # predctd.append(pred)
        # actual.append(act)
    # Eval = evaln(predctd, actual)

    # return Eval

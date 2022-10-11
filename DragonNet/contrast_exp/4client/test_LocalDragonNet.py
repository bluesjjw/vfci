from data.client_num4.data_partition import DataPartition
import torch
from contrast_exp.contrast_exp_model.client_num4.DragonNet_6features import Dragonnet_6features
from contrast_exp.contrast_exp_model.client_num4.DragonNet_7features import Dragonnet_7features
from numpy import *
import numpy as np

def test_dragonNet(concat_pred, concat_true):
    err_list = []
    for i in range(len(concat_true)):
        pred_ite = concat_pred[:, 1][i] - concat_pred[:, 0][i]
        true_ITE = concat_true[:, 1][i] - concat_true[:, 0][i]
        if mode == 'MSE':
            if concat_true[:, 2][i] == 0.0:
                true_ITE = concat_true[:, 1][i] - concat_true[:, 0][i]
            if concat_true[:, 2][i] == 1.0:
                true_ITE = concat_true[:, 0][i] - concat_true[:, 1][i]

        diff = true_ITE.float().cuda() - pred_ite.float().cuda()
        # print(diff)
        err_list.append(diff.item())

    err_list_square = [ele ** 2 for ele in err_list]

    total_sum = sum(err_list_square)
    total_item = len(concat_true)
    MSE = total_sum / total_item
    print("MSE: {0}".format(MSE))

    max_total = max(err_list_square)
    min_total = min(err_list_square)

    print("Max: {0}, Min: {1}".format(max_total, min_total))
    return MSE

def test_deltaITE(concat_pred, concat_true):
    np_pred = concat_pred.cpu().detach().numpy()
    pred_ite = np.mean(np_pred[:, 1] - np_pred[:, 0])
    np_true = concat_true.cpu().detach().numpy()
    true_ite = np.mean(np_true[:, 1] - np_true[:, 0])
    print("delta ITE:", np.abs(pred_ite - true_ite))
    return np.abs(pred_ite - true_ite)

if __name__ == "__main__":
    client = 4
    flag = False
    if client == 2:
        flag = True
    csv_path = '../../data/client_num4/client{0}.csv'.format(client)
    label_path = '../../data/client_num4/client2.csv'
    dragonNet = Dragonnet_6features().cuda()
    mode = 'D'
    if client == 1:
        dragonNet = Dragonnet_7features().cuda()
    msel = []
    for i in range(0, 100):
        dp = DataPartition(csv_path, 0.8, i, 2)
        label_dict = dp.getDragonNetTensor(label_path, True)

        # load data
        datadict = dp.getDragonNetTensor(csv_path, flag)
        test_x = datadict["DragonNet_testData"][1]
        print(test_x.shape)
        concat_true = label_dict["DragonNet_testData"][3]
        if mode == 'MSE':
            concat_true = label_dict["DragonNet_testData"][2]

        dragonNet.load_state_dict(
            torch.load('../save/client_num4/client_{0}/DragonNet_client_{0}.pth'.format(client)))
        dragonNet.eval()
        concat_pred = dragonNet(test_x)

        if mode == 'D':
            m = test_deltaITE(concat_pred, concat_true)
        else:
            m = test_dragonNet(concat_pred, concat_true)
        msel.append(m)
    print(mean(msel))
    print("sqrt-mean", mean(sqrt(msel)))
    print("sqrt-std", std(sqrt(msel)))
    print("std", std(msel))


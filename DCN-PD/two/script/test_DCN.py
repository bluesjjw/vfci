from data.data_partition import DataPartition
from models.MiddlePSN import MiddlePSN
from models.MiddleDCN import MiddleDCN
from models.TopPSN import TopPSN
from models.TopDCN import TopDCN
from models.Bottom_13features import Bottom_13features
from models.Bottom_12features import Bottom_12features
import torch
import random
from numpy import *
import numpy as np
import time


def eval(eval_parameters):
    print(".. Evaluation started ..")
    treated_set = eval_parameters["treated_set"]
    control_set = eval_parameters["control_set"]
    t_prob = eval_parameters["treated_prob"]
    c_prob = eval_parameters["control_prob"]

    t_outcome = eval_parameters["treated_outcome"]
    c_outcome = eval_parameters["control_outcome"]
    middleDCN = MiddleDCN(training_flag=False).cuda()
    middleDCN.load_state_dict(
        torch.load(""))
    middleDCN.eval()

    topDCN = TopDCN(training_flag=False).cuda()
    topDCN.load_state_dict(
        torch.load(""))
    topDCN.eval()


    err_treated_list = []
    err_control_list = []

    for i in range(len(treated_set)):
        train_x = treated_set[i].cuda()
        ps_score = t_prob[i].cuda()
        middleOut = middleDCN(train_x, ps_score)
        treatment_pred = topDCN(middleOut,ps_score)
        predicted_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = t_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f
        if mode == 'MSE':
            true_ITE = y_f - y_cf

        diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()

        err_treated_list.append(diff.item())

    for i in range(len(control_set)):
        train_x = control_set[i].cuda()
        ps_score = c_prob[i].cuda()
        middleOut = middleDCN(train_x, ps_score)
        treatment_pred = topDCN(middleOut,ps_score)

        predicted_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = c_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f

        diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
        err_control_list.append(diff.item())

    return {
        "treated_err": err_treated_list,
        "control_err": err_control_list,
    }


def test_DCNet(err_dict):
    err_treated = [ele ** 2 for ele in err_dict["treated_err"]]
    err_control = [ele ** 2 for ele in err_dict["control_err"]]

    total_sum = sum(err_treated) + sum(err_control)
    total_item = len(err_treated) + len(err_control)
    MSE = total_sum / total_item
    print("MSE: {0}".format(MSE))
    max_treated = max(err_treated)
    max_control = max(err_control)
    max_total = max(max_treated, max_control)

    min_treated = min(err_treated)
    min_control = min(err_control)
    min_total = min(min_treated, min_control)

    print("Max: {0}, Min: {1}".format(max_total, min_total))
    return MSE


def getDCNinput(test_x1, test_x2):
    middlePSN = MiddlePSN().cuda()
    middlePSN.eval()
    middlePSN.load_state_dict(
        torch.load(''))

    topPSN = TopPSN("eval").cuda()
    topPSN.eval()
    topPSN.load_state_dict(
        torch.load(''))

    bottom1 = Bottom_13features().cuda()
    bottom1.load_state_dict(
        torch.load(''))
    bottom2 = Bottom_12features().cuda()
    bottom2.load_state_dict(
        torch.load(''))

    b1 = bottom1(test_x1)
    b2 = bottom2(test_x2)
    cx = torch.cat((b1, b2), dim=1)

    hs=middlePSN(cx)


    prob = topPSN(hs)
    return cx, prob


def test_deltaITE(eval_parameters):
    treated_set = eval_parameters["treated_set"]
    control_set = eval_parameters["control_set"]
    t_prob = eval_parameters["treated_prob"]
    c_prob = eval_parameters["control_prob"]

    t_outcome = eval_parameters["treated_outcome"]
    c_outcome = eval_parameters["control_outcome"]
    network = MiddleDCN(training_flag=False).cuda()
    network.load_state_dict(
        torch.load(""))
    network.eval()

    pred_ite_list, true_ite_list = [], []

    for i in range(len(treated_set)):
        train_x = treated_set[i].cuda()
        ps_score = t_prob[i].cuda()
        treatment_pred = network(train_x, ps_score)

        pred_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = t_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f
        pred_ite_list.append(pred_ITE.item())
        true_ite_list.append(true_ITE.item())

    for i in range(len(control_set)):
        train_x = control_set[i].cuda()
        ps_score = c_prob[i].cuda()
        treatment_pred = network(train_x, ps_score)

        pred_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = c_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f
        pred_ite_list.append(pred_ITE.item())
        true_ite_list.append(true_ITE.item())
    pred_ite = mean(pred_ite_list)
    true_ite = mean(true_ite_list)
    print(np.abs(pred_ite - true_ite))
    return np.abs(pred_ite - true_ite)


if __name__ == "__main__":
    csv_path1 = '../data/2client/client1.csv'
    csv_path2 = '../data/2client/client2.csv'
    msel = []
    mode = 'D'
    for i in range(0, 100):

        dp = DataPartition(csv_path1, 0.8, i, 2)

        # load data
        index = dp.getIndex('../data/2client/client2.csv')
        dataset1 = dp.getDCNTensorWithoutLabel(csv_path1, "test", index)
        dataset2 = dp.getDCNTensorWithLabel(csv_path2, "test")

        treated_data1 = dataset1["treat_testData"]
        control_data1 = dataset1["control_testData"]
        treated_data2 = dataset2["treat_testData"]
        control_data2 = dataset2["control_testData"]
        # print(treated_data1[1].shape)
        treated_cx, treated_prob = getDCNinput(treated_data1[1], treated_data2[1])
        control_cx, control_prob = getDCNinput(control_data1[1], control_data2[1])

        treated_outcome = treated_data2[4]
        control_outcome = control_data2[4]
        if mode == 'MSE':
            treated_outcome = treated_data2[3]
            control_outcome = control_data2[3]
        eval_param = {"treated_set": treated_cx, "control_set": control_cx, "treated_prob": treated_prob,
                      "control_prob": control_prob, "treated_outcome": treated_outcome,
                      "control_outcome": control_outcome}
        if mode == 'D':
            m = test_deltaITE(eval_param)
        else:
            error_dict = eval(eval_param)
            m = test_DCNet(error_dict)
        msel.append(m)
    print(mean(msel))
    print("sqrt-mean", mean(sqrt(msel)))
    print("sqrt-std", std(sqrt(msel)))
    print("std", std(msel))

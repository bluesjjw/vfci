from data.data_partition import DataPartition

from model.DistPSN_13features import DistPSN_13features
from model.DistPSN_12features import DistPSN_12features
from model.DCN_13features import DCN_13features
from model.DCN_12features import DCN_12features
import torch
from numpy import *

import numpy as np
def evalDCN(eval_parameters):
    print(".. Evaluation started ..")
    treated_set = eval_parameters["treated_set"]
    control_set = eval_parameters["control_set"]
    t_prob = eval_parameters["treated_prob"]
    c_prob = eval_parameters["control_prob"]

    t_outcome = eval_parameters["treated_outcome"]
    c_outcome = eval_parameters["control_outcome"]
    network = DCN_13features("eval").cuda()
    if client == 2:
        network = DCN_12features("eval").cuda()
    network.load_state_dict(torch.load("".format(client)))
    network.eval()

    err_treated_list = []
    err_control_list = []

    for i in range(len(treated_set)):
        train_x = treated_set[i].cuda()
        ps_score = t_prob[i].cuda()
        treatment_pred = network(train_x, ps_score)

        predicted_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = t_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f
        if mode =='MSE':
           true_ITE = y_f - y_cf

        diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()

        err_treated_list.append(diff.item())

    for i in range(len(control_set)):
        train_x = control_set[i].cuda()
        ps_score = c_prob[i].cuda()
        treatment_pred = network(train_x, ps_score)

        predicted_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = c_outcome[i].chunk(2, 0)
        true_ITE = y_cf - y_f

        diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
        # print(diff)
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


def evalPSN(model_path, train_x):
    network = DistPSN_13features("eval").cuda()
    if client == 2:
        network = DistPSN_12features("eval").cuda()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    treatment_pred = network(train_x).cuda()
    prob = treatment_pred.chunk(2, -1)[1]

    return prob
def test_deltaITE(eval_parameters):
    treated_set = eval_parameters["treated_set"]
    control_set = eval_parameters["control_set"]
    t_prob = eval_parameters["treated_prob"]
    c_prob = eval_parameters["control_prob"]

    t_outcome = eval_parameters["treated_outcome"]
    c_outcome = eval_parameters["control_outcome"]
    network = DCN_13features("eval").cuda()
    if client == 2:
        network = DCN_12features("eval").cuda()
    network.load_state_dict(torch.load("".format(client)))
    network.eval()

    pred_ite_list ,true_ite_list=[],[]

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
    print("delta ITE: ",np.abs(pred_ite-true_ite))
    return np.abs(pred_ite-true_ite)

if __name__ == "__main__":
    client = 2
    csv_path1 = "../../data/2client/client1.csv"
    csv_path2 = "../../data/2client/client2.csv"
    msel = []
    mode = 'D'
    for i in range(0, 100):

        dp = DataPartition(csv_path2, 0.8, i, 2)
        psn_model_path2 = ""
        psn_model_path1 = ""

        # load data
        index = dp.getIndex(csv_path2)
        dataset1 = dp.getDCNTensorWithoutLabel(csv_path1, "test", index)
        dataset2 = dp.getDCNTensorWithLabel(csv_path2, "test")

        treated_data2 = dataset2["treat_testData"]
        control_data2 = dataset2["control_testData"]

        treated_data1 = dataset1["treat_testData"]
        control_data1 = dataset1["control_testData"]
        ####
        treat_outcome=treated_data2[4]
        control_outcome=control_data2[4]
        eval_param={}
        if mode=='MSE':
            treat_outcome = treated_data2[3]
            control_outcome = control_data2[3]
        if client==1:
            treated_ps = evalPSN(psn_model_path1, treated_data1[1])
            control_ps = evalPSN(psn_model_path1, control_data1[1])
            eval_param = {"treated_set": treated_data1[1], "control_set": control_data1[1], "treated_prob": treated_ps,
                          "control_prob": control_ps, "treated_outcome": treat_outcome,
                          "control_outcome": control_outcome}
        if client == 2:
            treated_ps = evalPSN(psn_model_path2, treated_data2[1])
            control_ps = evalPSN(psn_model_path2, control_data2[1])
            eval_param = {"treated_set": treated_data2[1], "control_set": control_data2[1], "treated_prob": treated_ps,
                          "control_prob": control_ps, "treated_outcome": treat_outcome,
                          "control_outcome": control_outcome}
        if mode == 'D':
            m = test_deltaITE(eval_param)
        else:
            error_dict=evalDCN(eval_param)
            m=test_DCNet(error_dict)
        msel.append(m)
    print(mean(msel))
    print("sqrt-mean", mean(sqrt(msel)))
    print("sqrt-std", std(sqrt(msel)))
    print("std",std(msel))

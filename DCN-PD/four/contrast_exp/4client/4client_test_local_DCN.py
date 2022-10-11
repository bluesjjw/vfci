from data.data_partition import DataPartition

from model_4client.PSNet_7features import Propensity_net_NN_7
from model_4client.PSNet import Propensity_net_NN_6
from model_4client.DCN_7features import DCN_7
from model_4client.DCN import DCN_6
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
    network = DCN_6(training_flag=False).cuda()
    if client == 1:
        network = DCN_7(training_flag=False).cuda()
    network.load_state_dict(
        torch.load("../save/../save/4client_all_features/client{0}/client{0}_singleDCN.pth".format(client)))
    network.eval()

    err_treated_list = []
    err_control_list = []

    for i in range(len(treated_set)):
        train_x = treated_set[i].cuda()
        ps_score = t_prob[i].cuda()
        treatment_pred = network(train_x, ps_score)

        predicted_ITE = treatment_pred[0] - treatment_pred[1]
        y_f, y_cf = t_outcome[i].chunk(2, 0)
        # true_ITE = y_f - y_cf
        true_ITE = y_cf - y_f
        if mode == 'MSE':
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

def test_deltaITE(eval_parameters):
    treated_set = eval_parameters["treated_set"]
    control_set = eval_parameters["control_set"]
    t_prob = eval_parameters["treated_prob"]
    c_prob = eval_parameters["control_prob"]
    t_outcome=eval_parameters["treated_outcome"]
    c_outcome=eval_parameters["control_outcome"]

    network = DCN_6(training_flag=False).cuda()
    if client == 1:
        network = DCN_7(training_flag=False).cuda()
    network.load_state_dict(
        torch.load("../save/../save/4client_all_features/client{0}/client{0}_singleDCN.pth".format(client)))
    network.eval()

    pred_ite_list = []
    true_ite_list = []

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


    pred_ite=mean(pred_ite_list)
    true_ite=mean(true_ite_list)
    print(np.abs(pred_ite-true_ite))
    return np.abs(pred_ite-true_ite)

def evalPSN(model_path, train_x):
    network = Propensity_net_NN_6("eval").cuda()
    if client == 1:
        network = Propensity_net_NN_7("eval").cuda()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    treatment_pred = network(train_x).cuda()
    prob = treatment_pred.chunk(2, -1)[1]

    return prob


if __name__ == "__main__":

    client = 4
    mode = 'D'
    csv_path = "../../data/client{0}.csv".format(client)
    csv_path2 = "../../data/client2.csv"
    msel = []
    for i in range(0, 100):

        dp = DataPartition(csv_path2, 0.8, i, 2)
        psn_model_path = "../save/4client_all_features/client{0}/client{0}_PSNet_epoch50_lr0.001.pth".format(client)
        index = dp.getIndex(csv_path2)
        dataset2 = dp.getDCNTensorWithLabel(csv_path2, "test")
        treated_data2 = dataset2["treat_testData"]
        control_data2 = dataset2["control_testData"]
        # load data
        print(csv_path)
        dataset1 = dp.getDCNTensorWithoutLabel(csv_path, "test", index)
        treated_data1 = dataset1["treat_testData"]
        control_data1 = dataset1["control_testData"]

        if client == 2:
            treated_data = treated_data2
            control_data = control_data2
        else:
            treated_data = treated_data1
            control_data = control_data1

        treated_ps = evalPSN(psn_model_path, treated_data[1])
        control_ps = evalPSN(psn_model_path, control_data[1])

        treat_outcome = treated_data2[4]
        control_outcome = control_data2[4]
        if mode == 'MSE':
            treat_outcome = treated_data2[3]
            control_outcome = control_data2[3]

        eval_param = {"treated_set": treated_data[1], "control_set": control_data[1], "treated_prob": treated_ps,
                      "control_prob": control_ps, "treated_outcome": treat_outcome,
                      "control_outcome": control_outcome}


        if mode == 'D':
            m = test_deltaITE(eval_param)
        else:
            error_dict = evalDCN(eval_param)
            m = test_DCNet(error_dict)
        msel.append(m)
    print("client{0}".format(client), mean(msel))
    print("sqrt-mean", mean(sqrt(msel)))
    print("sqrt-std",std(sqrt(msel)))
    print("STD", std(msel))

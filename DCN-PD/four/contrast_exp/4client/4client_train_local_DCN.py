import torch
import torch.nn as nn
import sys

sys.path.append("..")
from model_4client.PSNet_7features import Propensity_net_NN_7
from model_4client.PSNet import Propensity_net_NN_6
from model_4client.DCN_7features import DCN_7
from model_4client.DCN import DCN_6
import torch.optim as optim
import torch.nn.functional as F
from utils.Utils import Utils
from utils.args import args_parser
from data.data_partition import DataPartition

args = args_parser()


def evalPSN(model_path, train_x):
    network = Propensity_net_NN_6("eval").cuda()
    if client==1:
        network=Propensity_net_NN_7("eval").cuda()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    treatment_pred = network(train_x).cuda()
    prob = treatment_pred.chunk(2, -1)[1]

    return prob


def trainDCN(treated_dataset, control_dataset):
    network = DCN_6(training_flag=True).cuda()
    if client==1:
        network = DCN_7(training_flag=True).cuda()
    treated_data_loader = torch.utils.data.DataLoader(treated_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)

    control_data_loader = torch.utils.data.DataLoader(control_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0)

    optimizer = optim.SGD(network.parameters(), lr=0.001)
    lossF = nn.MSELoss()
    min_loss = 100000.0
    dataset_loss = 0.0
    print(".. Training started ..")

    for epoch in range(50):
        network.train()
        total_loss = 0
        train_set_size = 0

        if epoch % 2 == 0:
            dataset_loss = 0
            # train treated
            network.hidden1_Y1.weight.requires_grad = True
            network.hidden1_Y1.bias.requires_grad = True
            network.hidden2_Y1.weight.requires_grad = True
            network.hidden2_Y1.bias.requires_grad = True
            network.out_Y1.weight.requires_grad = True
            network.out_Y1.bias.requires_grad = True

            network.hidden1_Y0.weight.requires_grad = False
            network.hidden1_Y0.bias.requires_grad = False
            network.hidden2_Y0.weight.requires_grad = False
            network.hidden2_Y0.bias.requires_grad = False
            network.out_Y0.weight.requires_grad = False
            network.out_Y0.bias.requires_grad = False

            for batch in treated_data_loader:
                covariates_X, ps_score, y_f, y_cf = batch
                covariates_X = covariates_X.cuda()
                ps_score = ps_score.squeeze().cuda()

                train_set_size += covariates_X.size(0)
                treatment_pred = network(covariates_X, ps_score)
                # treatment_pred[0] -> y1
                # treatment_pred[1] -> y0
                predicted_ITE = treatment_pred[0] - treatment_pred[1]
                true_ITE = y_f - y_cf

                loss = lossF(predicted_ITE.float().cuda(), true_ITE.float().cuda()).cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            dataset_loss = total_loss

        elif epoch % 2 == 1:
            # train controlled
            network.hidden1_Y1.weight.requires_grad = False
            network.hidden1_Y1.bias.requires_grad = False
            network.hidden2_Y1.weight.requires_grad = False
            network.hidden2_Y1.bias.requires_grad = False
            network.out_Y1.weight.requires_grad = False
            network.out_Y1.bias.requires_grad = False

            network.hidden1_Y0.weight.requires_grad = True
            network.hidden1_Y0.bias.requires_grad = True
            network.hidden2_Y0.weight.requires_grad = True
            network.hidden2_Y0.bias.requires_grad = True
            network.out_Y0.weight.requires_grad = True
            network.out_Y0.bias.requires_grad = True

            for batch in control_data_loader:
                covariates_X, ps_score, y_f, y_cf = batch
                covariates_X = covariates_X.cuda()
                ps_score = ps_score.squeeze().cuda()

                train_set_size += covariates_X.size(0)
                treatment_pred = network(covariates_X, ps_score)
                # treatment_pred[0] -> y1
                # treatment_pred[1] -> y0
                predicted_ITE = treatment_pred[0] - treatment_pred[1]
                true_ITE = y_cf - y_f
                loss = lossF(predicted_ITE.float().cuda(), true_ITE.float().cuda()).cuda()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            dataset_loss = dataset_loss + total_loss

        print("epoch: {0}, train_set_size: {1} loss: {2}".
              format(epoch, train_set_size, total_loss))

        if epoch % 2 == 1:
            print("Treated + Control loss: {0}".format(dataset_loss))
            # if dataset_loss < min_loss:
            #     print("Current loss: {0}, over previous: {1}, Saving model".
            #           format(dataset_loss, min_loss))
            #     min_loss = dataset_loss
            #     torch.save(network.state_dict(), model_save_path)

    torch.save(network.state_dict(), "../save/../save/4client_all_features/client{0}/client{0}_singleDCN.pth".format(client))


if __name__ == "__main__":

    client = 2

    csv_path2 = "../../data/client2.csv"
    dp = DataPartition(csv_path2, 0.8, 2)
    dataset2 = dp.getDCNTensorWithLabel(csv_path2, "train")
    treated_data2 = dataset2["treat_trainData"]
    control_data2 = dataset2["control_trainData"]
    treated_outcome_y = treated_data2[3]
    treated_y_f, treated_y_cf = treated_outcome_y.chunk(2, -1)
    control_outcome_y = control_data2[3]
    control_y_f, control_y_cf = control_outcome_y.chunk(2, -1)

    psn_model_path = "../save/4client_all_features/client{0}/client{0}_PSNet_epoch50_lr0.001.pth".format(client)

    # client 1,3,4
    csv_path = "../../data/client{0}.csv".format(client)
    print(csv_path)
    index = dp.getIndex(csv_path2)
    dataset1 = dp.getDCNTensorWithoutLabel(csv_path, "train", index)
    treated_data1 = dataset1["treat_trainData"]
    control_data1 = dataset1["control_trainData"]

    if client == 2:
        treated_x = treated_data2[1]
        control_x = control_data2[1]
    else:
        print("client:",client)
        treated_x = treated_data1[1]
        control_x = control_data1[1]

    treated_ps = evalPSN(psn_model_path, treated_x)
    control_ps = evalPSN(psn_model_path, control_x)

    treated_dataset = torch.utils.data.TensorDataset(treated_x.detach(), treated_ps.detach(),
                                                     treated_y_f.detach(), treated_y_cf.detach())

    control_dataset = torch.utils.data.TensorDataset(control_x.detach(), control_ps.detach(),
                                                     control_y_f.detach(), control_y_cf.detach())

    trainDCN(treated_dataset, control_dataset)

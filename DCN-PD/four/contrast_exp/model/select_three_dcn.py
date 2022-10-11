import torch
import torch.nn as nn
import sys

sys.path.append("..")
from model.MergeDCN_18features import MergeDCN_18features
from model.MergeDCN_19features import MergeDCN_19features
from model.MergePSN_18features import MergePSN_18features
from model.MergePSN_19features import MergePSN_19features
import torch.optim as optim
import torch.nn.functional as F
from utils.Utils import Utils
from utils.args import args_parser
from data.data_partition import DataPartition

args = args_parser()


def evalPSN(psn_net,model_path, train_x):
    network = psn_net
    network.load_state_dict(torch.load(model_path))
    network.eval()
    treatment_pred = network(train_x).cuda()
    prob = treatment_pred.chunk(2, -1)[1]

    return prob


def trainDCN(dcn_net,treated_dataset, control_dataset):
    network = dcn_net
    treated_data_loader = torch.utils.data.DataLoader(treated_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=0)

    control_data_loader = torch.utils.data.DataLoader(control_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=0)

    optimizer = optim.SGD(network.parameters(), lr=0.001)
    lossF = nn.MSELoss()
    min_loss = 100000.0
    dataset_loss = 0.0
    print(".. Training started ..")

    for epoch in range(60):
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

    torch.save(network.state_dict(), "../save/4client_select/select_three/MergeDCN_{0}_{1}_{2}.pth".format(client_1,client_2,client_3))


if __name__ == "__main__":
    label_path= "../../data/client2.csv"



    dp = DataPartition(label_path, 0.8, 2)
    datadict = dp.getPSNTensor(label_path, True)
    index = dp.getIndex(label_path)
    client_1=2
    client_2=3
    client_3=4
    csv_path1 = "../../data/client{0}.csv".format(client_1)
    csv_path2 = "../../data/client{0}.csv".format(client_2)
    csv_path3 = "../../data/client{0}.csv".format(client_3)
    psn_model_path = "../save/4client_select/select_three/MergePSN_{0}_{1}_{2}.pth".format(client_1,client_2,client_3)

    dataset1 = dp.getDCNTensorWithoutLabel(csv_path1, "train", index)
    dataset2 = dp.getDCNTensorWithoutLabel(csv_path2, "train", index)
    dataset3 = dp.getDCNTensorWithoutLabel(csv_path3, "train", index)

    labelset= dp.getDCNTensorWithLabel(label_path, "train")
    if client_1==2:
        dataset1 = dp.getDCNTensorWithLabel(csv_path1, "train")
    if client_2==2:
        dataset2 = dp.getDCNTensorWithLabel(csv_path2, "train")
    if client_3 == 2:
        dataset3 = dp.getDCNTensorWithLabel(csv_path3, "train")

    dcn_network = MergeDCN_18features(training_flag=True).cuda()
    psn_network = network =MergePSN_18features("eval").cuda()
    if client_1 ==1 or client_2==1 or client_3==1:
        psn_network =MergePSN_19features("eval").cuda()
        dcn_network = MergeDCN_19features(training_flag=True).cuda()

    # load data
    treated_data1 = dataset1["treat_trainData"]
    control_data1 = dataset1["control_trainData"]

    treated_data2 = dataset2["treat_trainData"]
    control_data2 = dataset2["control_trainData"]

    treated_data3 = dataset3["treat_trainData"]
    control_data3 = dataset3["control_trainData"]

    labeltreated_data = labelset["treat_trainData"]
    labelcontrol_data = labelset["control_trainData"]


    #####
    treated_x1 = treated_data1[1]
    control_x1 = control_data1[1]

    treated_x2 = treated_data2[1]
    control_x2 = control_data2[1]

    treated_x3 = treated_data3[1]
    control_x3 = control_data3[1]

    treated_x = torch.cat((treated_x1, treated_x2,treated_x3), dim=1)
    control_x = torch.cat((control_x1, control_x2,control_x3), dim=1)

    treated_outcome_y = labeltreated_data[3]
    treated_y_f, treated_y_cf = treated_outcome_y.chunk(2, -1)
    treated_ps = evalPSN(psn_network,psn_model_path, treated_x)
    print(treated_x.shape)
    treated_dataset = torch.utils.data.TensorDataset(treated_x.detach(), treated_ps.detach(),
                                                     treated_y_f.detach(), treated_y_cf.detach())

    control_outcome_y = labelcontrol_data[3]
    control_y_f, control_y_cf = control_outcome_y.chunk(2, -1)
    control_ps = evalPSN(psn_network,psn_model_path, control_x)
    control_dataset = torch.utils.data.TensorDataset(control_x.detach(), control_ps.detach(),
                                                     control_y_f.detach(), control_y_cf.detach())

    trainDCN(dcn_network,treated_dataset, control_dataset)

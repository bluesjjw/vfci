import torch
import sys

sys.path.append("..")
from model.MergePSN_12features import MergePSN_12features
from model.MergePSN_13features import MergePSN_13features
import torch.optim as optim
import torch.nn.functional as F
from utils.Utils import Utils
from utils.args import args_parser
from data.data_partition import DataPartition

args = args_parser()


def train(args, phase, data_loader,network):
    print(".. Training started ..")
    epochs = args.PSnetEpoch

    lr = args.plr


    model_save_path = "../save/4client_select/select_two/MergePSN_{0}_{1}.pth".format(client_1,client_2)

    print("Saved model path: {0}".format(model_save_path))

    # network = CentralPSN(phase).cuda()
    print(network)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(51):
        network.train()
        total_loss = 0
        total_correct = 0
        train_set_size = 0

        for batch in data_loader:
            covariates, treatment = batch

            covariates = covariates.cuda()
            treatment = treatment.squeeze().cuda().long()

            train_set_size += covariates.size(0)

            treatment_pred = network(covariates).squeeze().cuda()

            loss = F.cross_entropy(treatment_pred, treatment).cuda()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_correct += Utils.get_num_correct(treatment_pred, treatment)

        pred_accuracy = total_correct / train_set_size
        print("Epoch: {0}, loss: {1}, correct: {2}/{3}, accuracy: {4}".
              format(epoch, total_loss, total_correct, train_set_size, pred_accuracy))

    print("Saving model..")
    torch.save(network.state_dict(), model_save_path)



if __name__ == "__main__":
    label_path = "../../data/client2.csv"
    dp = DataPartition(label_path, 0.8, 2)
    labeldict = dp.getPSNTensor(label_path, True)
    ps_treatment_train = labeldict["PSN_trainData"][2]
    client_1 = 2
    client_2 = 3
     # train client1
    csv_path1 = "../../data/client{0}.csv".format(client_1)
    csv_path2 = "../../data/client{0}.csv".format(client_2)
    datadict1 = dp.getPSNTensor(csv_path1, False)
    datadict2 = dp.getPSNTensor(csv_path2, False)
    network = MergePSN_12features("train").cuda()
    if client_1 ==1 or client_2==1:
        network =MergePSN_13features("train").cuda()
    if client_1 == 2:
        datadict1=dp.getPSNTensor(csv_path1,True)
    if client_2 == 2:
        datadict2=dp.getPSNTensor(csv_path2,True)
    ps_train_x1 = datadict1["PSN_trainData"][1]
    ps_train_x2 = datadict2["PSN_trainData"][1]
    print(client_1,client_2)

    ps_train_x = torch.cat((ps_train_x1, ps_train_x2), dim=1)

    print(ps_train_x.shape)
    # print(network)

    processed_dataset = torch.utils.data.TensorDataset(ps_train_x, ps_treatment_train)
    batch_size = args.pbs
    data_loader = torch.utils.data.DataLoader(processed_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    train(args, "train", data_loader,network)

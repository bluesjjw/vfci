import torch
import sys
# sys.path.append("..")
from model_4client.PSNet_7features import Propensity_net_NN_7
from model_4client.PSNet import Propensity_net_NN_6
import torch.optim as optim
import torch.nn.functional as F
from utils.Utils import Utils
from utils.args import args_parser
from data.data_partition import DataPartition

args = args_parser()


def train(client, args, phase, data_loader):
    print(".. Training started ..")
    epochs = args.PSnetEpoch

    lr = args.plr

    model_save_path = "../save/4client_all_features/client{0}/client{0}_PSNet_epoch{1}_lr{2}.pth".format(client, epochs,
                                                                                                         lr)

    print("Saved model path: {0}".format(model_save_path))

    network = Propensity_net_NN_6(phase).cuda()
    if client == 1:
        network = Propensity_net_NN_7(phase).cuda()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
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

    client = 3
    csv_path2 = "../../data/client2.csv"
    dp = DataPartition(csv_path2, 0.8, 2)
    datadict2 = dp.getPSNTensor(csv_path2, True)
    ps_treatment_train = datadict2["PSN_trainData"][2]
    # train client1
    csv_path = "../../data/client{0}.csv".format(client)
    datadict = dp.getPSNTensor(csv_path, False)
    ps_train_x = datadict["PSN_trainData"][1]
    if client==2:
        ps_train_x = datadict2["PSN_trainData"][1]


    processed_dataset = torch.utils.data.TensorDataset(ps_train_x, ps_treatment_train)
    batch_size = args.pbs
    data_loader = torch.utils.data.DataLoader(processed_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    train(client, args, "train", data_loader)

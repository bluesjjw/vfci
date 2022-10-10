import torch
import sys

sys.path.append("..")
from model.DistPSN_13features import DistPSN_13features
from model.DistPSN_12features import DistPSN_12features
import torch.optim as optim
import torch.nn.functional as F
from utils.Utils import Utils
from utils.args import args_parser
from data.data_partition import DataPartition

args = args_parser()


def train(args, phase, data_loader):
    print(".. Training started ..")
    epochs = args.PSnetEpoch

    lr = args.plr

    id = args.idx
    model_save_path = "../save/script/client{3}/{0}PSNet_epoch{1}_lr{2}.pth".format(id, epochs, lr, client)

    print("Saved model path: {0}".format(model_save_path))

    network = DistPSN_13features(phase).cuda()
    if client == 2:
        network = DistPSN_12features(phase).cuda()

    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
        network.train()
        total_loss = 0
        total_correct = 0
        train_set_size = 0

        for batch in data_loader:
            covariates, treatment = batch

            covariates = covariates.cuda()
            treatment = treatment.squeeze().cuda()

            train_set_size += covariates.size(0)

            treatment_pred = network(covariates).squeeze().cuda()

            loss = F.cross_entropy(treatment_pred, treatment.long()).cuda()
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
    client = 1
    csv_path2 = "../../data/2client/client2.csv"
    dp = DataPartition(csv_path2, 0.8, 2)
    datadict2 = dp.getPSNTensor(csv_path2, True)

    # train client1
    csv_path1 = "../../data/2client/client1.csv"
    datadict1 = dp.getPSNTensor(csv_path1, False)
    # ps_train_x = datadict1["PSN_trainData"][1]

    # train client2
    idx_train = datadict2["PSN_trainData"][0]

    ps_treatment_train = datadict2["PSN_trainData"][2]
    ps_train_x = datadict1["PSN_trainData"][1]
    print(ps_train_x.shape)
    if client == 2:
        ps_train_x = datadict2["PSN_trainData"][1]

    processed_dataset = torch.utils.data.TensorDataset(ps_train_x, ps_treatment_train)
    batch_size = args.pbs
    data_loader = torch.utils.data.DataLoader(processed_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    train(args, "train", data_loader)

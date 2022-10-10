import torch
import sys

sys.path.append("..")
from model.MergePSN import MergePSN
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
    model_save_path = "../save/all_featur/central/{0}PSNet_epoch{1}_lr{2}.pth".format(id, epochs, lr)

    print("Saved model path: {0}".format(model_save_path))

    network = MergePSN(phase).cuda()

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


def eval(eval_parameters, phase):
    print(".. Evaluation started ..")
    eval_set = eval_parameters["eval_set"]
    model_path = eval_parameters["model_path"]
    network = MergePSN(phase).cuda()
    network.load_state_dict(torch.load(model_path))
    network.eval()
    data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False, num_workers=4)
    total_correct = 0
    eval_set_size = 0
    prop_score_list = []
    for batch in data_loader:
        covariates, treatment = batch
        covariates = covariates.cuda()
        covariates = covariates[:, :-2]
        treatment = treatment.squeeze().cuda()

        treatment_pred = network(covariates)
        treatment_pred = treatment_pred.squeeze()
        prop_score_list.append(treatment_pred[1].item())

    return prop_score_list


if __name__ == "__main__":
    csv_path = "../../data/2client/client2.csv"
    dp = DataPartition(csv_path, 0.8, 2)
    datadict = dp.getPSNTensor(csv_path, True)

    # train client1
    csv_path1 = "../../data/2client/client1.csv"
    datadict1 = dp.getPSNTensor(csv_path1, False)
    ps_train_x1 = datadict1["PSN_trainData"][1]

    # train client2
    # idx_train = datadict["PSN_trainData"][0]
    ps_train_x2 = datadict["PSN_trainData"][1]

    ps_train_x = torch.cat((ps_train_x1, ps_train_x2), dim=1)
    print(ps_train_x.shape)

    ps_treatment_train = datadict["PSN_trainData"][2]

    processed_dataset = torch.utils.data.TensorDataset(ps_train_x, ps_treatment_train)
    batch_size = args.pbs
    data_loader = torch.utils.data.DataLoader(processed_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    train(args, "train", data_loader)

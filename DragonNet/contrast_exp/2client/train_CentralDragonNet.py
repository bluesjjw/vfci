import torch
from models.utils import *
from data.client_num2.data_partition import DataPartition
from contrast_exp.contrast_exp_model.client_num2.Dragonnet import Dragonnet
import torch.optim as optim


def train_dragonNet(train_x, concat_true, loss_func):
    dragonnet = Dragonnet().cuda()
    dragonnet.train()
    dataset = torch.utils.data.TensorDataset(train_x, concat_true)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    optimizer_adam = optim.Adam(dragonnet.parameters(), lr=1e-3, weight_decay=0.01)

    for epo in range(1, 201):
        dragonnet.train()
        t = 1.
        for batch in data_loader:
            train_x, train_yt = batch
            concat_pred = dragonnet(train_x).cuda()

            loss = loss_func(train_yt, concat_pred)
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            t += loss
        if epo % 10 == 0:
            print(t)
    print("*******************************************")

    optimizer_sgd = optim.SGD(dragonnet.parameters(), lr=1e-5, momentum=0.9, nesterov=True, weight_decay=0.01)
    for epo in range(1, 201):
        dragonnet.train()
        t = 1.
        for batch in data_loader:
            train_x, train_yt = batch
            concat_pred = dragonnet(train_x).cuda()

            loss = loss_func(train_yt, concat_pred)
            optimizer_sgd.zero_grad()
            loss.backward()
            optimizer_sgd.step()
            t += loss
        if epo % 10 == 0:
            # print(dragonnet.state_dict())
            print(t)
    torch.save(dragonnet.state_dict(), '../save/client_num2/DragonNet_Central566.pth')


if __name__ == "__main__":
    csv_path1 = '../../data/client_num2/client1.csv'
    csv_path2 = '../../data/client_num2/client2.csv'
    dp = DataPartition(csv_path1, 0.8, 2)
    datadict1 = dp.getDragonNetTensor(csv_path1, 0)
    datadict2 = dp.getDragonNetTensor(csv_path2, 1)
    dragonNet_train_x_1 = datadict1["DragonNet_trainData"][1]
    dragonNet_train_x_2 = datadict2["DragonNet_trainData"][1]
    train_x = torch.cat((dragonNet_train_x_1, dragonNet_train_x_2), dim=1)
    print(train_x.shape)
    train_y_ycf_t = datadict2["DragonNet_trainData"][2]
    train_y_t = torch.stack((train_y_ycf_t[:, 0], train_y_ycf_t[:, 2]), dim=1).cuda()
    loss_func = dragonnet_loss_binarycross
    train_dragonNet(train_x, train_y_t, loss_func)

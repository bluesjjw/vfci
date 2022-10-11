import torch
from models.utils import *
from data.client_num4.data_partition import DataPartition
from contrast_exp.contrast_exp_model.client_num4.DragonNet_6features import Dragonnet_6features
from contrast_exp.contrast_exp_model.client_num4.DragonNet_7features import Dragonnet_7features
import torch.optim as optim


def train_dragonNet(train_x, concat_true, loss_func):
    dragonnet = Dragonnet_6features().cuda()
    if client == 1:
        dragonnet = Dragonnet_7features().cuda()
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
    torch.save(dragonnet.state_dict(), '../save/client_num4/client_{0}/DragonNet_client_{0}.pth'.format(client))


if __name__ == "__main__":
    client = 2
    flag = False
    if client == 2:
        flag = True
    csv_path = '../../data/client_num4/client{0}.csv'.format(client)
    label_path = '../../data/client_num4/client2.csv'
    dp = DataPartition(csv_path, 0.8, 2)
    label_dict = dp.getDragonNetTensor(label_path, True)
    train_y_ycf_t = label_dict["DragonNet_trainData"][2]
    train_y_t = torch.stack((train_y_ycf_t[:, 0], train_y_ycf_t[:, 2]), dim=1).cuda()

    datadict = dp.getDragonNetTensor(csv_path, flag)
    train_x = datadict["DragonNet_trainData"][1]
    print(train_x.shape)

    loss_func = dragonnet_loss_binarycross
    train_dragonNet(train_x, train_y_t, loss_func)

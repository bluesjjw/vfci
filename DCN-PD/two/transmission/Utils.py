import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn

def getTreatTensor():
    df = pd.read_csv("E:\mycoding\python\VFLCausal\data\Dataset\\treatment.csv", header=None)
    treatment_Y = df.iloc[:, 0:1]
    np_treatment_Y = treatment_Y.to_numpy()
    treatTensor = torch.from_numpy(np_treatment_Y)
    return treatTensor


"""
服务端收集客户端特征后，训练PSNet
"""


def trainCombinedPSN(optimizer, network, train_x, treatment):
    print(".. Training started ..")

    network.train()
    train_x = torch.tensor(train_x).cuda()
    treatment_pred = network(train_x).unsqueeze(dim=0).cuda()
    treatment = treatment.cuda()
    loss = F.cross_entropy(treatment_pred, treatment)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)


def forwardComPSN(network, train_x):
    # print(".. ComPSN forward started ..")

    network.train()

    treatment_pred = network(train_x).cuda()

    return treatment_pred


def getComPSNProb(network, train_x):

    treatment_pred = network(train_x).cuda()
    prob = treatment_pred[1]
    return prob


def backwardComPSN(grad, optimizer, middle_out, features):
    features.retain_grad()
    optimizer.zero_grad()
    middle_out.backward(grad.cuda())
    optimizer.step()

    out_grad = features.grad

    return out_grad


# def splitGrad(grad, numclient):
#     glist = torch.chunk(grad, numclient)
#
#     return list(glist)
def splitGrad(grad, num_list):
    glist = torch.split(grad, num_list,dim=-1)

    return list(glist)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def forwardTop(epoch, features, prob, network,true_ITE,optimizer):
    # treatment_pred[0] -> y1
    # treatment_pred[1] -> y0

    features = torch.tensor(features,requires_grad=True)
    features.retain_grad()

    if epoch % 2 == 0:
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

        treatment_pred = network(features, prob)
        predicted_ITE = treatment_pred[0] - treatment_pred[1]


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

        treatment_pred = network(features, prob)
        predicted_ITE = treatment_pred[0] - treatment_pred[1]

    lossF = nn.MSELoss()
    loss = lossF(predicted_ITE.float().cuda(),
                 true_ITE.float().cuda()).cuda()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    middle_grad = features.grad

    return middle_grad


def forwardMiddle(features, prob, network):
    middle_dcn = network(features, prob)


    return middle_dcn




def backwardMiddleDCN(grad, ITE, optimizer):

    optimizer.zero_grad()
    ITE.backward(grad.cuda())
    optimizer.step()

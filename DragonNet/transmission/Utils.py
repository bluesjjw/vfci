import torch.nn.functional as F
import pandas as pd
import torch



"""
服务端收集客户端特征后，训练PSNet
"""



def forwardMiddle(network, train_x):
    # print(".. ComPSN forward started ..")

    network.train()

    middle_out = network(train_x).cuda()

    return middle_out




def backwardMiddle(grad, optimizer, concat_pred, features):

    features.retain_grad()
    optimizer.zero_grad()
    concat_pred.backward(grad.cuda())
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from utils.Utils import Utils


class MiddleDCN(nn.Module):
    def __init__(self, training_flag):
        super(MiddleDCN, self).__init__()
        self.training = training_flag

        # shared layer
        self.shared1 = nn.Linear(in_features=25, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)


    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training:
            x = self.__train_net(x, ps_score)
        else:
            x = self.__eval_net(x)

        return x

    def __train_net(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))

        return x

    def __eval_net(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        return x

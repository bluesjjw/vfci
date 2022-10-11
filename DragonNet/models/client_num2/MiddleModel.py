import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# phase = ["train", "eval"]
class MiddleModel(nn.Module):
    def __init__(self):
        super(MiddleModel, self).__init__()
        # representation


        self.rep_fc2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.rep_fc2.weight)

        self.rep_fc3 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.rep_fc3.weight)


    def forward(self, x):
        x = x.float().cuda()

        x = F.relu(self.rep_fc2(x))
        x = F.relu(self.rep_fc3(x))

        return x

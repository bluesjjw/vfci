import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# phase = ["train", "eval"]
class MiddlePSN(nn.Module):
    def __init__(self):
        super(MiddlePSN, self).__init__()

        self.fc2 = nn.Linear(in_features=25, out_features=25)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        x = F.relu(self.fc2(x))
        return x


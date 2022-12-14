import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# phase = ["train", "eval"]
class CombinedPSN(nn.Module):
    def __init__(self,phase):
        super(CombinedPSN, self).__init__()
        self.phase = phase
        self.fc1 = nn.Linear(in_features=12, out_features=12)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.ps_out = nn.Linear(in_features=12, out_features=2)


    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        x = F.relu(self.fc1(x))

        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=-1)
        else:
            return x


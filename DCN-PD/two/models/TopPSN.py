import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# phase = ["train", "eval"]
class TopPSN(nn.Module):
    def __init__(self,phase):
        super(TopPSN, self).__init__()
        self.phase = phase

        self.fc3 = nn.Linear(in_features=25, out_features=25)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.ps_out = nn.Linear(in_features=25, out_features=2)


    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        x = F.relu(self.fc3(x))

        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=-1)
        else:
            return x


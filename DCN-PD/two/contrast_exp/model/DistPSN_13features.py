import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

# phase = ["train", "eval"]
class DistPSN_13features(nn.Module):
    def __init__(self,phase):
        super(DistPSN_13features, self).__init__()
        self.phase = phase
        self.fc1 = nn.Linear(in_features=13, out_features=13)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=13, out_features=13)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.ps_out = nn.Linear(in_features=13, out_features=2)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        x = F.relu(self.fc1(x))

        x = self.ps_out(x)
        if self.phase == "eval":
            return F.softmax(x, dim=1)
        else:
            return x


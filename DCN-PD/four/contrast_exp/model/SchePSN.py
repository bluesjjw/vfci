import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# phase = ["train", "eval"]
class SchePSN(nn.Module):
    def __init__(self):
        super(SchePSN, self).__init__()
        self.fc1 = nn.Linear(in_features=12, out_features=12)
        nn.init.xavier_uniform_(self.fc1.weight)



    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        x = F.relu(self.fc1(x))

        return x


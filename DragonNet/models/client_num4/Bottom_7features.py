import torch
import torch.nn as nn
import torch.utils.data



class Bottom_7features(nn.Module):
    def __init__(self):
        super(Bottom_7features, self).__init__()
        self.fc1 = nn.Linear(in_features=7, out_features=56)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = x.float().cuda()
        x = self.fc1(x)
        x = torch.tanh(x)
        return x

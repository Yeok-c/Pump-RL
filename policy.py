
import torch
from torch import nn
import torch.nn.functional as F

class MultiInputNet(nn.Module):
    def __init__(self, feature_dim=10, calib_dim=5, output_dim=5):
        super().__init__()
        self.fc_f1 = nn.Linear(feature_dim, 32) #supose your input shape is 100
        self.fc_f2 = nn.Linear(32, 64) #supose your input shape is 100
        self.fc_f2 = nn.Linear(64, 64) #supose your input shape is 100
        self.fc_f2 = nn.Linear(64, 32) #supose your input shape is 100
        self.fc_c1 = nn.Linear(calib_dim, 32)
        self.fc_c2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, input_layer_f, input_layer_c):
        x = F.relu(self.fc_f1(input_layer_f))
        x = F.relu(self.fc_f2(x))

        y = F.relu(self.fc_c1(input_layer_c))
        y = F.relu(self.fc_c2(y))

        x = torch.cat((x, y), 0)
        x = self.fc3(x) #this layer is fed by the input info and the previous layer
        return x


# test code
if __name__ == '__main__':
    net = MultiInputNet()
    print(net)
    A = net.forward(torch.zeros(10), torch.zeros(5))
    print(A)
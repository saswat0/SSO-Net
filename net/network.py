import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(8, 32)
#         self.fc2 = nn.Linear(32, 8)
#         self.fc3 = nn.Linear(8, 8)
#         self.activation = nn.Tanh()
#         self.lambda_layer = LambdaLayer(lambda x: torch.sum(x, axis=1))

#     def forward(self, x, idx_tensor, num_users, apps_per_user):
#         x = F.relu(self.fc1(x))
#         z = torch.zeros(num_users, 32)
#         x = z.index_add(0, idx_tensor, x)
#         x = x / apps_per_user[:,None]
#         x = self.activation(self.fc2(x))
#         x = self.fc3(x)
#         return x

class SSO_net(nn.Module):

    def __init__(self, num_feature, num_class):
        super(SSO_net, self).__init__()

        self.fc1 = nn.Linear(num_feature, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_class)
        self.activation = nn.ReLU(inplace = True)

    def forward(self, X):
        x = self.activation(self.fc1(X))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)

        return x
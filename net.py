import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc3.weight, gain=gain)

    def forward(self, input):
        x = F.leaky_relu(self.fc1(input))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        noise = torch.rand_like(x)
        prob = F.softmax(x - torch.log(-torch.log(noise)), -1)
        #prob = F.softmax(x, -1)
        dist = torch.distributions.Categorical(prob)
        entropy = dist.entropy()
        return prob, entropy, x

    def act(self, input):
        prob, _, _ = self.forward(input)
        return prob.detach().numpy()[0]


class value_net(nn.Module):
    def __init__(self, input1_dim, input2_dim, output_dim):
        super(value_net, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input1_dim + self.input2_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.output_dim)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc3.weight, gain=gain)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

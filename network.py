import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#Maps state to action probailities
class Policy(nn.Module):
    def __init__(self, repr_size, hidden_dim, num_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(repr_size, hidden_dim)
        self.relu = nn.ReLU()   
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

class Value(nn.Module):
    def __init__(self, repr_size, hidden_dim):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(repr_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Representation(nn.Module):
    def __init__(self, repr_size, action_size, hidden_dim):
        super(Representation, self).__init__()
        self.fc1 = nn.Linear(repr_size + action_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, repr_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
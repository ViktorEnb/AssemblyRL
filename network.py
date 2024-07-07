import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.fc_value = nn.Linear(hidden_size, 1)  # Value head

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        
        value = self.fc_value(x)
        return logits, value
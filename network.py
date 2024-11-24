import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Maps state to action probailities
class Policy(nn.Module):
    def __init__(self, repr_size, hidden_dim, num_actions):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(repr_size, hidden_dim)
        self.relu = nn.ReLU()   
        self.hl1 = nn.Linear(hidden_dim, hidden_dim)
        self.hl2 = nn.Linear(hidden_dim, hidden_dim)
        self.hl3 = nn.Linear(hidden_dim, hidden_dim)
        self.hl4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.hl1(x)
        # x = self.relu(x)
        # x = self.hl2(x)
        # x = self.relu(x)
        # x = self.hl3(x)
        # x = self.relu(x)
        # x = self.hl4(x)
        # x = self.relu(x)
        logits = self.fc2(x)
        return logits

class Representation(nn.Module):
    def __init__(self, repr_size, action_size, hidden_dim):
        super(Representation, self).__init__()
        self.fc1 = nn.Linear(repr_size + action_size, hidden_dim)
        self.relu = nn.ReLU()
        self.hl1 = nn.Linear(hidden_dim, hidden_dim)
        self.hl2 = nn.Linear(hidden_dim, hidden_dim)
        self.hl3 = nn.Linear(hidden_dim, hidden_dim)
        self.hl4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, repr_size)
    
    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.hl1(x)
        # x = self.relu(x)
        # x = self.hl2(x)
        # x = self.relu(x)
        # x = self.hl3(x)
        # x = self.relu(x)
        # x = self.hl4(x)
        # x = self.relu(x)
        x = self.fc2(x)
        return x
    
class ValueAndPolicy(nn.Module):
    #Network for representing value + action probabilities
    def __init__(self, repr_size, hidden_dim, action_size):
        super(ValueAndPolicy, self).__init__()
        self.fc1 = nn.Linear(repr_size, hidden_dim)
        self.relu = nn.ReLU()   
        self.fc2 = nn.Linear(hidden_dim, action_size + 1) #policy plus value

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        value = torch.tensor([x[0]])

        #The last `action_size` elements are the policy probabilities
        policy_logits = x[1:]
        policy = F.softmax(policy_logits, dim=-1)  # Normalize to sum to 1
        return torch.cat([value, policy], dim=-1)
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import * 
# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        action_probs = self.softmax(x)
        return action_probs

# Define the combined Transformer model and Policy network
class AlphaDevModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, action_dim):
        super(AlphaDevModel, self).__init__()
        self.transformer = SimpleTransformerModel(input_dim, hidden_dim, num_heads, num_layers)
        self.policy_network = PolicyNetwork(hidden_dim, action_dim, hidden_dim)

    def forward(self, x):
        embeddings = self.transformer(x)
        # Use only the last element in the sequence for action prediction
        action_probs = self.policy_network(embeddings[-1])
        return action_probs

# Dummy data loader for training (replace with real data)
class DummyDataLoader:
    def __iter__(self):
        # Generating a batch of 10 sequences, each of length 6
        # Each sequence has input_dim features and is followed by a dummy reward
        return iter([(torch.randn(6, 1, input_dim), torch.tensor(1.0, requires_grad=False)) for _ in range(10)])

# Define the dimensions
input_dim = len(instruction_set)
hidden_dim = 64
num_heads = 2
num_layers = 2
action_dim = len(instruction_set)  # Assuming the number of possible actions is the same as instructions

# Initialize the model
model = AlphaDevModel(input_dim, hidden_dim, num_heads, num_layers, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training process
def train_model(model, data_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for sequence, reward in data_loader:
            optimizer.zero_grad()
            
            # Ensure sequence requires grad
            sequence = sequence.requires_grad_()

            # Encode the sequence and predict action probabilities
            action_probs = model(sequence)

            # Convert reward to a tensor that supports gradient computation
            # Assume a simple loss function: -reward * log(probability of chosen action)
            reward = torch.tensor(reward, requires_grad=False)

            # Here, we assume you select the action with the highest probability
            action = torch.argmax(action_probs, dim=-1)
            
            # Convert action to a one-hot vector for computing log probability
            action_one_hot = torch.nn.functional.one_hot(action, num_classes=action_dim).float()
            log_prob = torch.sum(action_one_hot * torch.log(action_probs), dim=-1)

            # Loss is negative reward weighted by the log probability
            loss = -reward * log_prob

            # Backpropagate the loss
            loss.backward()
            optimizer.step()
            
            print(f"Action probabilities: {action_probs}")
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Action: {action.item()}")

# Train the model
data_loader = DummyDataLoader()
train_model(model, data_loader, optimizer, num_epochs=100)
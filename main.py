import torch
import torch.nn as nn
import torch.optim as optim
from transformer import * 

from game import Game, ToyGame
from agent import Agent
from mcts import MCTS
from node import Node

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

if __name__ == "__main__":
    # Example usage
    game = ToyGame()
    input_size = 1  # ToyGame has single state dimension
    hidden_size = 32  # Hidden layer size of the neural network
    num_actions = 2  # ToyGame has two possible actions per state

    agent = Agent(game, input_size, hidden_size, num_actions)
    agent.train(num_iterations=100)

    # After training, use the agent to play the game
    state = game.initialize_state()
    while not game.is_terminal(state):
        action = agent.get_action(state)
        print(f"Current state: {state}, Selected action: {action}")
        state = game.apply_action(state, action)

    print(f"Game ended with reward: {game.get_reward(state)}")
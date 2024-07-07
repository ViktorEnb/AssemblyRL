from mcts import MCTS
from game import Game
from torch import nn
from network import Network
from torch import optim
import torch 

class Agent:
    def __init__(self, game, input_size, hidden_size, num_actions):
        self.game = game
        self.mcts = MCTS(game)
        self.policy_value_network = Network(input_size, hidden_size, num_actions)
        self.value_network = None
        self.optimizer = optim.Adam(self.policy_value_network.parameters(), lr=0.001)

    def start_game(self):
        self.mcts.reset()

    def get_action(self, state):
        # Perform MCTS rollouts
        for _ in range(100):  # Perform 100 rollouts per action selection
            self.mcts.rollout(self.policy_value_network)

        # Select action based on visit counts
        best_action = self.mcts.select_best_action()
        return best_action

    def train(self, num_iterations):
        for _ in range(num_iterations):
            self.start_game()
            state = self.game.initialize_state()

            # Perform MCTS rollouts and collect data
            for _ in range(100):  # Perform 100 rollouts per action selection
                self.mcts.rollout(self.policy_value_network)

            # Update policy and value network based on MCTS data
            self.update_network()

    def update_network(self):
        states = []
        actions = []
        rewards = []

        # Collect data from MCTS
        for node in self.mcts.root.children:
            states.append(node.state)
            actions.append(node.state - self.mcts.root.state)
            rewards.append(node.total_reward / node.visit_count)

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Convert rewards to advantages (subtracting mean and normalizing)
        rewards -= rewards.mean()
        rewards /= (rewards.std() + 1e-6)

        # Forward pass
        batch_size = states.size(0)  # Get batch size
        states = states.view(batch_size, -1)  # Reshape to (batch_size, input_size)
        logits, values = self.policy_value_network(states)

        print(logits)
        # Compute policy loss (cross-entropy loss)
        policy_loss = nn.functional.cross_entropy(logits, actions)

        # Compute value loss (mean squared error)
        value_loss = nn.functional.mse_loss(values, rewards)

        # Total loss
        loss = policy_loss + value_loss

        # Zero gradients, backward pass, and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
from game import Game
from node import Node
from torch import nn
import torch
import numpy as np
import random 

class MCTS:
    def __init__(self, game):
        self.game = game
        self.root = Node(state=self.game.initialize_state(), parent=None)

    def rollout(self, policy_value_network):
        node = self.root
        path = [node]

        # 1. Selection
        while node.is_expanded and not self.game.is_terminal(node.state):
            node = self.select(node)
            path.append(node)

        # 2. Expansion
        if not self.game.is_terminal(node.state):
            self.expand(node)
            node = self.select(node)
            path.append(node)

        # 3. Simulation
        reward = self.simulate(node.state, policy_value_network)

        # 4. Backpropagation
        self.backpropagate(path, reward)

    def select(self, node):
        C = 1.41  # Exploration parameter
        best_value = -float('inf')
        best_node = None
        for child in node.children:
            uct_value = (child.total_reward / (child.visit_count + 1e-6)) + C * np.sqrt(np.log(node.visit_count + 1) / (child.visit_count + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node

    def expand(self, node):
        state = node.state
        actions = self.game.get_actions(state)
        for action in actions:
            next_state = self.game.apply_action(state, action)
            child_node = Node(state=next_state, parent=node)
            node.add_child(child_node)
        node.is_expanded = True

    def simulate(self, state, policy_value_network):
        current_state = state

        while not self.game.is_terminal(current_state):
            logits, _ = policy_value_network(torch.tensor([current_state], dtype=torch.float32))
            action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
            action = np.random.choice(self.game.get_actions(current_state), p=action_probs)
            current_state = self.game.apply_action(current_state, action)

        reward = self.game.get_reward(current_state)
        return reward

    def backpropagate(self, path, reward):
        for node in reversed(path):
            node.update(reward)

    def select_best_action(self):
        # Select the action with the highest visit count from the root's children
        best_action_node = max(self.root.children, key=lambda c: c.visit_count)
        return best_action_node.state - self.root.state

    def reset(self):
        self.root = Node(state=self.game.initialize_state(), parent=None)
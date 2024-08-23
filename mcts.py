from game import Game
from node import Node
from torch import nn
import torch
import numpy as np
import random 
import time
class MCTS:
    def __init__(self, game):
        self.game = game
        self.root = Node(state=self.game.initialize_state(), parent=None)

    def rollout(self, policy_network, value_network, node, _lambda = 0):
        # 1. Selection with UCB
        while node.is_expanded and not self.game.is_terminal(node):
            #Wait for node to expand in another thread
            while node.is_expanding:
                continue
            node = self.select(node)
        
        # 2. Use value network for more accurate reward estimate        
        reward = _lambda * value_network(node.state)
        if _lambda == 1:
            self.expand(node)
            self.backpropagate(node, reward)
            return 
        
        # 3. Simulating a reward
        while not self.game.is_terminal(node):
            self.expand(node)
            logits = policy_network(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1)
            #Remove illegal moves
            action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
            action_probs = 1.0 / sum(action_probs) * action_probs
            action = np.random.choice(self.game.get_actions(), p=action_probs)
            for child in node.children:
                if child.action == action:
                    node = child
        
        reward += (1 - _lambda) * self.game.get_reward(node)
        # 4. Backpropagation
        self.backpropagate(node, reward)

        return node, reward.item()

    def select(self, node):
        C = 1.41  # Exploration parameter
        best_value = -float('inf')
        best_nodes = []
        uct_values = []
        for child in node.children:
            if self.game.get_legal_moves(node)[child.action] == 0:
                #Never select illegal move
                continue

            uct_value = (child.total_reward / (child.visit_count + 1e-6)) + C * np.sqrt(np.log(node.visit_count + 1) / (child.visit_count + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            #To not have a bias towards lower indicies
            elif uct_value == best_value:
                best_nodes.append(child)
            uct_values.append(uct_value)
        return random.choice(best_nodes)

    def expand(self, node):
        if node.is_expanded or node.is_expanding:
            return
        node.is_expanding = True
        actions = self.game.get_actions()
        for action in actions:
            next_state = self.game.apply_action(node.state, action.item())
            child_node = Node(state=next_state, parent=node, action = action.item())
            node.add_child(child_node)
        node.is_expanding = False
        node.is_expanded = True



    def backpropagate(self, node, reward):
        current_node = node

        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent

    def softmax(self, x):
        """Compute softmax values for each set of scores in x."""
        temperature=1
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()
    
    # Select the action with the highest visit count from the root's children
    def select_best_action(self, node):
        # Extract visit counts from children
        visit_counts = np.array([child.visit_count for child in node.children])
        
        # Calculate softmax probabilities
        probabilities = self.softmax(visit_counts)
        probabilities = np.multiply(probabilities, self.game.get_legal_moves(node).detach().numpy())
        probabilities = 1.0 / sum(probabilities) * probabilities


        # Select a child based on the softmax probabilities
        selected_index = np.random.choice(len(node.children), p=probabilities)
        return node.children[selected_index]


    def reset(self):
        self.root = Node(state=self.game.initialize_state(), parent=None)
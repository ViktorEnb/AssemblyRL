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

    def rollout(self, policy_network, value_network, node):
        #Todo: come up with a cleaner way without this extra variable
        final_state = None
        # 1. Selection with UCB
        while node.is_expanded and not self.game.is_terminal(node):
            node = self.select(node)
        # 2. Expansion with policy network
        if not self.game.is_terminal(node):
            self.expand(node)
            logits = policy_network(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
            action = np.random.choice(self.game.get_actions(node), p=action_probs)
            final_state = self.game.apply_action(node, action)
        else:
            final_state = node.state
        # 3. Simulation approximated by value  network
        reward = value_network(final_state)
        # 4. Backpropagation
        self.backpropagate(node, reward)

    def select(self, node):
        C = 1.41  # Exploration parameter
        best_value = -float('inf')
        best_nodes = []
        uct_values = []
        for child in node.children:
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
        state = node.state
        actions = self.game.get_actions(node)
        for action in actions:
            next_state = self.game.apply_action(node, action)
            child_node = Node(state=next_state, parent=node, action = action)
            node.add_child(child_node)
        node.is_expanded = True

    # def simulate(self, node, policy_network):
    #     current_node = node
    #     terminal = False
    #     while not self.game.is_terminal(current_node):
    #         terminal = True
    #         # logits = policy_network(current_node.state)
    #         # action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
    #         action = np.random.choice(self.game.get_actions(current_node))
    #         current_state = self.game.apply_action(current_node.state, action)
    #         current_node = Node(current_state, current_node, action)

    #     actions = self.game.collect_game(current_node)
    #     reward = self.game.get_reward(actions)
    #     self.replay.append({'game': actions, 'reward': reward})
    #     return reward

    def backpropagate(self, node, reward):
        current_node = node

        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent
    
    # Select the action with the highest visit count from the root's children
    def select_best_action(self, node):
        #Shuffle in order to not favour small indicies in case of a tie    
        random.shuffle(node.children)
        # print("SELECTING THE HIGHEST OF THE FOLLOWING")
        # for child in node.children:
        #     print(child.visit_count, "   ", child.action,    "    ", self.game.assembly.instruction_decode(child.action))
        best_action_node = max(node.children, key=lambda c: c.visit_count)
        return best_action_node

    def reset(self):
        self.root = Node(state=self.game.initialize_state(), parent=None)
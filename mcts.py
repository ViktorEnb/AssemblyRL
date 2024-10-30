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
        while (node.is_expanded or node.is_expanding) and not self.game.is_terminal(node):
            #Wait for node to expand in another thread
            while node.is_expanding:
                continue
            node = self.select(node)

        # 2. Use value network for more accurate reward estimate  
        reward = torch.tensor([0.0])
        if _lambda > 0:    
            self.expand(node)     
            reward += _lambda * value_network(node.state)
            if _lambda == 1:
                self.expand(node)
                self.backpropagate(node, reward)
                return 
        
        # 3. Simulating a reward
        while not self.game.is_terminal(node):
            self.expand(node)
            if node.is_expanding:
                continue
            logits = policy_network(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1).to("cpu")
            #Remove illegal moves
            action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
            action_probs = 1.0 / sum(action_probs) * action_probs
            action = np.random.choice(self.game.get_actions())
            if action not in node.children:
                child_node = Node(state=None, parent=node, action = action)
                node.add_child(action, child_node)
            node = node.children[action]
        reward += (1 - _lambda) * self.game.get_reward(node)

        # 4. Backpropagation
        self.backpropagate(node, reward.item())

        return node, reward.item()

    def select(self, node):
        C = 3  #exploration parameter
        best_value = -float('inf')
        best_nodes = []
        uct_values = []
        for action in self.game.get_actions():
            if self.game.get_legal_moves(node)[action] == 0:
                #skip illegal moves
                continue

            #check if the action already has an associated child node
            if action.item() not in node.children:
                child = Node(state=None, parent=node, action=action.item())
                node.add_child(action.item(), child)
            else:
                child = node.children[action.item()]

            total_reward = child.total_reward / 200 #Since our max reward is 200 we have to scale it to make it fair for exploration value
            visit_count = child.visit_count

            uct_value = (total_reward / (visit_count + 1e-6)) + C * np.sqrt(np.log(node.visit_count + 1) / (visit_count + 1e-6))

            # if node == self.root:
            #     print(f"Action: {action}")
            #     print(f"  - Total Reward: {total_reward}")
            #     print(f"  - Visit Count: {visit_count}")
            #     print(f"  - Node Visit Count (parent): {node.visit_count}")
            #     print(f"  - UCT Value: {uct_value}")
            #     print(f"  - Exploration value: { C * np.sqrt(np.log(node.visit_count + 1) / (visit_count + 1e-6))}")


            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)

            uct_values.append(uct_value)
        selected_node = random.choice(best_nodes)
        # if node==self.root:
        #     print("SELECTED NODE: " + str(selected_node.action))
        #     print("visit count: " + str(selected_node.visit_count))
        return selected_node 
    

    def expand(self, node):
        if node.is_expanded or node.is_expanding:
            return
        node.is_expanding = True
        #Only give a node a state if it's expanded. This is a way to optimize
        #in case of many children per node
        if node.state == None:
            node.state = self.game.apply_action(node.parent.state, node.action)
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
        most_visited_nodes = [None]
        most_visits = -float('inf')
        print(len(node.children.keys()))
        for action in node.children.keys():
            print("Visit Count: " + str(node.children[action].visit_count))
            if(node.children[action].visit_count > most_visits):
                most_visited_nodes = [node.children[action]]
                most_visits = node.children[action].visit_count
            elif (node.children[action].visit_count == most_visits):
                most_visited_nodes.append(node.children[action])
        print("Most visits: " + str(most_visits))
        # Calculate softmax probabilities
        # probabilities = self.softmax(visit_counts)
        # probabilities = np.multiply(probabilities, self.game.get_legal_moves(node).detach().numpy())
        # probabilities = 1.0 / sum(probabilities) * probabilities
        # Select a child based on the softmax probabilities
        # selected_index = np.random.choice(len(node.children), p=probabilities)
        
        return random.choice(most_visited_nodes)


    def reset(self):
        self.root = Node(state=self.game.initialize_state(), parent=None)
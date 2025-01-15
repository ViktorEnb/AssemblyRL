from game import Game
from node import Node
from torch import nn
import torch
import numpy as np
import random 
import time
import yaml
from mctsplotter import MCTSPlotter

with open('config.yaml', 'r') as f:
    params = yaml.safe_load(f)

class MCTS:
    def __init__(self, game):
        self.game = game
        self.root = Node(state=self.game.initialize_state(), parent=None)
        self.plotter = MCTSPlotter(self.root, game)
        self.plotter.root = self.root
        self.network_values = {}

        self.iterations = 0
        self.exp_factor = params["mcts_exploration"]  #exploration parameter
        if self.iterations >= params["mcts_network_start"]:
            self.net_factor = params["mcts_network"] #policy network bias parameter
        else:
            self.net_factor = 0
    def rollout(self, node, policy_network = None, policy_and_value = None):
        #Performs a rollout either with a policy netowrk or with a policy combined with value network
        #Exactly one of the policy_network, policy_and_value network has to not be None

        # 1. Selection with UCB
        while node.is_expanded and not self.game.is_terminal(node):
            network = policy_network if policy_network != None else policy_and_value
            node = self.select(node, network)
        reward = 0

        # 2. Simulating a reward
        if policy_and_value == None:
            while not self.game.is_terminal(node):
                self.expand(node)
                logits = policy_network(node.state)
                action_probs = nn.functional.softmax(logits, dim=-1).to("cpu")
                #Remove illegal moves
                action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
                action_probs = 1.0 / sum(action_probs) * action_probs
                action = np.random.choice(self.game.get_actions(), p=action_probs)    

                if action not in node.children:
                    child_node = Node(state=None, parent=node, action = action)
                    node.add_child(action, child_node)
                    #We may need the network term for plotting
                    network_term = self.net_factor * action_probs[action]
                    self.network_values[child_node.unique_id()] = network_term
                    self.plotter.add_network_value(child_node.unique_id(), network_term)

                node = node.children[action]
            reward = self.game.get_reward(node)

        elif policy_network == None:
            self.expand(node)
            reward = policy_and_value(node.state)[-1].item() #Estimate reward with network

        # 4. Backpropagation
        self.backpropagate(node, reward)

        #Transfer changes to plotter
        self.plotter.root = self.root

        return node, reward

    def select(self, node, policy_network):
        best_value = -float('inf')
        best_nodes = []
        uct_values = []
        legal_moves = [torch.tensor(action) for action, value in enumerate(self.game.get_legal_moves(node)) if value.item() == 1]

        logits = policy_network(node.state)
        action_probs = nn.functional.softmax(logits, dim=-1).to("cpu")
        #Remove illegal moves
        action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
        action_probs = 1.0 / sum(action_probs) * action_probs

        for action in legal_moves:
            #check if the action already has an associated child node
            if action.item() not in node.children:
                child = Node(state=None, parent=node, action=action.item())
                node.add_child(action.item(), child)
            else:
                child = node.children[action.item()]

            total_reward = child.total_reward / 200 #Since our max reward is 200 we have to scale it to make it fair for exploration value
            visit_count = child.visit_count
            P_s_a = self.net_factor * action_probs[action]

            reward_term = (total_reward / (visit_count + 1e-6)) 
            exploration_term = self.exp_factor * np.sqrt(np.log(node.visit_count + 1) / (visit_count + 1e-6))
            id = child.unique_id()
            if id not in self.network_values:
                network_term = self.net_factor * P_s_a.item()
                self.network_values[id] = network_term
                self.plotter.add_network_value(id, network_term)
            uct_value = reward_term + exploration_term + self.network_values[id]

            # if node == self.root:
            #     print(f"Action: {action.item()}: {self.game.assembly.decode(action)}")
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
        #Only give a node a state if it's expanded. This is a way to optimize
        #in case of many children per node
        if node.state == None:
            node.state = self.game.apply_action(node.parent.state, node.action)
        node.is_expanded = True



    def backpropagate(self, node, reward):
        current_node = node
        while current_node is not None:
            current_node.update(reward)
            current_node = current_node.parent

    def softmax(self, x):
        #Compute softmax values for each set of scores in x
        temperature=1
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()
    
    # Select the action with the highest visit count from the root's children
    def select_best_action(self, node):
        # Extract visit counts from children
        most_visited_nodes = [None]
        most_visits = -float('inf')
        for action in node.children.keys():
            if(node.children[action].visit_count > most_visits):
                most_visited_nodes = [node.children[action]]
                most_visits = node.children[action].visit_count
            elif (node.children[action].visit_count == most_visits):
                most_visited_nodes.append(node.children[action])

        selected_node = random.choice(most_visited_nodes)
        return selected_node


    def reset(self):
        self.root = Node(state=self.game.initialize_state(), parent=None)
        self.plotter.reset(self.root)
        self.network_values = {} #have to reset network_values as network will be trained inbetween iterations
        self.iterations += 1
        if self.iterations >= params["mcts_network_start"]:
            self.net_factor = params["mcts_network"]





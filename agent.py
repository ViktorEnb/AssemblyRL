from mcts import MCTS
from game import Game
from torch import nn
from network import Policy, Value
from torch import optim
import torch 
from assembly import AssemblyGame
from datetime import datetime
import os
import random

class Agent:
    def __init__(self, game, repr_size, action_dim, game_with_repr_network=False):
        self.game = game
        self.mcts = MCTS(game)
        self.repr_size = repr_size
        hidden_size = 20
        self.action_dim = action_dim
        self.policy_network = Policy(repr_size, hidden_size, action_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        self.value_network = Value(repr_size, hidden_size)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)
        self.replay = [] #List of played games and the rewards achieved
        self.batch_size = 25
        self.previous_best_reward = -float('inf')
        #Boolean value indicating whether the game uses a neural network to represent states
        self.has_repr_network = game_with_repr_network


    def get_action(self, node):
        # Perform MCTS rollouts
        for _ in range(50):  # Perform 100 rollouts per action selection
            self.mcts.rollout(self.policy_network, node)

        # Select action based on visit counts
        best_node = self.mcts.select_best_action(node)
        return best_node
    def print_values(self):
        for i in range(7):
            t = torch.tensor([i], dtype=torch.float32)
            print("self.value_network(" + str(i) +") = ", self.value_network(t))


    def train(self, num_iterations):
        for i in range(num_iterations):
            # print(i, "th iteration")
            self.mcts.reset()
            node = self.mcts.root
            while not self.game.is_terminal(node):
                for _ in range(25): 
                    self.mcts.rollout(self.policy_network, self.value_network, node)
                node = self.mcts.select_best_action(node)
                # print("Selected action: ", node.action)
            reward = self.game.get_reward(node)
            self.save_experience(node, reward)
            # self.save_best_game(i)
            self.update_networks()
            
    def save_experience(self, node, reward):
        game = []
        while node.parent != None:
            game.append({"action": node.action, "state": node.state})
            node = node.parent
        game.reverse()
        self.replay.append({"game": game, "reward": reward})


    def save_best_game(self,iteration):        
        best_game = max(self.mcts.replay, key=lambda x: x['reward'])
        actions = best_game["game"]
        reward = round(best_game["reward"],2)
        if reward <= self.previous_best_reward:
            return

        self.previous_best_reward = reward
        filename = str(reward) + ".c"
        self.game.write_game(actions, filename=filename, meta_info = ["Reward: " + str(reward), "Iteration: " + str(iteration), "Time since start: " + str((datetime.now() - self.game.time_started).seconds)])

    def update_networks(self):
        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)

        policy_loss = 0
        policy_loss_fn = torch.nn.CrossEntropyLoss()
        self.policy_optimizer.zero_grad()
        #TRAINING POLICY NETWORK
        current_state = self.game.initialize_state()
        for replay in batch:
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                action = move["action"]
                state = move["state"]
                predicted_actions = self.policy_network(current_state)
                current_state = state
                # action_onehot = torch.zeros(self.action_dim)
                # action_onehot[action] = 1
                policy_loss += policy_loss_fn(predicted_actions, torch.tensor(action))
        policy_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.zero_grad()
        value_loss = 0
        value_loss_fn = torch.nn.MSELoss()
        #TRAINING VALUE NETWORK
        current_state = self.game.initialize_state()
        for replay in batch:
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                state = move["state"]
                predicted_reward = self.value_network(current_state)
                value_loss += value_loss_fn(predicted_reward, torch.tensor(reward, dtype=torch.float32))
                current_state = state
        value_loss.backward()
        self.value_optimizer.step()
                


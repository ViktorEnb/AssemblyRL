from mcts import MCTS
from game import Game
from torch import nn
from network import Policy, Value
from node import Node
from torch import optim
import torch 
from assembly import AssemblyGame
from datetime import datetime
import os
import random
import numpy as np

class Agent:
    def __init__(self, game, repr_size, action_dim):
        self.game = game
        self.mcts = MCTS(game)
        self.repr_size = repr_size
        hidden_size = 20
        self.action_dim = action_dim
        self.policy_network = Policy(repr_size, hidden_size, action_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.value_network = Value(repr_size, hidden_size)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.01)
        self.replay = [] #List of played games and the rewards achieved
        self.batch_size = 1
        self.previous_best_reward = -float('inf')
        self.update_policy = False


    def get_action(self, node):
        # Perform MCTS rollouts
        for _ in range(20):  # Perform 100 rollouts per action selection
            self.mcts.rollout(self.policy_network, node)

        # Select action based on visit counts
        best_node = self.mcts.select_best_action(node)
        return best_node


    def train(self, num_iterations):
        for i in range(num_iterations):
            #It doesn't make sense to update the policy with a random value network
            if i >= num_iterations // 5:
                self.update_policy = True
            # print(i, "th iteration")
            self.mcts.reset()
            node = self.mcts.root
            while not self.game.is_terminal(node):
                for _ in range(50): 
                    self.mcts.rollout(self.policy_network, self.value_network, node)
                node = self.mcts.select_best_action(node)
                # print("Selected action: ", node.action)
            reward = self.game.get_reward(node)
            self.save_experience(node, reward)
            # self.save_best_game(i)
            self.update_networks()
        # self.train_on_entire_game()
            
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
        self.game.repr_network.zero_grad()
        #TRAINING POLICY NETWORK
        for replay in batch:
            current_state = self.game.initialize_state()
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                action = move["action"]
                current_state = self.game.apply_action(current_state, action)
                predicted_actions = self.policy_network(current_state)
                policy_loss += policy_loss_fn(predicted_actions, torch.tensor(action))   
        policy_loss.backward()

        if self.update_policy:
            self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss = 0
        value_loss_fn = torch.nn.MSELoss()
        #TRAINING VALUE NETWORK
        for replay in batch:
            current_state = self.game.initialize_state()
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                predicted_reward = self.value_network(current_state)                
                value_loss += value_loss_fn(predicted_reward, torch.tensor([reward], dtype=torch.float32))
                action = move["action"]
                current_state = self.game.apply_action(current_state, action)
        value_loss.backward()
        self.value_optimizer.step()
        if self.update_policy:
            self.game.repr_optimizer.step()

    
    def print_network_predictions(self):
        unique_states = self.game.get_unique_states()
        for game in self.game.random_games:
            repr = self.game.initialize_state()
            logits = self.policy_network(repr)
            action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
            predicted_value = self.value_network(repr)
            print("State: ", 0, " Action probs: ", action_probs)
            print("State: ", 0, " repr: ", repr)
            print("State: ", 0, " predicted value: ", predicted_value[0].item())
            for (state, action) in game:
                action_onehot = torch.zeros(self.action_dim)
                action_onehot[action] = 1
                repr = self.game.repr_network(torch.concat((repr, action_onehot)))
                logits = self.policy_network(repr)
                action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
                predicted_value = self.value_network(repr)

                print("State: ", state, " Action probs: ", action_probs)
                print("State: ", state, " repr: ", repr)
                print("State: ", state, " predicted value: ", predicted_value[0].item())

    #Used for testing training of value and policy network. Is called after gathering of experiences
    def train_on_entire_game(self):
        #Print before training values
        self.print_network_predictions()

        #Sample 100 batches
        for i in range(1000):
            self.update_networks()
        print("\n\n\n")
        #Print after training values
        self.print_network_predictions()

    def play_game(self):
        nodes = []
        node = Node(self.game.initialize_state(), None)
        nodes.append(node)
        while not self.game.is_terminal(node):
            logits = self.policy_network(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
            selected_action = np.random.choice(self.game.get_actions(node), p=action_probs)
            print("Selecting action: ", selected_action)
            node = Node(self.game.apply_action(node.state, selected_action), node, action=selected_action)
            nodes.append(node)
        print("Got reward: ", self.game.get_reward(node))



                


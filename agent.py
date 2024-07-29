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
    def __init__(self, game, repr_size, action_dim):
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
        self.batch_size = 1
        self.previous_best_reward = -float('inf')


    def get_action(self, node):
        # Perform MCTS rollouts
        for _ in range(50):  # Perform 100 rollouts per action selection
            self.mcts.rollout(self.policy_network, node)

        # Select action based on visit counts
        best_node = self.mcts.select_best_action(node)
        return best_node


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
        if self.game.has_repr_network:
            self.game.repr_network.zero_grad()
        #TRAINING POLICY NETWORK
        for replay in batch:
            current_state = self.game.initialize_state()
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                action = move["action"]
                if self.game.has_repr_network:
                    current_state = self.game.apply_action(current_state, action)
                else:
                    state = move["state"]
                    current_state = state
                predicted_actions = self.policy_network(current_state)
                policy_loss += policy_loss_fn(predicted_actions, torch.tensor(action))   
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.game.has_repr_network:
            self.game.repr_network.step()

        self.value_optimizer.zero_grad()
        value_loss = 0
        value_loss_fn = torch.nn.MSELoss()
        #TRAINING VALUE NETWORK
        for replay in batch:
            current_state = self.game.initialize_state()
            game = replay["game"]
            reward = replay["reward"]
            for move in game:
                if self.game.has_repr_network:
                    action = move["action"]
                    current_state = self.game.apply_action(current_state, action)
                else:
                    state = move["state"]
                    current_state = state
                predicted_reward = self.value_network(current_state)
                value_loss += value_loss_fn(predicted_reward, torch.tensor([reward], dtype=torch.float32))
                current_state = state
        value_loss.backward()
        self.value_optimizer.step()

    def get_unique_states(self):
        unique_states = []
        unique_states.append(self.game.initialize_state())
        for replay in self.replay:
            for move in replay["game"]:
                if move["state"] not in unique_states:
                    unique_states.append(move["state"])
        return unique_states
    
    def print_network_predictions(self):
        unique_states = self.get_unique_states()
        for state in unique_states:
            print("State: ", state, "   Value(state): ", self.value_network(state))
            logits = self.policy_network(state)
            action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
            print("State: ", state, " Action probs: ", action_probs)
            if self.game.has_repr_network:
                random_action = self.game.get_actions(None)[0]
                action_onehot = torch.zeros(2)
                action_onehot[random_action] = 1
                print("State: ", state, " self.game.repr_network(state, self.game.get_actions(state)[0]): ", self.game.repr_network(torch.concat((state, action_onehot))))
        
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



                


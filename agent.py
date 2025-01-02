from mcts import MCTS
from game import Game
from torch import nn
from network import Policy, ValueAndPolicy
from node import Node
from torch import optim
import torch 
from assembly import AssemblyGame
from datetime import datetime
import os
import random
import numpy as np
import time 
import concurrent.futures
import yaml 

with open('config.yaml', 'r') as f:
    params = yaml.safe_load(f)

class Agent:
    def __init__(self, game, repr_size, hidden_size, action_size, load=False, save=False):
        self.game = game
        self.mcts = MCTS(game)
        self.repr_size = repr_size
        self.action_size = action_size
        # self.device = ("cuda" if torch.cuda.is_available() else "cpu") 
        self.device = "cpu"

        self.policy_network = Policy(repr_size, hidden_size, action_size).to(self.device) if params["pol_net"] else None
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=params["pol_net_lr"]) if params["pol_net"] else None
        self.policy_value = ValueAndPolicy(repr_size, hidden_size, action_size).to(self.device) if params["pol_val"] else None
        self.policy_value_optimizer = optim.Adam(self.policy_value.parameters(), lr=params["pol_val_lr"]) if params["pol_val"] else None
        if load:
            self.load_models(os.path.join(".", "saved_models", self.game.algo_name))

        self.save = save
        self.replay = [] #List of played games and the rewards achieved
        self.batch_size = params["batch_size"]
        self.highest_reward = -float('inf')

        self.training_time = 0 #Time spent training networks
        self.action_dict = {}

    def train(self, num_iterations):
        current_best_game = None
        self.time_started = datetime.now()
        for i in range(num_iterations):
            print(i, "th iteration")
            self.mcts.reset()
            batch = []
            node = self.mcts.root

            while not self.game.is_terminal(node):
                for j in range(params["num_iterations"]):
                    # (One of policy_network, policy_and_value will be none)
                    end_node, reward = self.mcts.rollout(node, policy_network=self.policy_network, policy_and_value=self.policy_value)                    
                    game = {"game": end_node.get_actions(), "reward": reward}
                    if self.policy_network != None:
                        batch.append(game) #Append all games and sort out games later
                    else:
                        if self.game.is_terminal(end_node):
                            #We have to simulate the reward in this case as the reward we have was
                            #simply an estimation by the value network
                            actions = [v["action"] for v in game["game"]]
                            reward = self.game.get_reward(actions)
                            game = {"game": game["game"], "reward": reward}
                            batch.append(game)  

                    if reward > self.highest_reward:
                        print("Got new best reward: ", reward)
                    #Check for terminal game to prevent weird situations while using pol-val
                    if reward >= self.highest_reward and self.game.is_terminal(end_node):
                        current_best_game = game
                        self.highest_reward = reward
                node = self.mcts.select_best_action(node)

            
            reward = current_best_game["reward"]
            print("Got reward ", reward)
            if self.policy_network != None:
                self.sort_and_train(batch)
            elif self.policy_value != None:
                batches = [batch[i:i + self.batch_size] for i in range(0, len(batch), self.batch_size)]
                for m_batch in batches:
                    self.update_networks(m_batch)
            
            #Saves the weights of the networks to file         
            if self.save:
                self.save_models(os.path.join(".", "saved_models", self.game.algo_name))
            
            #Saves the best assembly game to file
            if isinstance(self.game,AssemblyGame):
                self.save_game(current_best_game, i)      
            

    def sort_and_train(self, games):
        #Sorts out the best games and trains the policy network on them
        #This approach does not work for training the value network as it will tend to a constant function in case all of the best games have high reward
        batch_sorted = sorted(games, key=lambda x: x['reward'], reverse=True)
        num_games_to_keep = int(len(batch_sorted) * params["best_games_perc"])
        top_batch = batch_sorted[:num_games_to_keep]
        print("Length of training: " + str((len(top_batch))))
        start_time = time.time()
        batches = [top_batch[i:i + self.batch_size] for i in range(0, len(top_batch), self.batch_size)]
        batch_counter = 0
        for m_batch in batches:
            print("Training on batch nr: ", str(batch_counter),"/", str(len(batches)))
            self.update_networks(m_batch)
            batch_counter += 1
        print("Training took " + str(time.time() - start_time))


    def update_networks(self, batch):
        start_training = time.time()

        if self.policy_network != None:
            #TRAINING POLICY NETWORK
            policy_loss = 0
            policy_loss_fn = torch.nn.CrossEntropyLoss()
            self.policy_optimizer.zero_grad()
            self.game.repr_network.zero_grad()
            for replay in batch:
                current_state = self.game.initialize_state()
                game = replay["game"]
                reward = replay["reward"]
                for move in game:
                    action = move["action"]
                    predicted_actions = self.policy_network(current_state)
                    current_state = self.game.apply_action(current_state, action)
                    if action in self.action_dict:
                        self.action_dict[action] += 1
                    else:
                        self.action_dict[action] = 1
                    policy_loss += policy_loss_fn(predicted_actions, torch.tensor(action).to(self.device))   
            policy_loss.backward()
            self.policy_optimizer.step()

        
        if self.policy_value != None:
            #TRAINING POLICY + VALUE-NETWORK
            policy_value_loss = 0
            policy_loss_fn = torch.nn.CrossEntropyLoss()
            value_loss_fn = torch.nn.MSELoss()
            self.policy_value_optimizer.zero_grad()
            self.game.repr_network.zero_grad()
            for replay in batch:
                current_state = self.game.initialize_state()
                game = replay["game"]
                reward = replay["reward"]
                for move in game:
                    action = move["action"]
                    predicted_reward = self.policy_value(current_state)[-1]
                    predicted_actions = self.policy_value(current_state)[:-1]
                    current_state = self.game.apply_action(current_state, action)
                    if action in self.action_dict:
                        self.action_dict[action] += 1
                    else:
                        self.action_dict[action] = 1
                    value_loss = value_loss_fn(torch.tensor(reward), predicted_reward)
                    policy_loss = policy_loss_fn(predicted_actions, torch.tensor(action).to(self.device))
                    policy_value_loss += value_loss + policy_loss   
            policy_value_loss.backward()
            self.policy_value_optimizer.step()

        self.game.repr_optimizer.step()
        self.training_time += time.time() - start_training

    #Used for testing training of value and policy network. Is called after gathering of experiences
    def train_on_entire_game(self):
        #Print before training values
        self.print_network_predictions()

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
            #With the optimizations made every node is not guranteed to have a state
            if node.state == None:
                node.state = self.game.apply_action(node.parent.state, node.action)
            network = self.policy_network if self.policy_network != None else self.policy_value
            logits = network(node.state)[:self.action_size]
            print(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1)
            #Remove illegal moves
            action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
            action_probs = 1.0 / sum(action_probs) * action_probs
            print(action_probs)
            index = np.argmax(action_probs)
            selected_action = self.game.get_actions()[index]
            # selected_action = np.random.choice(self.game.get_actions(), p=action_probs)
            # print("Selecting action: ", selected_action, "which is ", self.game.assembly.decode(selected_action.item()))
            node = Node(self.game.apply_action(node.state, selected_action), node, action=selected_action)
            nodes.append(node)
        print("Got reward: ", self.game.get_reward(node))

    def save_game(self, game, iteration):        
        actions = []
        for d in game["game"]:
            actions.append(d["action"])

        filename = os.path.join(".", "best_algos", self.time_started.strftime("%m-%d-%H-%M") + self.game.algo_name, "n=" + str(iteration) + "r=" + str(self.highest_reward) + ".c")        
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        total_time = (datetime.now() - self.time_started).seconds
        perc_training = 0 if total_time == 0 else self.training_time * 100.0 / total_time 
        decoded_actions = []
        for action in actions:
            decoded_actions.append(self.game.assembly.decode(action))
        self.game.write_game(decoded_actions, filename=filename, meta_info = ["Reward: " + str(self.highest_reward), "Iteration: " + str(iteration), "Time since start: " + str(total_time) + " seconds", "Percentage of time updating networks: " + str(perc_training) + "%"])

    def save_models(self, save_path):
        if self.policy_network != None:
            torch.save({
                'policy_network_state_dict': self.policy_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            }, save_path)
        if self.policy_value != None:
            torch.save({
                'value_network_state_dict': self.policy_value.state_dict(),
                'value_optimizer_state_dict': self.policy_value_optimizer.state_dict(),
            }, save_path)
        print(f"Models saved to {save_path}")

    def load_models(self, load_path):
        checkpoint = torch.load(load_path)
        if self.policy_network != None:
            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        if self.policy_value != None:
            self.policy_value.load_state_dict(checkpoint['value_network_state_dict'])
            self.policy_value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Models loaded from {load_path}")


                


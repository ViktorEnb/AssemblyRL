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
import time 
import concurrent.futures

class Agent:
    def __init__(self, game, repr_size, action_dim, load=False, save=False):
        self.game = game
        self.mcts = MCTS(game)
        self.repr_size = repr_size
        hidden_size = 32
        self.action_dim = action_dim
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") #Not used at the moment as my cpu is faster than my gpu¨
        self.policy_network = Policy(repr_size, hidden_size, action_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.value_network = Value(repr_size, hidden_size)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.01)        

        if load:
            self.load_models(os.path.join(".", "saved_models", self.game.algo_name))

        self.save = save
        self.replay = [] #List of played games and the rewards achieved
        self.batch_size = 1
        self.highest_reward = -float('inf')
        self.update_policy = False

        self.training_time = 0 #Time spent training networks
        self.max_threads = 2


    def get_action(self, node):
        # Perform MCTS rollouts
        for _ in range(20):  # Perform 100 rollouts per action selection
            self.mcts.rollout(self.policy_network, node)

        # Select action based on visit counts
        best_node = self.mcts.select_best_action(node)
        return best_node


    def train(self, num_iterations):
        current_best_game = None
        thread_counter = 0
        for i in range(num_iterations):
            #It doesn't make sense to update the policy with a random value network
            if i >= 0:
                self.update_policy = True
            print(i, "th iteration")
            self.mcts.reset()
            batch = []
            node = self.mcts.root
            while not self.game.is_terminal(node):
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                    # Submit tasks to the executor using a lambda function
                    futures = [
                        executor.submit(
                            lambda: self.mcts.rollout(self.policy_network, self.value_network, node)
                        )
                        for _ in range(100)
                    ]
                    
                    # As each thread completes, process the results
                    for future in concurrent.futures.as_completed(futures):
                        end_node, reward = future.result()
                        game = {"game": end_node.get_actions(), "reward": reward}
                        batch.append(game)
                        
                        if reward > self.highest_reward:
                            current_best_game = game
                            self.highest_reward = reward
                node = self.mcts.select_best_action(node)
            reward = current_best_game["reward"]
            print("Got reward ", reward)
            self.update_networks(batch)

            if self.save:
                self.save_models(os.path.join(".", "saved_models", self.game.algo_name))
            self.save_game(current_best_game, i)

    def save_game(self, game, iteration):        
        actions = []
        for d in game["game"]:
            actions.append(d["action"])

        filename = os.path.join(".", "best_algos", self.game.time_started.strftime("%m-%d-%H-%M") + self.game.algo_name, str(self.highest_reward) + ".c")        
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        total_time = (datetime.now() - self.game.time_started).seconds
        perc_training = self.training_time * 100.0 / total_time
        self.game.write_game(actions, filename=filename, meta_info = ["Reward: " + str(self.highest_reward), "Iteration: " + str(iteration), "Time since start: " + str(total_time) + " seconds", "Percentage of time updating networks: " + str(perc_training) + "%"])

    def update_networks(self, batch):
        # if len(self.replay) < self.batch_size:
        #     return
        # batch = random.sample(self.replay, self.batch_size)

        start_training = time.time()

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
                predicted_actions = self.policy_network(current_state)
                current_state = self.game.apply_action(current_state, action)
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

        self.training_time += time.time() - start_training

    def print_network_predictions(self):
        #This only works at the moment if all moves in all states are valid
        unique_states = self.game.get_unique_states()
        for game in self.game.random_games:
            repr = self.game.initialize_state()
            logits = self.policy_network(repr)
            # logits = torch.mul(logits, self.game.get_legal_moves(node))
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
                # logits = torch.mul(logits, self.game.get_legal_moves(node))

                action_probs = nn.functional.softmax(logits, dim=-1).detach().numpy()
                predicted_value = self.value_network(repr)

                print("State: ", state, " Action probs: ", action_probs)
                print("State: ", state, " repr: ", repr)
                print("State: ", state, " predicted value: ", predicted_value[0].item())

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
            logits = self.policy_network(node.state)
            action_probs = nn.functional.softmax(logits, dim=-1)
            #Remove illegal moves
            action_probs = torch.mul(action_probs, self.game.get_legal_moves(node)).detach().numpy()
            action_probs = 1.0 / np.linalg.norm(action_probs) * action_probs

            index = np.argmax(action_probs)
            selected_action = self.game.get_actions()[index]
            # selected_action = np.random.choice(self.game.get_actions(), p=action_probs)
            print("Selecting action: ", selected_action)
            node = Node(self.game.apply_action(node.state, selected_action), node, action=selected_action)
            nodes.append(node)
        print("Got reward: ", self.game.get_reward(node))


    def save_models(self, save_path):
        """Save the policy and value networks along with their optimizers."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, save_path)
        print(f"Models saved to {save_path}")

    def load_models(self, load_path):
        """Load the policy and value networks along with their optimizers."""
        checkpoint = torch.load(load_path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Models loaded from {load_path}")


                


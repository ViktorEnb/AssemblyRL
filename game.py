import torch
from network import Representation
from torch import optim
import random 
class Game:
    def __init__(self, actions):
        self.actions = actions
        self.current_state = self.initialize_state()

    def initialize_state(self):
        raise NotImplementedError

    def get_reward(self, state):
        raise NotImplementedError

    def get_actions(self, state):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    def apply_action(self, state, action):
        raise NotImplementedError


#Class used for testing mcts algorithm
class ToyGame(Game):
    def __init__(self):
        self.tree = {
            0: [1, 2],      
            1: [3, 4],
            2: [5, 6],
            3: [],
            4: [],
            5: [],
            6: []
        }
        self.rewards = {
            3: 10,
            4: 5,
            5: -3,
            6: 2
        }
        #Boolean value indicating whether the game uses a neural network to represent states
        self.has_repr_network = False
    #Returns the unique states (not represantations) of the game
    def get_unique_states(self):
        return self.tree.keys()
    def initialize_state(self):
        # Start from the root node
        return torch.tensor([0],dtype=torch.float32)

    def get_reward(self, node):
        if self.is_terminal(node):
            num = node.state[0].item()
            return self.rewards[num]
        return 0

    def get_actions(self, node):
        return range(0,2)
    def get_num_actions(self, state):
        return 2

    def is_terminal(self, node):
        # A state is terminal if it has no children
        num = node.state[0].item()
        return len(self.tree[num]) == 0

    def apply_action(self, state, action):
        # Move to the selected child node
        num = state[0].item()
        return torch.tensor([self.tree[num][action]],dtype=torch.float32)
    def write_game(self, actions, filename="", meta_data=[]):
        pass


#Class used for testing mcts algorithm
class ToyGameWithReprNetwork(Game):
    def __init__(self):
        self.tree = {
            0: [1, 2],      
            1: [3, 4],
            2: [5, 6],
            3: [],
            4: [],
            5: [],
            6: []
        }
        self.rewards = {
            3: 10,
            4: -3,
            5: 25,
            6: 12
        }
        self.state_size = len(self.tree.keys())
        self.has_repr_network = True
        self.repr_network = Representation(self.state_size, 2, self.state_size)
        self.repr_optimizer =  optim.Adam(self.repr_network.parameters(), lr=0.01)
        self.initialize_random_games()

    def initialize_random_games(self):
        self.random_games = []
        for i in range(15):
            game = []
            state = 0
            while len(self.tree[state]) != 0:
                action = random.choice(self.get_actions("whatever"))
                state = self.tree[state][action]
                game.append((state, action))
            if game in self.random_games:
                #Make sure we get unique games
                i-=1
                continue
            self.random_games.append(game)
                
    #Returns the unique states (not represantations) of the game
    def get_unique_states(self):
        return self.tree.keys()

    def initialize_state(self):
        # Start from the root node
        return torch.zeros(self.state_size)

    def get_current_state(self, node):
        actions = []
        while node.parent != None:
            actions.append(node.action)
            node = node.parent
        
        actions.reverse()
        new_state = 0
        for action in actions:
            new_state = self.tree[new_state][action]
        return new_state
    
    def get_reward(self, node):
        if self.is_terminal(node):
            new_state = self.get_current_state(node)
            return self.rewards[new_state]
        return 0

    def get_actions(self, node):
        return range(0,2)
    
    def get_num_actions(self, state):
        return 2

    def is_terminal(self, node):
        new_state = self.get_current_state(node)

        # A state is terminal if it has no children
        return len(self.tree[new_state]) == 0

    def apply_action(self, state, action):
        action_tensor = torch.zeros(2)
        action_tensor[action] = 1
        return self.repr_network(torch.concat((state, action_tensor)))
    
    def write_game(self, actions, filename="", meta_data=[]):
        pass



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

    def get_actions(self):
        raise NotImplementedError
    
    def get_legal_moves(self, node):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    def apply_action(self, state, action):
        raise NotImplementedError
    def present_action(self, action):
        #Will be overwritten in assemblygame
        return action


class ToyGameWithReprNetwork(Game):
    def __init__(self, depth=5, width=2):
        self.depth = depth
        self.width = width
        self.tree = self.generate_random_tree(depth, width)
        self.max_reward = -float('inf')
        self.rewards = self.generate_random_rewards()
        self.state_size = len(self.tree.keys())
        self.has_repr_network = True
        self.repr_size = self.state_size
        self.repr_network = Representation(self.state_size, self.width, self.repr_size)
        self.repr_optimizer = optim.Adam(self.repr_network.parameters(), lr=0.0001, weight_decay=1e-5)

    def generate_random_tree(self, depth, width):
        tree = {}
        next_node_id = 1  # Start assigning node IDs from 1 (0 is the root)
        current_level = [0]  # Start with the root node

        for _ in range(depth):
            next_level = []
            for node in current_level:
                # Each node will have 'width' children
                children = list(range(next_node_id, next_node_id + width))
                tree[node] = children
                next_level.extend(children)
                next_node_id += width
            current_level = next_level
        
        # Leaf nodes should have empty children lists
        for node in current_level:
            tree[node] = []
        self.nrof_leaves = len(current_level)
        return tree

    def generate_random_rewards(self):
        rewards = {}
        for node, children in self.tree.items():
            if not children:  # This is a leaf node
                rewards[node] = random.randint(-150, 150)  # Assign random rewards between -50 and 50
                if rewards[node] > self.max_reward:
                    self.max_reward = rewards[node]
        return rewards

    # Returns the unique states (not representations) of the game
    def get_unique_states(self):
        return self.tree.keys()

    def initialize_state(self):
        # Start from the root node
        return torch.zeros(self.state_size)

    def get_current_state(self, node):
        actions = []
        while node.parent is not None:
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
            return self.rewards.get(new_state, 0)
        return 0

    def get_actions(self):
        return torch.arange(self.width)

    def get_num_actions(self):
        return self.width

    def get_legal_moves(self, node):
        return torch.tensor([1] * self.width)

    def is_terminal(self, node):
        new_state = self.get_current_state(node)
        # A state is terminal if it has no children
        return len(self.tree[new_state]) == 0

    def apply_action(self, state, action):
        action_tensor = torch.zeros(self.width)
        action_tensor[action] = 1
        return self.repr_network(torch.concat((state, action_tensor)))
    
    def write_game(self, actions, filename="", meta_data=[]):
        pass



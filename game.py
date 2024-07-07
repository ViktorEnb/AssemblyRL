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
        # Define the tree structure
        self.tree = {
            0: [1, 2],      # Root node
            1: [3, 4],      # Level 1 nodes
            2: [5, 6],
            3: [],          # Leaf nodes with rewards
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

    def initialize_state(self):
        # Start from the root node
        return 0

    def get_reward(self, state):
        if self.is_terminal(state):
            return self.rewards[state]
        return 0

    def get_actions(self, state):
        # The actions are the children of the current state
        return self.tree[state]

    def is_terminal(self, state):
        # A state is terminal if it has no children
        return len(self.tree[state]) == 0

    def apply_action(self, state, action):
        # Move to the selected child node
        return action



class Node:
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0
        self.is_expanded = False

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, reward):
        self.visit_count += 1
        self.total_reward += reward
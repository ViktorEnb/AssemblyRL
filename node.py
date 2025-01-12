class Node:
    def __init__(self, state, parent, action = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0
        self.is_expanded = False
        self.action = action

    def add_child(self, action, child_node):
        self.children[action] = child_node
    def update(self, reward):
        self.visit_count += 1
        self.total_reward += reward
    def get_actions(self):
        node = self
        actions = []
        while node.parent != None:
            actions.append({"action": node.action})
            node = node.parent
        actions.reverse()
        return actions
    def unique_id(self):
        if self.parent == None:
            #It's the root
            return "0"
        id = self.parent.unique_id() + ":" + str(self.action)
        return id
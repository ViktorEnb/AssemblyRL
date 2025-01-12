import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from node import Node
from game import Game
import threading 
import numpy as np
import yaml


with open('config.yaml', 'r') as f:
    params = yaml.safe_load(f)


"""
This class is used to create nice diagrams of the mcts search, including the visit count of each node, simulated rewards etc.
It can be updated live as the training is going on
"""
class MCTSPlotter:
    def __init__(self, root : Node, game: Game):
        self.root = root
        self.current_level = 0
        self.game = game
        self.nodes = {}
        self.nodes_at_level = {}

        self.G = nx.DiGraph()
        self.G.add_node(self.root.unique_id(), reward = self.root.total_reward, visit_count = self.root.visit_count, action = None, level=0)
        self.app = Dash(__name__)
        self.app.layout = html.Div([
            dcc.Graph(id='live-graph'),
            dcc.Interval(id='interval-component', interval=500, n_intervals=0)  #Update every 500 ms
        ])
        self._setup_callbacks()


    def calculate_xpos(self, id : str, level : int):
        #Spread out all nodes at a certain level evenly over the x-axis
        actions = [int(action) for action in id.split(":")]
        #Make use of lexiograpical python comparison, ie [3,2,1] < [3,2,3]
        #float(inf) are added to center in case there's only one or a few elements on a line
        return sum(other_actions < actions for other_actions in self.nodes_at_level[level] + [[-float("inf")], [float("inf")]]) * 1.0 / (len(self.nodes_at_level[level]) + 1)
    

    def add_new_nodes(self, level, node):
        if level > self.current_level:
            return
        
        for child in node.children.values():
            id = child.unique_id()
            if not self.G.has_node(id):
                self.G.add_node(id, reward = child.total_reward, visit_count = child.visit_count, action = child.action, level=level + 1)
                id_int_list = [int(action) for action in child.unique_id().split(":")]
                try:
                    self.nodes_at_level[level + 1].append(id_int_list)
                except: 
                    self.nodes_at_level[level + 1] = [id_int_list]
                if child.parent != None:
                    #It's not the root
                    self.G.add_edge(child.parent.unique_id(), child.unique_id())
            self.add_new_nodes(level + 1, child)

    def _setup_callbacks(self):
        """Set up the callback for live updates."""
        @self.app.callback(
            Output('live-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_graph(n_intervals):
            #Recreate everything from scratch to make sure we get new rewards/visit_counts
            self.G.clear()
            self.nodes_at_level.clear()
            self.nodes_at_level[0] = [[0]]
            self.G.add_node(self.root.unique_id(), reward = self.root.total_reward, visit_count = self.root.visit_count, action = 1, level=0)
            self.add_new_nodes(0, self.root)
            return self.generate_figure()

        thread = threading.Thread(target=self.run_server_in_thread)
        thread.daemon = True
        thread.start()

    def run_server_in_thread(self):
        self.app.run_server(debug=False, use_reloader=False)

    def generate_figure(self):
        """Generate the current state of the graph as a Plotly figure."""

        #Find all nodes with the most visits in order make them green and to not draw unnecessary nodes
        selected_nodes = {}
        for node in self.G.nodes:
            level = self.G.nodes[node]['level']
            if level not in selected_nodes or self.G.nodes[node]['visit_count'] > self.G.nodes[selected_nodes[level]]['visit_count']:
                selected_nodes[level] = node
        
        nodes_to_remove = []
        for node in self.G.nodes():
            level = self.G.nodes[node]['level']
            parent_node = next((edge[0] for edge in self.G.edges() if edge[1] == node), None)

            # Check if the parent is not the most visited in this level
            if parent_node:
                if parent_node not in selected_nodes.values():
                    nodes_to_remove.append(node)
                    #Remove from nodes_at_level list as well
                    for level in range(0, self.current_level + 2):
                        try:
                            self.nodes_at_level[level].remove([int(action) for action in node.split(":")])
                        except:
                            pass
        
        self.G.remove_nodes_from(nodes_to_remove)

        pos_dict = {
            node: {
                "x": self.calculate_xpos(node, self.G.nodes[node]['level']),
                "y": 1 - self.G.nodes[node]["level"] * 1.0 / (self.current_level + 2) , 
            }
            for node in self.G.nodes
        } 

        # Determine node colors
        node_colors = [
            'green' if node in selected_nodes.values() else 'lightblue' 
            for node in self.G.nodes()
        ]

        #sizes = [(self.G.nodes[node]['reward'] + 1) * 20 for node in self.G.nodes()]
        sizes = [20 for node in self.G.nodes()]
        hover_texts = [
            f"""
            {self.game.present_action(self.G.nodes[node]['action'])}<br>
            Reward: {self.G.nodes[node]['reward'] / self.G.nodes[node]['visit_count'] if self.G.nodes[node]['visit_count'] != 0 else 0}<br>
            Visit count: {self.G.nodes[node]['visit_count']}<br>
            Reward term: {reward_term}<br>
            Exploration term: {exploration_term} <br>
            Total UCT value: {uct_value} <br> 
            """ 
            for node in self.G.nodes()
            for reward_term, exploration_term, uct_value in [self.get_uct_terms(self.G.nodes[node]['reward'], self.G.nodes[node]['visit_count'])]
        ]

        fig = go.Figure()
        
        # Add edges
        for edge in self.G.edges():
            x0, y0 = pos_dict[edge[0]]["x"], pos_dict[edge[0]]["y"] 
            x1, y1 = pos_dict[edge[1]]["x"], pos_dict[edge[1]]["y"]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    hoverinfo='none'
                )
            )
        x_coords = [pos["x"] for pos in pos_dict.values()]
        y_coords = [pos["y"] for pos in pos_dict.values()]

        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=node_colors,
                    line=dict(width=2, color='black')
                ),
                text=[str(self.G.nodes[node]['action']) for node in self.G.nodes()],
                textposition='top center',
                hovertext=hover_texts,
                hoverinfo='text'
            )
        )
        
        # Update layout to ensure vertical space expands dynamically
        fig.update_layout(
            height=(self.current_level + 2) * 200,
            title="MCTS Reward Plotter",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.1, 1.1]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

    def get_uct_terms(self, reward, visit_count):
        C = params["mcts_exploration"]
        total_reward = reward / 200 #Since our max reward is 200 we have to scale it to make it fair for exploration value
        visit_count = visit_count

        reward_term = (total_reward / (visit_count + 1e-6)) 

        exploration_term = C * np.sqrt(np.log(visit_count + 1) / (visit_count + 1e-6)) 
        uct_value = reward_term + exploration_term
        # network_term = "NaN"
        # if hasattr(self, 'network'):
        #     network_term = D * self.network(state)[action]

        return reward_term, exploration_term, uct_value
    def reset(self, root):
        self.root = root
        self.nodes = {}
        self.nodes_at_level = {}
        self.current_level = 0
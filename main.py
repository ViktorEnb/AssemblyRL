from agent import Agent
from matrix import Swap2Elements, MatrixMultiplication
if __name__ == "__main__":
    repr_size = 32
    hidden_size = 32
    game = Swap2Elements(repr_size, hidden_size)
    agent = Agent(game, repr_size, game.get_num_actions("whatever"), load=False)
    agent.train(num_iterations=100)  
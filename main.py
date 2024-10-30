from agent import Agent
from matrix import Swap2Elements, MatrixMultiplication, DotProduct2x1, DotProduct1x1
import psutil
import os

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.IDLE_PRIORITY_CLASS)

    repr_size = 32
    hidden_size = 32
    game = DotProduct2x1(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False, save=True)
    agent.train(num_iterations=1000)  
from agent import Agent
from assembly import  AssemblyGame


if __name__ == "__main__":
    repr_size = 32
    hidden_size = 32
    game = AssemblyGame(repr_size, hidden_size)
    agent = Agent(game, repr_size, game.get_num_actions("whatever"))
    agent.train(num_iterations=1)  
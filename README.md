# Using reinforcement learning for finding shorter assembly programs for matrix multiplication
This is a personal project by Viktor Enbom that was inspired by [this DeepMind paper](https://www.nature.com/articles/s41586-023-06004-9). The main motivation was to use a similar model as the authors of the DeepMind paper, and use it to look for clever 2x2 matrix multiplication algorithms. The original goal was to independently discover something like the [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), using reinforcement learning. As it turned out however, the search space for 2x2 matrix multiplication was too large for my algorithm to effectively search through it, as it requires 35+ lines of code (compared to the target algorithms in the DeepMind paper which only required around 10 lines of code). Therefore, I decided to limit the target algorithm to something simpler, a 2x1 vector dot product. 

# Overview of algorithm
Before delving into the details, let's describe the basic aim for this project and how we're going to approach it. We're looking for algorithms that can compute the dot product of two 2-d vectors with few lines of assembly code. The process of finding assembly algorithms will be looked at as a single-player game where an agent makes a sequence of moves and subsequently receives a reward. A move is just a line of assembly code (we will restrict the number of allowed assembly lines heavily, e.g. for matrix multiplication only `LOAD`, `ADD` and `MUL` are required), and the reward is calculated based on how correct the algorithm is (measured by automatically generated test-cases) and how short the algorithm is. The agent is a reinforcement learning entity which uses the MCTS algorithm combined with a policy network (and sometimes a value network, more on this later) for selecting promising moves. The network is trained on past games played by the agent, taking into account each move and the calculated reward.

# MCTS algorithm

# Utilizing a neural network for better MCTS 

# n-d vector representation of assembly algorithms


# Implementing a points system for target algorithms

 

# Results





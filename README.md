# Using reinforcement learning for finding shorter assembly programs for matrix multiplication
This is a personal project by Viktor Enbom that was inspired by [this DeepMind paper](https://www.nature.com/articles/s41586-023-06004-9). The main motivation was to use a similar model as the authors of the DeepMind paper, and use it to look for clever 2x2 matrix multiplication algorithms. The original goal was to independently discover something like the [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), using reinforcement learning. As it turned out however, the search space for 2x2 matrix multiplication was too large for my algorithm to effectively search through it, as it requires 35+ lines of code (compared to the target algorithms in the DeepMind paper which only required around 10 lines of code). Therefore, I decided to limit the target algorithm to something simpler, a 2x1 vector dot product. 

# Overview of algorithm
Before delving into the details, let's describe the basic aim for this project and how we're going to approach it. We're looking for algorithms that can compute the dot product of two 2-d vectors with few lines of assembly code. The process of finding assembly algorithms will be looked at as a single-player game where an agent makes a sequence of moves and subsequently receives a reward. A move is just a line of assembly code (we will restrict the number of allowed assembly lines heavily, e.g. for matrix multiplication only `LOAD`, `ADD` and `MUL` are required), and the reward is calculated based on how correct the algorithm is (measured by automatically generated test-cases) and how short the algorithm is. The agent is a reinforcement learning entity which uses the MCTS algorithm combined with a policy network (and sometimes a value network, more on this later) for selecting promising moves. The network(s) is trained on past games played by the agent, taking into account each move and the calculated reward.

# MCTS algorithm

The MCTS algorithm is used for traversing through a general tree where each leaf node is associated with a number, and maximizing the .

# Utilizing a neural network for better MCTS 

# n-d vector representation of assembly algorithms
In order to use a policy (or value) network, one needs to be able to input a vector to the network which in some way represents the assembly program to be evaluated. For this project, I decided to use a RNN-like representation network. The representation network takes a N+D-dimensional vector as input and outputs a D dimensional vector, where D is the size of the representation vector and N is the number of allowed lines of assembly.  As the number of allowed assembly lines are heavily restricted, one can iterate through each allowed assembly line one by one, and assign a number to every allowed line. If a program consists of lines corresponding to the numbers $l_1, l_2, l_3$, the representation of the program is $\text{RNN}(\text{RNN}(\text{RNN}([0]^D, L_1), L_2), L3)$ ($[0]^D$ is the representation of an empty program and $L_1,L_2,L_3$ are one-hot vectors with a 1 at $l_1, l_2, l_3$ respectively). The representation network is trained through backpropagation as the policy (or value) network is trained. 

This approach differs from the representation algorithm in the DeepMind paper, where they use a transformer-like architecture. It's known that RNN tend to struggle to learn long sequences, which impacts this project negatively. Maybe I'll try to implement a transformer-like representation algorithm for sequence based games in the future, it'd be a good way to learn about transformers.

# Implementing a points system for target algorithms
We want to reward algorithms for being correct and fast, but also prioritize correctness over fastness (we shouldn't give a higher score for a fast-but-wrong algorithm than a correct-but-slow algorithm). Correctness is evaluated by generating test cases for each target algorithm, the algorithm gets points for each correctly passed test case, but also gets partial points if it partially passes a test. Fastness is evaluated by the length of the program (there's no branching so the length of the program will correlate very strongly to the time of execution).

```
passed_cases = self.get_nrof_passed_test_cases(printf)
reward = passed_cases

#Give extra points for passing all tests making it impossible for a fast 
#but wrong algorithm to beat a slow but correct algorithm
if passed_cases == 100:
    reward += 50
    
#Give up to 50 points for fast algorithm if the algorithm is atleast somewhat-correct
if passed_cases >= 40:
    reward += 50.0 * (self.max_lines - len(decoded_actions) + self.min_lines) / self.max_lines
```

The partial points for partially passed tests is calculated according to the following code in `ger_nrof_passed_test_cases`

```
#Passing a test case should be more important than passing lots of elements without passing cases
return passed_tests * 70.0 / len(self.targets) + passed_element_counter * 30.0 / element_counter
```
 

# Results





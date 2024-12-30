# Using reinforcement learning for finding shorter assembly programs for matrix multiplication
This is a personal project by Viktor Enbom that was inspired by [this DeepMind paper](https://www.nature.com/articles/s41586-023-06004-9). In the DeepMind paper the authors demonstrate a new algorithm for sorting arrays with 3 elements which is one line shorter than the state of art, entirely discovered by reinforcement learning. The main motivation for this project is to use a similar model as the authors of the DeepMind paper, and use it to search for clever 2x2 matrix multiplication algorithms. The original goal was to independently discover something like the [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), using reinforcement learning. As it turned out however, the search space for 2x2 matrix multiplication was too large for my algorithm to effectively search through it, as it requires 35+ lines of code (compared to the target algorithms in the DeepMind paper which only required around 10 lines of code). Therefore, I decided to limit the target algorithm to something simpler, a 2x1 vector dot product. 

# Overview of algorithm
Before delving into the details, let's describe the basic aim for this project and how we're going to approach it. We're looking for algorithms that can compute the dot product of two 2-d vectors with few lines of assembly code. The process of finding assembly algorithms will be looked at as a single-player game where an agent makes a sequence of moves and subsequently receives a reward. A move is just a line of assembly code (we will restrict the number of allowed assembly lines heavily, e.g. for matrix multiplication only `MOV`, `ADD` and `MUL` are required), and the reward is calculated based on how correct the algorithm is (measured by automatically generated test-cases) and how short the algorithm is. The agent is a reinforcement learning entity which uses the MCTS algorithm combined with a policy network (and sometimes a value network, more on this later) for selecting promising moves. The network(s) is trained on past games played by the agent, taking into account each move and the reward.

# MCTS algorithm

The MCTS algorithm is used for traversing through a general tree where each leaf node is associated with a number, and maximizing the .

# Utilizing a neural network for better MCTS 

# n-d vector representation of assembly algorithms
In order to use a policy (or value) network, one needs to be able to input a vector to the network which in some way represents the assembly program to be evaluated. For this project, I decided to use a RNN-like representation network. The representation network takes a N+D-dimensional vector as input and outputs a D dimensional vector, where D is the size of the representation vector and N is the number of allowed lines of assembly.  As the number of allowed assembly lines are heavily restricted, one can iterate through each allowed assembly line one by one, and assign a number to every line. If a program consists of lines corresponding to the numbers $l_1, l_2, l_3$, then the representation of the program is $\text{RNN}(\text{RNN}(\text{RNN}([0]^D, L_1), L_2), L3)$ ($[0]^D$ is the representation of an empty program and $L_1,L_2,L_3$ are one-hot vectors with a 1 at $l_1, l_2, l_3$ respectively). The representation network is trained through backpropagation as the policy (or value) network is trained. 

This approach differs from the representation algorithm in the DeepMind paper, where they use a transformer-like architecture. It's known that RNN tend to struggle to learn long sequences, which impacts this project negatively. Maybe I'll try to implement a transformer-like representation algorithm for sequence based games in the future, it'd be a good way to learn about transformers.

# "Vocabulary" of a target algorithm
With the code in this repository, it's quite easy to create a new target algorithm to train the reinforcement learning agent on. There are only two things that has to be done: create test cases to evaluate the algorithm's correctness, and define a vocabulary for the target algorithm. The vocabulary consists of which assembly instructions the agent can make use of, but also which addresses in memory the agent can read from and which registers it can use. The vocabulary of the target algorithm for calculating 2x1 vector dot products looks like this
```
self.assembly.registers = ["%%eax", "%%ebx", "%%ecx", "%%edx"]
self.assembly.target_mem_locs = ["(%2)"]        
self.assembly.input_mem_locs = ["(%0)", "4(%0)", "(%1)", "4(%1)"]
    
#All assembly instructions required for matrix multiplication
self.assembly.vocab = [
    "movl REG, REG",
    "movl IMEM, REG",
    "movl REG, TMEM",
    "imull REG, REG",
    "add REG, REG",
    "END"
]
```
In order to significantly reduce the size of the search space, I prevent the agent from choosing obviously incorrect moves. The agent is explicitly prevented from doing any of the following
- Using the value in a register before first allocating something from memory to that register
- Moving from a register to itself
- Allocating to the same memory address multiple times

# Implementing a points system for target algorithms
We want to reward algorithms for being correct and fast, but also prioritize correctness over fastness (we shouldn't give a higher score for a fast-but-wrong algorithm than a correct-but-slow algorithm). Correctness is evaluated by generating test cases for each target algorithm. A assembly program with the test cases as input is run from python using the [PeachPy library](https://pypi.org/project/PeachPy/). The algorithm gets points for each correctly passed test case, but also gets partial points if it partially passes a test.  Fastness is evaluated by the length of the program (there's no branching so the length of the program will correlate very strongly to the time of execution). The scoring system is laid out simply in the following code, with this system the maximum possible score for an algorithm is 200.

```.
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
As soon as we encounter an algorithm which gets a higher reward than any previous algorithsm, we save it to a file to make sure we don't lose track of good algorithms.

# Results
While testing out and tweaking the algorithm, adjusting parameters and so on, I created several different games of varying difficulty to test the algorithm on. 
- "ToyGame", a simple tree I created with parametrizable depth and width where each leaf gets a random reward. Useful for only testing MCTS without the assembly representation involved
- "SimplestAssemblyGame" move a value from one memory address to another.
- "Swap2Elements" move two values in an array to a new array in reverse order.
- **"DotProduct2x1"** This is the target algorithm we're actually interested in, given two arrays with 2 elements each, compute the dot product and save it to a new memory address



| Algorithm                                         | Name of Algorithm     | Time to Discover |
|---------------------------------------------------|------------------------|-------------------|
| ```c                                             
| void supersimple(int* input0,int* target0){      
| __asm__ (                                        
| "movl (%0) , %%eax;"                             
| "movl %%eax , (%1);"                             
| :                                                
| : "r"(input0),"r"(target0)                       
| : "%eax", "%ebx", "%ecx", "%edx"                 
| );                                               
| }                                                
| ```                                              | "SimplestAssemblyGame"      | < 1 second        |
| ```c                                             
| void swap2elements(int* input0,int* target0){    
| __asm__ (                                        
| "movl 4(%0) , %%eax;"                            
| "movl %%eax , (%1);"                             
| "movl (%0) , %%eax;"                             
| "movl %%eax , 4(%1);"                            
| :                                                
| : "r"(input0),"r"(target0)                       
| : "%eax", "%ebx", "%ecx", "%edx"                
| );                                               
| }                                                
| ```                                               | "Swap2Elements"       | 47 seconds         |






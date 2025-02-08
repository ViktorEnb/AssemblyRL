# Using reinforcement learning for finding shorter assembly programs for matrix multiplication
This is a personal project by Viktor Enbom that was inspired by [this DeepMind paper](https://www.nature.com/articles/s41586-023-06004-9). In the DeepMind paper the authors demonstrate a new algorithm for sorting arrays with 3 elements which is one line shorter than the state of art, entirely discovered by reinforcement learning. The main motivation for this project is to use a similar model as the authors of the DeepMind paper, and use it to search for clever 2x2 matrix multiplication algorithms. The original goal was to independently discover something like the [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm), using reinforcement learning. As it turned out however, the search space for 2x2 matrix multiplication was too large for my algorithm to effectively search through it, as it requires 35+ lines of code (compared to the target algorithms in the DeepMind paper which only required around 10 lines of code). Therefore, I decided to limit the target algorithm to something simpler, a 2x1 vector dot product. 

# Overview of algorithm
Before delving into the details, let's describe the basic aim for this project and how we're going to approach it. We're looking for algorithms that can compute the dot product of two 2-d vectors using few lines of assembly code. The process of finding assembly algorithms will be looked at as a single-player game where an agent makes a sequence of moves and subsequently receives a reward. A move is just a line of assembly code (we will restrict the number of allowed assembly lines heavily, e.g. for matrix multiplication only `MOV`, `ADD` and `MUL` are required), and the reward is calculated based on how correct the algorithm is (measured by automatically generated test-cases) and how short the algorithm is. The agent is a reinforcement learning entity which uses the MCTS algorithm combined with a policy network (and sometimes a value network, more on this later) for selecting promising moves. The network(s) is trained on past games played by the agent, taking into account each move and the reward.

# MCTS algorithm
The MCTS algorithm is used for traversing through a general tree where each leaf node is associated with a reward, maximizing the  end-node reward. This framework fits our problem nicely, each node will represent a certain state of the assembly program and selecting a child of that node will represent adding a certain line of assembly to the end of the program; the root of the tree will represent an empty program. The basic principle in MCTS is to iteratively select nodes based on exploitation (nodes which give high reward) and exploration (nodes which haven't been visited much). The MCTS algorithm is computed by performing the 3 steps below iteratively until a high enough reward has been achieved or for a fixed number of iterations.

1. Selection: Start from the root node, and select the child with the highest UCT value. The UCT value of a node is calculated with the formula:

   $`
   \frac{R}{N} + c \cdot \sqrt{\frac{\ln N}{N}} \; \; \; \; \; \; \; \; \; \;(1)
   `$

   where \( N \) is the number of times the node has been visited before (visit count), \( R \) is the average reward achieved in previous iterations, and \( c \) is a parameter decided based on the nature of the tree. Note that the higher the average reward a node has and the less it's been visited, the more likely it is to be selected. Each time a node is visited, increment its visit count and continue until an unvisited node has been reached.
 2. Simulating a reward: When arriving at an unvisited node with $`N = 0`$, randomly traverse the rest of the tree until reaching a leaf node. 
 3. Backpropagating the reward up the tree: After having reached a leaf node and having received a reward, update the average reward of all nodes which have been traversed.

# Utilizing a neural network for better MCTS 
Neural networks can be used to make the MCTS algorithm more efficient in a few different ways. There are 3 different improvements to the MCTS algorithm using neural networks which I've experimented with in this project, namely:
1. Implement a policy network and add a term $`d \cdot \text{Policy\_NN}(\text{state})`$ to eq. (1), making the MCTS favor nodes which are favored by the policy network.
2. Instead of randomly traversing through the rest of the tree in step 2, choose them randomly with a probability distribution equal to the softmax of the policy network of all the nodes in that step. This makes the simulation of a reward more accuarate.
3. Replace step 2 with simply using the value predicted by a value network, speeding up the computation significantly and possibly giving more accurate predictions if the value network is trained well.

Note that method 2 and 3 cannot be implemented simultaneously. In this project we didn't have much success with method 3, so for the most part we used method 1 and 2 to improve the MCTS algorithm.
# n-d vector representation of assembly algorithms
In order to use a policy (or value) network, one needs to be able to input a vector to the network which in some way represents the assembly program to be evaluated. For this project, I decided to use a RNN-like representation network. The representation network takes a N+D-dimensional vector as input and outputs a D dimensional vector, where D is the size of the representation vector and N is the number of allowed lines of assembly.  As the number of allowed assembly lines are heavily restricted, one can iterate through each allowed assembly line one by one, and assign a number to every line. If a program consists of lines corresponding to the numbers $l_1, l_2, l_3$, then the representation of the program is $\text{RNN}(\text{RNN}(\text{RNN}([0]^D, L_1), L_2), L3)$ ($[0]^D$ is the representation of an empty program and $L_1,L_2,L_3$ are one-hot vectors with a 1 at $l_1, l_2, l_3$ respectively). The representation network is trained through backpropagation as the policy (or value) network is trained. 

This approach differs from the representation algorithm in the DeepMind paper, where they use a transformer-like architecture. It's known that RNN tend to struggle to learn long sequences, which impacts this project negatively. Maybe I'll try to implement a transformer-like representation algorithm for sequence based games in the future, it'd be a good way to learn about transformers.
# Visualization of RL simulations
Mainly for debugging reasons, I also created a visualizer with the Plotly library for this project. In the picture below the result after 500 simulations where the target algorithm is Swap2Elements is shown, with the green nodes being the selected nodes at each step. The selected lines of code, together with the reward term, exploration term and policy network term is shown to the left. This information can be seen for any node by hovering over it.
![assemblyrl](https://github.com/user-attachments/assets/b0312349-9ef2-4615-8fd0-2bd6e2edcf4d)

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


<table style="background: none;">
  <thead>
    <tr>
      <th>Algorithm Name</th>
      <th>Algorithm Code</th>
      <th>Discovery Time</th>
      <th>Reward</th>
      <th>Analysis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SimpleAssemblyGame</td>
      <td>
        <pre>
void supersimple(int* input0,int* target0){ 
    __asm__ ( 
        "movl (%0) , %%eax;" 
        "movl %%eax , (%1);" 
        : 
        : "r"(input0),"r"(target0)
        : "%eax"
    ); 
}
        </pre>
      </td>
      <td>< 1 second</td>
      <td> 200</td>
      <td>
       -
      </td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Swap2Elements</td>
      <td>
        <pre>
void swap2elements(int* input0,int* target0){ 
    __asm__ ( 
        "movl 4(%0) , %%eax;" 
        "movl %%eax , (%1);" 
        "movl (%0) , %%ebx;" 
        "movl %%ebx , 4(%1);" 
        : 
        : "r"(input0),"r"(target0)
        : "%eax", "%ebx"
    ); 
}
        </pre>
      </td>
      <td>43 seconds</td>
       <td> 200</td>
       <td>
       -
      </td>
    </tr>
    <tr>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>DotProduct2x1</td>
      <td>
        <pre>
void dotproduct(int* input0,int* input1,int* target0){ 
      __asm__ ( 
      "movl (%0) , %%eax;",   
      "movl (%1) , %%ecx;",    
      "imull %%eax , %%ecx;",
      "movl %%ecx , %%eax;",
      "movl 4(%0) , %%ebx;",  
      "movl 4(%1) , %%ecx;",
      "imull %%ebx , %%ecx;",
      "add %%ecx , %%eax;",
      "movl %%eax , (%2;)",
      : 
      : "r"(input0),"r"(input1),"r"(target0)
       : "%eax", "%ebx", "%ecx" 
      ); 
} 
        </pre>
      </td>
      <td>17 hours, 42 minutes</td>
      <td>
       195
      </td>
       <td>
       The algorithm is almost perfect except for the erroneous $movl %%ecx, %%eax;" on line 4. I'm quite impressed that the agent was able to accomplish close to a perfect algorithm with this method, as the search space for this problem is huge. The number of legal moves after each line in this algorithm is 12, 19, 19, 34, 34, 34, 34, 31, 31; meaning that the total number of legal games is around 5 trillion. 
      </td>
    </tr>
  </tbody>
</table>
        


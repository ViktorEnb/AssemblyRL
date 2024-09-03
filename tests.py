import torch
import torch.nn as nn
import torch.optim as optim

from assembly import AssemblyGame
from game import Game, ToyGameWithReprNetwork
import subprocess
from agent import Agent
from matrix import MatrixMultiplication, Swap2Elements

def train_on_toy_game():
    #Trains an agent on an extremely simple game
    game = ToyGameWithReprNetwork()
    repr_size = 7
    num_actions = 2
    agent = Agent(game, repr_size, repr_size, num_actions)
    agent.train(num_iterations=40)
    print("\n\n\n")
    agent.print_network_predictions()
    agent.play_game()


def test_legal_moves():
    repr_size = 32
    hidden_size = 32
    #Make the results non-random to be able to test performance on different machines
    torch.manual_seed(1)
    game = MatrixMultiplication(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False)
    current = agent.mcts.root
    agent.mcts.expand(current)
    instructions = [
        "movl 8(%0) %%ecx", 
        "add %%ecx %%ecx", 
        "movl (%1) %%ecx"  
    ]
    for instruction in instructions:
        current = current.children[game.assembly.encode(instruction)]  
        agent.mcts.expand(current)
    print(game.get_legal_moves(current))

def train_on_swap_2_elements():
    repr_size = 32
    hidden_size = 32
    #Make the results non-random to be able to test performance on different machines
    torch.manual_seed(1)
    game = Swap2Elements(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False)
    agent.train(num_iterations=10)  
    agent.play_game()


def test_swap_2_elements():
    #Tests that basic assembly for swapping two numbers get 100% pass rate
    game = Swap2Elements(32, 32)
    swap_instructions = ["movl (%0) %%eax", "movl 4(%0) %%ebx", "movl %%ebx (%1)", "movl %%eax 4(%1)"]
    swap_instructions_encode = [game.assembly.encode(line) for line in swap_instructions]
    print(game.get_reward(swap_instructions_encode))

def test_matrix_multiplication():
    game = MatrixMultiplication(32, 32)
    instructions = [
                #calculating target[0]
                "movl (%0) %%eax",   
                "movl 4(%0) %%ebx",  
                "movl (%1) %%ecx",    
                "movl 8(%1) %%edx",
                "imull %%eax %%ecx",
                "imull %%ebx %%edx",
                "add %%ecx %%edx",
                "movl %%edx (%2)",

                #calculating target[1]
                "movl (%0) %%eax",   
                "movl 4(%0) %%ebx",  
                "movl 4(%1) %%ecx",    
                "movl 12(%1) %%edx",
                "imull %%eax %%ecx",
                "imull %%ebx %%edx",
                "add %%ecx %%edx",
                "movl %%edx 4(%2)",

                #calculating target[2]
                "movl 8(%0) %%eax",   
                "movl 12(%0) %%ebx",  
                "movl (%1) %%ecx",    
                "movl 8(%1) %%edx",
                "imull %%eax %%ecx",
                "imull %%ebx %%edx",
                "add %%ecx %%edx",
                "movl %%edx 8(%2)",

                #calculating target[3]
                "movl 8(%0) %%eax",   
                "movl 12(%0) %%ebx",  
                "movl 4(%1) %%ecx",    
                "movl 12(%1) %%edx",
                "imull %%eax %%ecx",
                "imull %%ebx %%edx",
                "add %%ecx %%edx",
                "movl %%edx 12(%2)"
                ]
    print(len(instructions))
    matmul_instructions_encode = [game.assembly.encode(line) for line in instructions]
    # print(game.get_reward(matmul_instructions_encode))

def test_matrix_encoder():
    game = MatrixMultiplication(32, 32) 
    print(game.assembly.decode(game.assembly.encode("imull %%eax %%ebx")))

def test_load():
    #Make sure that agent.play_game() is the same as loaded_agent.play_game
    #Be careful to not run this if you have an important model saved because
    #this may overwrite it
    repr_size = 32
    hidden_size = 32

    game = Swap2Elements(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False, save=True)
    agent.train(num_iterations=1)  
    agent.play_game()

    new_game = Swap2Elements(repr_size, hidden_size)
    loaded_agent = Agent(new_game, repr_size, hidden_size, new_game.get_num_actions(), load=True, save=False)
    loaded_agent.play_game()  


if __name__ == "__main__":
    train_on_toy_game()
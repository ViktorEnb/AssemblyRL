import torch
import torch.nn as nn
import torch.optim as optim

from assembly import AssemblyGame
from game import Game, ToyGame, ToyGameWithReprNetwork
import subprocess
from agent import Agent

def train_on_toy_game():
    #Trains an agent on an extremely simple game
    game = ToyGameWithReprNetwork()
    repr_size = 7
    num_actions = 2
    agent = Agent(game, repr_size, num_actions)
    # agent.print_network_predictions()
    agent.train(num_iterations=100)
    print("\n\n\n")
    agent.print_network_predictions()
    agent.play_game()
    
def cmd_test():
    #Tests compiling c file from cmd
    subprocess.run(["gcc", "-g", "-o", "cmd_test", "cmd_test.c"])
    printf = subprocess.run(["cmd_test"], capture_output=True, text=True).stdout
    buffer = printf.split("\n")
    reward = 0
    print(buffer)
    if int(buffer[0]) == 2:
        reward += 1
    if int(buffer[1]) == 1:
        reward += 1
    print(reward) 

def test_cases():
    #Tests that basic assembly for swappign two numbers get 100% pass rate
    game = AssemblyGame(32, 32)
    swap_instructions = ["movl (%0) %%eax", "movl 4(%0) %%ebx", "movl %%ebx (%1)", "movl %%eax 4(%1)"]
    swap_instructions_encode = [game.assembly.instruction_encode(line) for line in swap_instructions]
    print(swap_instructions_encode)
    # game.write_game([1,2], filename="testing_test_cases.c")
    print(game.get_reward(swap_instructions_encode))

if __name__ == "__main__":
    train_on_toy_game()
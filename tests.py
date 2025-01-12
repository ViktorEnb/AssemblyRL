import torch
import torch.nn as nn
import torch.optim as optim

from assembly import AssemblyGame
from game import Game, ToyGameWithReprNetwork
import subprocess
from agent import Agent
from matrix import MatrixMultiplication, Swap2Elements, SimplestAssemblyGame, DotProduct2x1, DotProduct1x1
from peachpy import *
from peachpy.x86_64 import *
from node import Node

#For peachpy test
import ctypes
import numpy as np

import time 
def train_on_toy_game():
    #Trains an agent on an extremely simple game
    game = ToyGameWithReprNetwork()
    repr_size = len(game.tree.keys())
    num_actions = game.width
    agent = Agent(game, repr_size, repr_size, num_actions)
    agent.train(num_iterations=200)
    print("\n\n\n")
    # agent.print_network_predictions()
    agent.play_game()
    print("max reward is " + str(game.max_reward))
    print("tree size is " + str(game.nrof_leaves))


def train_on_simple_assembly_game():
    #Trains an agent on an extremely simple game
    repr_size = 12
    hidden_size = 12
    game = SimplestAssemblyGame(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False)
    agent.train(num_iterations=300)  
    agent.play_game()

def test_legal_moves():
    repr_size = 32
    hidden_size = 32
    #Make the results non-random to be able to test performance on different machines
    torch.manual_seed(1)
    game = DotProduct2x1(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False)
    current = agent.mcts.root
    agent.mcts.expand(current)
    instructions = [
        "movl (%0) %%eax",
        "movl %%eax (%2)"
        # "movl 4(%0) %%eax",
        # "movl 4(%0) %%ebx",
        # "movl %%ebx 4(%2)"
    ]
    for instruction in instructions:
        action = game.assembly.encode(instruction) 
        child = Node(state=None, parent=current, action=action)
        current.add_child(action, child)
        current = current.children[action]  
        agent.mcts.expand(current)
    print(game.get_legal_moves(current))
    
def train_on_swap_2_elements():
    repr_size = 32
    hidden_size = 32
    game = Swap2Elements(repr_size, hidden_size)
    agent = Agent(game, repr_size, hidden_size, game.get_num_actions(), load=False)
    time.sleep(5)
    agent.train(num_iterations=300)  
    agent.play_game()

def train_on_multiplication_with_headstart():
    game = DotProduct1x1(16,16)
    agent = Agent(game, 16,16, game.get_num_actions(), load=False)
    root = Node(state=game.initialize_state(), parent=None)
    root1 = Node(state=game.apply_action(root.state, game.assembly.encode("movl (%0) %%eax")), action=game.assembly.encode("movl (%0) %%eax"),parent=root)
    root2 = Node(state=game.apply_action(root1.state, game.assembly.encode("movl (%1) %%ebx")), action=game.assembly.encode("movl (%1) %%ebx"),parent=root1)
    root3 = Node(state=game.apply_action(root2.state, game.assembly.encode("imull %%ebx %%eax")), action=game.assembly.encode("imull %%ebx %%eax"),parent=root2)
    agent.mcts.root = root3
    agent.train(num_iterations=1)
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
    print(game.get_reward(matmul_instructions_encode))

def test_dot_product():
    game = DotProduct2x1(32, 32)
    instructions = [
                "movl (%0) %%eax",   
                "movl (%1) %%ecx",    
                "imull %%eax %%ecx",
                "movl 4(%0) %%ebx",  
                "movl 4(%1) %%eax",
                "imull %%ebx %%eax",
                "add %%ecx %%eax",
                "movl %%eax (%2)",
                "END"
                ]
    node = Node(state=game.initialize_state(), parent=None)
    for instruction in instructions:
        encoded_action = game.assembly.encode(instruction)
        node = Node(state=game.apply_action(node.state, encoded_action), parent=node, action=encoded_action)
    print(game.get_reward(node))

def test_multiplication():
    game = DotProduct1x1(32, 32)
    instructions = [
                "movl (%0) %%eax",   
                "movl (%1) %%ebx",    
                "imull %%eax %%ebx",
                "movl %%ebx (%2)",
                "END"
                ]
    node = Node(state=game.initialize_state(), parent=None)
    for instruction in instructions:
        encoded_action = game.assembly.encode(instruction)
        node = Node(state=game.apply_action(node.state, encoded_action), parent=node, action=encoded_action)
    print(game.assembly.encode("movl (%0) %%eax"))
    print(game.assembly.encode("movl (%1) %%ebx"))
    print(game.assembly.encode("imull %%ebx %%eax"))
    print(game.assembly.encode("movl %%eax (%2)"))

    print(game.get_reward(node))

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

def test_plotter():
    game = ToyGameWithReprNetwork()
    repr_size = len(game.tree.keys())
    num_actions = game.width
    agent = Agent(game, repr_size, repr_size, num_actions)
    time.sleep(5)
    agent.train(num_iterations=1)

def peachpy_matmul():
    a = Argument(ptr(const_int32_t))  # pointer to matrix A
    b = Argument(ptr(const_int32_t))  # pointer to matrix B
    c = Argument(ptr(int32_t))        # pointer to matrix C (output)
    with Function("matmul_2x2", (a, b, c)) as function:
        # Load base addresses of matrices A, B, and C into general-purpose registers
        reg_a = GeneralPurposeRegister64()
        reg_b = GeneralPurposeRegister64()
        reg_c = GeneralPurposeRegister64()

        LOAD.ARGUMENT(reg_a, a)  # Load matrix A base address
        LOAD.ARGUMENT(reg_b, b)  # Load matrix B base address
        LOAD.ARGUMENT(reg_c, c)  # Load matrix C base address

        # Temporary registers for elements of matrix A, B, and the result C
        reg_a00 = GeneralPurposeRegister32()  # A[0][0]
        reg_a01 = GeneralPurposeRegister32()  # A[0][1]
        reg_a10 = GeneralPurposeRegister32()  # A[1][0]
        reg_a11 = GeneralPurposeRegister32()  # A[1][1]

        reg_b00 = GeneralPurposeRegister32()  # B[0][0]
        reg_b01 = GeneralPurposeRegister32()  # B[0][1]
        reg_b10 = GeneralPurposeRegister32()  # B[1][0]
        reg_b11 = GeneralPurposeRegister32()  # B[1][1]

        reg_c00 = GeneralPurposeRegister32()  # C[0][0]
        reg_c01 = GeneralPurposeRegister32()  # C[0][1]
        reg_c10 = GeneralPurposeRegister32()  # C[1][0]
        reg_c11 = GeneralPurposeRegister32()  # C[1][1]

        # Load elements of matrix A
        MOV(reg_a00, [reg_a + 0])  # A[0][0]
        MOV(reg_a01, [reg_a + 4])  # A[0][1]
        MOV(reg_a10, [reg_a + 8])  # A[1][0]
        MOV(reg_a11, [reg_a + 12]) # A[1][1]

        # Load elements of matrix B
        MOV(reg_b00, [reg_b + 0])  # B[0][0]
        MOV(reg_b01, [reg_b + 4])  # B[0][1]
        MOV(reg_b10, [reg_b + 8])  # B[1][0]
        MOV(reg_b11, [reg_b + 12]) # B[1][1]

        # Compute C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
        MOV(reg_c00, reg_a00)    # C[0][0] = A[0][0]
        IMUL(reg_c00, reg_b00)   # C[0][0] = A[0][0] * B[0][0]
        MOV(reg_c01, reg_a01)    # C[0][1] = A[0][1]
        IMUL(reg_c01, reg_b10)   # C[0][1] = A[0][1] * B[1][0]
        ADD(reg_c00, reg_c01)    # C[0][0] += C[0][1]

        # Store C[0][0] in the output matrix C
        MOV([reg_c + 0], reg_c00)

        # Compute C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
        MOV(reg_c01, reg_a00)    # C[0][1] = A[0][0]
        IMUL(reg_c01, reg_b01)   # C[0][1] = A[0][0] * B[0][1]
        MOV(reg_c00, reg_a01)    # C[0][0] = A[0][1]
        IMUL(reg_c00, reg_b11)   # C[0][0] = A[0][1] * B[1][1]
        ADD(reg_c01, reg_c00)    # C[0][1] += C[0][0]

        # Store C[0][1] in the output matrix C
        MOV([reg_c + 4], reg_c01)

        # Compute C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
        MOV(reg_c10, reg_a10)    # C[1][0] = A[1][0]
        IMUL(reg_c10, reg_b00)   # C[1][0] = A[1][0] * B[0][0]
        MOV(reg_c11, reg_a11)    # C[1][1] = A[1][1]
        IMUL(reg_c11, reg_b10)   # C[1][1] = A[1][1] * B[1][0]
        ADD(reg_c10, reg_c11)    # C[1][0] += C[1][1]

        # Store C[1][0] in the output matrix C
        MOV([reg_c + 8], reg_c10)

        # Compute C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]
        MOV(reg_c11, reg_a10)    # C[1][1] = A[1][0]
        IMUL(reg_c11, reg_b01)   # C[1][1] = A[1][0] * B[0][1]
        MOV(reg_c10, reg_a11)    # C[1][0] = A[1][1]
        IMUL(reg_c10, reg_b11)   # C[1][0] = A[1][1] * B[1][1]
        ADD(reg_c11, reg_c10)    # C[1][1] += C[1][0]

        # Store C[1][1] in the output matrix C
        MOV([reg_c + 12], reg_c11)

        RETURN()
        
    # Finalize the function to generate machine code
    matmul_2x2_function = function.finalize(abi.detect()).encode().load()
    A = np.array([[1.0, 2.0],
                [3.0, 4.0]], dtype=np.int32)

    B = np.array([[5.0, 6.0],
                [7.0, 8.0]], dtype=np.int32)

    # Prepare an output matrix C to store the result (2x2)
    C = np.zeros((2, 2), dtype=np.int32)

    # Step 2: Convert matrices to raw pointers (C-style)
    # Allocate memory for the matrices using ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # Step 3: Ensure that the PeachPy function has been finalized
    # (Assuming `matmul_2x2_function` is already finalized as shown in the previous steps)

    # Step 4: Call the function
    # This assumes that `matmul_2x2_function` has been compiled to callable machine code
    matmul_2x2_function(A_ptr, B_ptr, C_ptr)

    # Step 5: Print the resulting matrix C
    print("Matrix C (Result of A * B):")
    print(C)

if __name__ == "__main__":
    train_on_swap_2_elements()
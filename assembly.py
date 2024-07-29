from game import Game
import torch
from torch import optim
from network import Representation
import subprocess
from datetime import datetime
import os

class Assembly:
    def __init__(self):
        self.init_registers()
        self.init_memory_location()
        self.init_vocab()
        self.calculate_vocab_size()
    
    def init_vocab(self):
        #Describes all allowed assembly lines  
        self.vocab = ["movl REG, REG",
        "movl MEM, REG",
        "movl REG, MEM",
        "END"
        ]

    def init_registers(self):
        #All allowed registers
        self.registers = ["%%eax", "%%ebx"]

    def init_memory_location(self):
        #All allowed memory locations
        self.mem_locs = ["(%0)", "4(%0)", "(%1)", "4(%1)"]

    def calculate_vocab_size(self):
        #Calculates the number of allowed assembly-lines
        self.vocab_size = 0
        self.line_sizes = []
        for line in self.vocab:
            possible_lines = 1
            words = line.split(" ")
            words = [word.replace(",", "") for word in words]
            for word in words:
                if word == "REG":
                    possible_lines *= len(self.registers)
                elif word == "MEM":
                    possible_lines *= len(self.mem_locs)
            self.line_sizes.append(possible_lines)
            self.vocab_size += possible_lines
    
    def instruction_encode(self, instruction):
        #Converts an assembly instruction in the vocabulary to a number
        #The idea is to have all instructions be iterable from 1 to the number of possible instructions
        self.previous_line_sizes = 0
        for (i, line) in enumerate(self.vocab):
            words_in_line = line.split(" ")
            words_in_instruction = instruction.split(" ")

            if len(words_in_line) != len(words_in_instruction):
                self.previous_line_sizes += self.line_sizes[i]
                continue
                
            current_line_index = 0
            remaining = self.line_sizes[i]
            for (j, word_in_line) in enumerate(words_in_line):
                word_in_line = word_in_line.replace(",","")
                word_in_instruction = words_in_instruction[j]
                if word_in_line == "REG" and word_in_instruction in self.registers:
                    remaining //= len(self.registers)
                    current_line_index += remaining * self.registers.index(word_in_instruction)
                elif word_in_line == "MEM" and word_in_instruction in self.mem_locs:
                    remaining //= len(self.mem_locs)
                    current_line_index += remaining * self.mem_locs.index(word_in_instruction)
                elif word_in_line == word_in_instruction:
                    pass
                else:
                    break

                if j == len(words_in_line) - 1:
                    return self.previous_line_sizes + current_line_index
                
            self.previous_line_sizes += self.line_sizes[i]
    
    def instruction_decode(self, num):
        #Converts a num to an assembly instruction
        cumulative = 0
        line_nr = -1
        while cumulative <= num and line_nr < len(self.line_sizes) - 1:
            line_nr += 1
            cumulative += self.line_sizes[line_nr]
        num -= cumulative
        words = list(reversed(self.vocab[line_nr].split(" ")))
        instruction = []
        commas = []
        #Words that contain a comma
        for i in range(len(words)):
            if "," in words[i]:
                words[i] = words[i].replace(",", "")
                commas.append(i)
        print(words)
        
        for (index, word) in enumerate(words):
            if index in commas:
                instruction.append(",")
            if word == "REG":
                instruction.append(self.registers[num % len(self.registers)])
                num = (num - num % len(self.registers)) // len(self.registers)
            elif word == "MEM":
                instruction.append(self.mem_locs[num % len(self.mem_locs)])
                num = (num - num % len(self.mem_locs)) // len(self.mem_locs)
            else:
                instruction.append(word)
            
        
        return " ".join(list(reversed(instruction)))


class AssemblyGame(Game):
    def __init__(self, repr_size, hidden_size):
        self.assembly = Assembly()
        self.repr_size = repr_size
        self.repr_network = Representation(self.repr_size, self.assembly.vocab_size, hidden_size)
        self.repr_optimizer = optim.Adam(self.repr_network.parameters(), lr=0.001)
        self.time_started = datetime.now() 
        self.generate_test_cases()
        if not self.validate_test_cases():
            print("Test cases are in the wrong format.")

    def initialize_state(self):
        return torch.zeros(self.repr_size)

    def generate_test_cases(self):
        #Has to be modified based on the target algorithm
        self.test_cases = []
        self.targets = []
        for _ in range(2):
            test_case = torch.randint(0,10,(2,))
            self.test_cases.append([test_case])
            target = test_case.flip(dims=(-1,))
            self.targets.append([target])

    def validate_test_cases(self):
        shapes = [a.shape for a in self.test_cases[0]]
        self.nrof_inputs = len(shapes)
        for case in self.test_cases:
            if len(case) != self.nrof_inputs:
                print("Inconsistent number of inputs")
                return False
            for i in range(len(case)):
                inp = case[i]
                if inp.shape != shapes[i]:
                    print("Inconsistent shape of inputs")
                    return False
        shapes = [a.shape for a in self.targets[0]]
        self.nrof_targets = len(shapes)
        for case in self.targets:
            if len(case) != self.nrof_targets:
                print("Inconsistent number of targets")
                return False
            for i in range(len(case)):
                inp = case[i]
                if inp.shape != shapes[i]:
                    print("Inconsistent shape of targets")
                    return False
        return True
                
    def get_nrof_passed_test_cases(self, stdout):
        #Compares the stdout from the program which was run to the target
        #Returns the total number of test cases which passed
        
        #Todo: give some points for partially correct tests
        element_counter = 0
        passed_tests = 0
        for i in range(len(self.targets)):
            passed = True
            for j in range(self.nrof_targets):
                nrof_elements = self.targets[i][j].numel()
                target_list = self.targets[i][j].reshape(nrof_elements,)
                for k in range(nrof_elements):
                    if float(stdout[element_counter]) != target_list[k].item():
                        passed = False
                    element_counter += 1
            if passed:
                passed_tests += 1

        return passed_tests

    #Runs the assembly program and calculates reward based on 
    #correctness and time of execution
    def get_reward(self, actions : list[tuple]):
        self.write_game(actions)
        subprocess.run(["gcc", "-o", "fast_swapping", "fast_swapping.c"])
        printf = subprocess.run(["fast_swapping"], capture_output=True, text=True).stdout
        passed_cases = self.get_nrof_passed_test_cases(printf)
        total_cases = len(self.targets)
        
        #Percentage of passed tests
        reward = int(passed_cases * 100.0 / total_cases)
        #Only take into account execution time if all test cases are passed
        if reward == 100:
            reward += int(1.0 / (len(actions) + 1) * 100)

        return reward
    

    #Number of  allowed lines of assembly
    def get_num_actions(self, state):
        return self.assembly.vocab_size
    
    def get_actions(self, node):
        return range(0, self.get_num_actions(node.state))

    def is_terminal(self, node):
        #Check for END of program line
        if node.action == 0:
            return True
        counter = 0
        while node.parent != None:
            counter += 1
            node = node.parent
        return counter >= 7    

    def apply_action(self, state, action : int):
        action_onehot = torch.zeros(self.get_num_actions(state))
        action_onehot[action] = 1
        return self.repr_network(torch.cat((state, action_onehot)))


    def write_game(self, actions, filename=None, meta_info = []):
        #Path for storing c-file.
        training_path = ""
        if filename == None:
            filename="fast_swapping.c"
        else:
            training_path = os.path.join(".", "best_algos")#, self.time_started.strftime("%m-%d-%H-%M"))
            if not os.path.exists(training_path):
                os.makedirs(training_path)

        filename = os.path.join(training_path, filename)
        with open(filename, "w") as f:
            f.write("#include <stdio.h> \n")
            f.write("#include <stdlib.h> \n")
            
            #Create arguments to swap function
            input_args = ["int* input" + str(k) for k in range(self.nrof_inputs)]
            target_args = ["int* target" + str(k) for k in range(self.nrof_targets)]
            args = ",".join(input_args + target_args)

            f.write("void swap(" + args + "){ \n")
            if len(actions) > 1:
                f.write("__asm__ ( \n")
            
            for line in actions:
                #Check for END
                if line == 0:
                    break
                f.write("\"" + self.assembly.instruction_decode(line) + ";\" \n")
            
            if len(actions) > 1:
                f.write(": \n")
                f.write(": \"r\"(" + args.replace("int* ", "").replace(",", "),\"r\"(") + ")\n ")
                f.write(": \"%eax\", \"%ebx\" \n")
                f.write("); \n")
            f.write("} \n")

            f.write("int main(int argc, char* argv[]){ \n")

            #Todo: change this to C for-loops. The generated code tends to get very long 
            # as it works now, which makes compile time increase
            for i in range(len(self.test_cases)):
                inputs = self.test_cases[i]
                targets = self.targets[i]
                
                #Initialize input arrays
                for j in range(self.nrof_inputs):
                    nrof_elements = inputs[j].numel()   
                    case_list = inputs[j].reshape(nrof_elements,).tolist()
                    #Only define type in the first test case, otherwise we get redefinition errors
                    if i==0:
                        f.write("int* ")

                    f.write("input" + str(j) + " = malloc(sizeof(int) * " + str(nrof_elements) + "); \n")
                    for k in range(nrof_elements):
                        f.write("input" + str(j) + "[" + str(k) + "] = " + str(case_list[k]) + "; \n")
                
                #Initialize target arrays
                for j in range(self.nrof_targets):
                    nrof_elements = targets[j].numel()   
                    case_list = targets[j].reshape(nrof_elements,).tolist()
                    #Only define type in the first test case, otherwise we get redefinition errors
                    if i==0:
                        f.write("int* ")

                    f.write("target" + str(j) + " = malloc(sizeof(int) * " + str(nrof_elements) + "); \n")
                
                f.write("swap(" + args.replace("int* ","") + "); \n")

                #Print target results
                for j in range(self.nrof_targets):
                    nrof_elements = targets[j].numel()   
                    for k in range(nrof_elements):
                        f.write("printf(\"%d\", target" + str(j) + "[" + str(k) + "]); \n")

            f.write("} \n")

            if len(meta_info) > 0:
                f.write("//META INFO \n")            
            for line in meta_info:
                f.write("//" + line + "\n")


if __name__ == "__main__":
    a = Assembly()
    num = a.instruction_encode("mov %rbx -0x4(%rbp)")
    print(num)
    print(a.instruction_decode(num))

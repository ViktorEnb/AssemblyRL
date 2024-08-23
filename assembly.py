from game import Game
import torch
from torch import optim
from network import Representation
import subprocess
from datetime import datetime
import os
import time 

def compile_and_link(c_path, exe_path, num=0):
    subprocess.run(["gcc", "-O0", "-c", c_path, "-o", "./tmp/tmp" + str(num) + ".o"], check=True)
    subprocess.run(["gcc", "./tmp/tmp" + str(num) + ".o", "./tmp/main.o", "-o", exe_path], check=True)

def delete_files(name, num):
    os.remove("./tmp/" + name + str(num) + ".c")
    os.remove("./tmp/tmp" + str(num) + ".o")
    os.remove("./tmp/"+ name + str(num) + ".exe")



class Assembly:
    def __init__(self):
        pass

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
                elif word == "IMEM":
                    possible_lines *= len(self.input_mem_locs)
                elif word == "TMEM":
                    possible_lines *= len(self.target_mem_locs)
            self.line_sizes.append(possible_lines)
            self.vocab_size += possible_lines
    
    def encode(self, instruction):
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
                elif word_in_line == "IMEM" and word_in_instruction in self.input_mem_locs:
                    remaining //= len(self.input_mem_locs)
                    current_line_index += remaining * self.input_mem_locs.index(word_in_instruction)
                elif word_in_line == "TMEM" and word_in_instruction in self.target_mem_locs:
                    remaining //= len(self.target_mem_locs)
                elif word_in_line == word_in_instruction:
                    pass
                else:
                    break

                if j == len(words_in_line) - 1:
                    return self.previous_line_sizes + current_line_index
                
            self.previous_line_sizes += self.line_sizes[i]
    
    def decode(self, num):
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
                        
        for (index, word) in enumerate(words):
            if index in commas:
                instruction.append(",")
            if word == "REG":
                instruction.append(self.registers[num % len(self.registers)])
                num = (num - num % len(self.registers)) // len(self.registers)
            elif word == "IMEM":
                instruction.append(self.input_mem_locs[num % len(self.input_mem_locs)])
                num = (num - num % len(self.input_mem_locs)) // len(self.input_mem_locs)
            elif word == "TMEM":
                instruction.append(self.target_mem_locs[num % len(self.target_mem_locs)])
                num = (num - num % len(self.target_mem_locs)) // len(self.target_mem_locs)
            else:
                instruction.append(word)
            
        
        return " ".join(list(reversed(instruction)))


class AssemblyGame(Game):
    def __init__(self, repr_size, hidden_size):
        self.assembly = Assembly()
        self.init_vocab()   
        self.repr_size = repr_size
        self.repr_network = Representation(self.repr_size, self.assembly.vocab_size, hidden_size)
        self.repr_optimizer = optim.Adam(self.repr_network.parameters(), lr=0.001)
        self.time_started = datetime.now() 
        self.generate_test_cases()
        if not self.validate_test_cases():
            print("Test cases are in the wrong format.")
        self.set_algo_name()
        for filename in os.listdir("./tmp"):
            file_path = os.path.join("./tmp", filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        self.write_main()
        self.write_header_file()
        subprocess.run(["gcc", "-c", "./tmp/main.c", "-o", "./tmp/main.o"])
        self.current_files = []

    
    def initialize_state(self):
        return torch.zeros(self.repr_size)

    def generate_test_cases(self):
        #Depends on the target algorithm
        raise NotImplementedError
    def set_algo_name(self):
        #Depends on the target algorithm
        raise NotImplementedError
    def init_vocab(self):
        #Depends on the target algorithm
        raise NotImplementedError
    def set_illegal_moves(self):
        #Used to prevent the agent from choosing obviously stupid moves
        #in order to reduce computation
        #Depends on the target algorithm
        raise NotImplementedError
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
        
        element_counter = 0
        passed_element_counter = 0
        passed_tests = 0
        outputs = stdout.split(",")[:-1]
        for i in range(len(self.targets)):
            passed = True
            for j in range(self.nrof_targets):
                nrof_elements = self.targets[i][j].numel()
                target_list = self.targets[i][j].reshape(nrof_elements,)
                for k in range(nrof_elements):
                    if float(outputs[element_counter]) == target_list[k].item():
                        #Give some points for partially correct scores
                        passed_element_counter += 1
                    else:
                        passed= False
                    element_counter += 1
            if passed:
                passed_tests += 1

        if passed_tests == len(self.targets):
            #perfect score
            return 100
        else:
            #Passing a test case should be more important than passing lots of elements without passing cases
            return passed_tests * 70.0 / len(self.targets) + passed_element_counter * 30.0 / element_counter
    
    #Runs the assembly program and calculates reward based on 
    #correctness and time of execution
    #Best possible score is 200
    def get_reward(self, node, thread=0):
        actions = []
        if type(node) == type([]):
            actions = node
        else:           
            actions = []
            while node.parent != None:
                actions.append(node.action)
                node = node.parent
            actions.reverse()
        #Since this is multi-threaded it's important that we don't write over the same file in 
        #Different threads
        num = 0
        while num in self.current_files:
            num += 1
        self.current_files.append(num)

        self.write_game(actions, filename=os.path.join(".", "tmp", self.algo_name + str(num) + ".c"))
        c_path = os.path.join(".", "tmp", self.algo_name + str(num) + ".c")
        exe_path = os.path.join(".", "tmp", self.algo_name + str(num) + ".exe")
        compile_and_link(c_path, exe_path, num)
        printf = ""
        try:
            printf = subprocess.run([exe_path], capture_output=True, text=True).stdout
        except Exception:
            #Just try again
            time.sleep(1)
            compile_and_run(c_path, exe_path, num)
            printf = subprocess.run([exe_path], capture_output=True, text=True).stdout
        
        self.current_files.remove(num)
        delete_files(self.algo_name, num)

        passed_cases = self.get_nrof_passed_test_cases(printf)
        reward = passed_cases

        #Give extra points for passing all tests making it impossible for a fast 
        #but wrong algorithm to beat a slow but correct algorithm
        if passed_cases == 100:
            reward += 50
            
        #Give up to 50 points for fast algorithm if the algorithm is atleast somewhat-correct
        if passed_cases >= 40:
            reward += 50.0 * (self.max_lines - len(actions) + self.min_lines) / self.max_lines

        return reward
    
    def get_num_actions(self):
        return self.assembly.vocab_size
    
    def get_actions(self):
        return torch.arange(self.assembly.vocab_size)

    def is_terminal(self, node):
        #Check for END of program line
        if node.action == self.assembly.vocab_size-1:
            return True
        counter = 0
        while node.parent != None:
            counter += 1
            node = node.parent
        return counter >= self.max_lines    

    def apply_action(self, state, action : int):
        action_onehot = torch.zeros(self.get_num_actions())
        action_onehot[action] = 1
        return self.repr_network(torch.cat((state, action_onehot)))

    def get_legal_moves(self, node):
        previous_moves = torch.ones(self.get_num_actions()) * 1.0 / self.get_num_actions()
        while node.parent != None:
            previous_moves[node.action] = 1 
            node = node.parent
        
        ret = torch.matmul(self.illegal_moves_matrix, previous_moves)
        ret = torch.floor(ret)
        ret = torch.clamp(ret, max=1.0, min=0.0)

        #Below is a nice print for debugging information when creating a new target algorithm

        # for i in range(len(previous_moves)):
        #     if ret[i].item() == 1:
        #         print(self.assembly.decode(i), "   legal")
        #     else:
        #         print(self.assembly.decode(i), "   illegal")
        return ret

    def write_game(self, actions, filename, meta_info = []):
        #Create arguments to swap function
        input_args = ["int* input" + str(k) for k in range(self.nrof_inputs)]
        target_args = ["int* target" + str(k) for k in range(self.nrof_targets)]
        args = ",".join(input_args + target_args)
        with open(filename, "w") as f:   
            f.write("void " + self.algo_name + "(" + args + "){ \n")
            if len(actions) > 1:
                f.write("__asm__ ( \n")
            for line in actions:
                # print(, "   ", line, "   ", self.assembly.vocab_size)
                #Check for END
                if line == self.assembly.vocab_size - 1:
                    break
                f.write("\"" + self.assembly.decode(line) + ";\" \n")
            if len(actions) > 1:
                f.write(": \n")
                f.write(": \"r\"(" + args.replace("int* ", "").replace(",", "),\"r\"(") + ")\n ")
                f.write(": \"%eax\", \"%ebx\", \"%ecx\", \"%edx\" \n")
                f.write("); \n")
            f.write("} \n")


            if len(meta_info) > 0:
                f.write("//META INFO \n")            
            for line in meta_info:
                f.write("//" + line + "\n")
    def write_main(self):
        #Create arguments to swap function
        input_args = ["int* input" + str(k) for k in range(self.nrof_inputs)]
        target_args = ["int* target" + str(k) for k in range(self.nrof_targets)]
        args = ",".join(input_args + target_args)
        with open("./tmp/main.c", "w") as f:   
            f.write("#include <stdio.h> \n")
            f.write("#include <stdlib.h> \n")
            f.write("#include \"" + self.algo_name + ".h\" \n")
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
                
                f.write(self.algo_name + "(" + args.replace("int* ","") + "); \n")

                #Print target results
                for j in range(self.nrof_targets):
                    nrof_elements = targets[j].numel()   
                    for k in range(nrof_elements):
                        f.write("printf(\"%d,\", target" + str(j) + "[" + str(k) + "]); \n")

            f.write("} \n")
    def write_header_file(self):
        input_args = ["int* input" + str(k) for k in range(self.nrof_inputs)]
        target_args = ["int* target" + str(k) for k in range(self.nrof_targets)]
        args = ",".join(input_args + target_args)
        with open("./tmp/" + self.algo_name + ".h", "w") as f:
            f.write("#ifndef " + self.algo_name.upper() + "_H \n")
            f.write("#define " + self.algo_name.upper() + "_H \n")
            f.write("#include \"" + self.algo_name + ".h\" \n")
            f.write("void " + self.algo_name + "(" + args + "); \n")
            f.write("#endif")

if __name__ == "__main__":
    a = Assembly()
    num = a.encode("mov %rbx -0x4(%rbp)")
    print(num)
    print(a.decode(num))

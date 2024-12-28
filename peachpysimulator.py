from peachpy import *
from peachpy.x86_64 import *
import ctypes
import numpy as np

class PeachPyAssemblyExecutor:
    def __init__(self, nrof_inputs, nrof_targets, test_cases, targets):
        self.arg_dict = {}
        self.input_args = []
        
        #Create arguments for the function in simulate()
        for i in range(len(test_cases[0])):
            arg_name = f"i{i}"
            self.arg_dict[arg_name] = Argument(ptr(const_int32_t), name=arg_name)
            self.input_args.append(self.arg_dict[arg_name])
        for i in range(len(targets[0])):
            arg_name = f"t{i}"
            self.arg_dict[arg_name] = Argument(ptr(int32_t), name=arg_name)
            self.input_args.append(self.arg_dict[arg_name])

        self.test_cases = test_cases
        self.targets = targets
        self.nrof_inputs = nrof_inputs
        self.nrof_targets = nrof_targets
        self.set_pointers()

    #Create pointer to objects with the values from test_cases
    def set_pointers(self):
        #A small optimization could be done here
        #The targets need to be reset after every run since they are changed
        #But the inputs don't need to be changed
        self.case_pointers = []
        self.target_numel = []
        for i in range(len(self.test_cases)):
            inputs = self.test_cases[i]
            outputs = self.targets[i]  
            pointers = []
            for j in range(self.nrof_inputs):
                dummy_object = inputs[j].numpy().astype(np.int32)
                pointers.append(dummy_object.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            for j in range(self.nrof_targets):
                #We only need the shape of target
                dummy_object = np.zeros(shape=outputs[j].shape, dtype=np.int32)
                pointers.append(dummy_object.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                if i == 0: #Don't store the same shape for every case
                    self.target_numel.append(outputs[j].numel())
            self.case_pointers.append(pointers)

    def convert_address(self, addr, GPR_dict):
        if "%%" in addr:  # Converts registers on the form %%eax
            return eval(addr.replace("%", "").lower())
        elif "(" in addr:  # Convert stuff on the form 4(%0) etc
            offset, reg_placeholder = addr.split("(")
            reg_placeholder = reg_placeholder.strip(")").replace("%","")
            offset = int(offset) if offset else 0
            if int(reg_placeholder) < self.nrof_inputs:
                reg_placeholder = "i" + reg_placeholder
            else:
                reg_placeholder = "t" + str(int(reg_placeholder) - self.nrof_inputs)
            reg = GPR_dict.get(reg_placeholder)  # Get the corresponding GeneralPurposeRegister64
            return [reg + offset]  # Return the memory reference
        
    def simulate(self, actions):
        GPR_dict = {}
        with Function("abcd", tuple(self.input_args)) as asm_function:
            #Loads in every argument in self.input_args
            for arg_name in self.arg_dict.keys():
                dummy = GeneralPurposeRegister64()
                LOAD.ARGUMENT(dummy, self.arg_dict[arg_name])
                GPR_dict[arg_name] = dummy

            for action in actions:
                words = [word for word in action.replace(",","").split(" ") if len(word) > 0] #Replace everything which isn't a operand or instruction
                if words[0] == "movl":
                    src = self.convert_address(words[1], GPR_dict)
                    dest = self.convert_address(words[2], GPR_dict)
                    MOV(dest, src)
                if words[0] == "add":
                    src = self.convert_address(words[1], GPR_dict)
                    dest = self.convert_address(words[2], GPR_dict)
                    ADD(dest, src)
                if words[0] == "imull":
                    src = self.convert_address(words[1], GPR_dict)
                    dest = self.convert_address(words[2], GPR_dict)
                    IMUL(dest, src)

            RETURN()
        
        target_function = asm_function.finalize(abi.detect()).encode().load()
        printf=""
        for i in range(len(self.case_pointers)):
            arguments = [self.case_pointers[i][k] for k in range(self.nrof_inputs + self.nrof_targets)]
            target_function(*arguments)
            for j in range(len(self.target_numel)):
                for k in range(self.target_numel[j]):
                    #Constructs a 'printf' result in the same format as main.c would print
                    printf += str(self.case_pointers[i][self.nrof_inputs + j][k])  + ","
        self.set_pointers()
        return printf
            
if __name__ == "__main__":
    #Just confirm that matrix multiplication works
    from matrix import MatrixMultiplication
    game = MatrixMultiplication(1, 1)
    peachpysimulator = PeachPyAssemblyExecutor(game.nrof_inputs, game.nrof_targets, game.test_cases, game.targets)
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
    peachpysimulator.simulate(instructions)
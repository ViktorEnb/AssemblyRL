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


        #Create pointer to objects with the values from test_cases
        self.case_pointers = []
        self.target_numel = []
        for i in range(len(test_cases)):
            inputs = test_cases[i]
            outputs = targets[i]  
            pointers = []
            for j in range(nrof_inputs):
                dummy_object = inputs[j].numpy().astype(np.int32)
                pointers.append(dummy_object.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            for j in range(nrof_targets):
                #We only need the shape of target
                dummy_object = np.zeros(shape=outputs[j].shape, dtype=np.int32)
                pointers.append(dummy_object.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                if i == 0: #Don't store the same shape for every case
                    self.target_numel.append(outputs[j].numel())
            self.case_pointers.append(pointers)
        self.nrof_inputs = nrof_inputs
        self.nrof_targets = nrof_targets
    

    def simulate(self, actions):
        with Function("abcd", tuple(self.input_args)) as asm_function:
            reg_a, reg_b = GeneralPurposeRegister64(), GeneralPurposeRegister64()
            LOAD.ARGUMENT(reg_a, self.arg_dict["i0"])
            LOAD.ARGUMENT(reg_b, self.arg_dict["t0"])
            MOV(eax, [reg_a])
            MOV([reg_b + 4], eax)
            MOV(eax, [reg_a + 4])
            MOV([reg_b], eax)
            RETURN()
        

        target_function = asm_function.finalize(abi.detect()).encode().load()
        printf=""
        for i in range(len(self.case_pointers)):
            arguments = [self.case_pointers[i][k] for k in range(self.nrof_inputs + self.nrof_targets)]
            target_function(*arguments)
            for j in range(len(self.target_numel)):
                for k in range(self.target_numel[j]):
                    #Constructs a 'printf' result in the same format as main.c would print
                    printf += self.case_pointers[i][self.nrof_inputs + j][k]  + ","
        
        return printf
            
if __name__ == "__main__":
    from matrix import Swap2Elements
    game = Swap2Elements(1, 1)
    peachpysimulator = PeachPyAssemblyExecutor(game.nrof_inputs, game.nrof_targets, game.test_cases, game.targets)
    peachpysimulator.simulate([])

from assembly import AssemblyGame
import torch


#Very basic algorithm used for testing
#My computer can find the optimal solution in about 10 minutes
class Swap2Elements(AssemblyGame):
    def generate_test_cases(self):
        #Has to be modified based on the target algorithm
        self.test_cases = []
        self.targets = []
        for _ in range(2):
            test_case = torch.randint(0,20,(2,))
            self.test_cases.append([test_case])
            target = test_case.flip(dims=(-1,))
            self.targets.append([target])
    def set_algo_name(self):
        self.algo_name = "swap2elements"

    def init_vocab(self):
        #We need 2 registers
        self.registers = ["%%eax", "%%ebx"]

        #Memory addresers for 2 arrays of length 2.
        self.mem_locs = ["(%0)", "4(%0)", "(%1)", "4(%1)"]
            
        #All assembly instructions required for swapping 2 elements
        self.vocab = ["movl REG, REG",
        "movl MEM, REG",
        "movl REG, MEM",
        "END"
        ]

        self.assembly.calculate_vocab_size()

class MatrixMultiplication(AssemblyGame):
    def generate_test_cases(self):
        #Has to be modified based on the target algorithm
        self.test_cases = []
        self.targets = []
        torch.manual_seed(1)
        for _ in range(10):
            test_case1 = torch.randint(0,20,(2,2))
            test_case2 = torch.randint(0,20,(2,2))
            self.test_cases.append([test_case1, test_case2])
            target = torch.matmul(test_case1, test_case2)
            self.targets.append([target])
    def set_algo_name(self):
        self.algo_name = "matmul"
    
    def init_vocab(self):
        #We need 4 registers
        self.assembly.registers = ["%%eax", "%%ebx", "%%ecx", "%%edx"]

        #Memory addresers for input0, input1 and target. All being 2x2 matricies.
        self.assembly.mem_locs = ["(%0)", "4(%0)", "8(%0)", "12(%0)", "(%1)", "4(%1)", "8(%1)", "12(%1)", "(%2)", "4(%2)", "8(%2)", "12(%2)"]
            
        #All assembly instructions required for matrix multiplication
        self.assembly.vocab = ["movl REG, REG",
            "movl MEM, REG",
            "movl REG, MEM",
            "imull REG, REG",
            "add REG, REG",
            "END"
        ]
        self.assembly.calculate_vocab_size()
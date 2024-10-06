from assembly import AssemblyGame
import torch


#Very basic algorithm used for testing
#My computer can find the optimal solution in about 10 minutes (30 seconds after various optimizations :))
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

    def set_illegal_moves(self):
        dim = self.get_num_actions()
        self.illegal_moves_matrix = torch.ones((dim,dim))
        #Maps registers to instructions which have register as dest
        reg_dest_map = {}
        #Maps targets to instructions which have targets as dest
        target_dest_map = {}
        for action in range(dim):
            words = self.assembly.decode(action).split(" ")
            if words[0] == "movl" and words[3] in self.assembly.registers:
                if words[3] in reg_dest_map:
                    reg_dest_map[words[3]][action] = 1
                else:
                    reg_dest_map[words[3]] = torch.zeros(dim)
                    reg_dest_map[words[3]][action] = 1
            if words[0] == "movl" and words[3] in self.assembly.target_mem_locs:
                if words[3] in target_dest_map:
                    target_dest_map[words[3]][action] = -1
                else:
                    target_dest_map[words[3]] = torch.zeros(dim)
                    target_dest_map[words[3]][action] = -1

        #Reg_dest_map has to be known in get_legal_moves
        self.reg_dest_map = reg_dest_map
        for action in range(dim):
            words = self.assembly.decode(action).split(" ")
            #Moving with the same src and dest is never allowed
            if words[0] == "movl" and words[1] == words[3]:
                self.illegal_moves_matrix[action, :] = 0
            
            #Don't allow to mov un handled registers
            elif words[0] == "movl" and words[1] in self.assembly.registers:
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]]

            #Don't allow multiple mov's to the same target

            #It's only allowed to move to a target
            #If the src reg has been filled AND we haven't allocated to this space before
            #If you think about it this works
            if words[0] == "movl" and words[3] in self.assembly.target_mem_locs:
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]]*2 + target_dest_map[words[3]]  * (2*sum(reg_dest_map[words[1]]) + 1)



    def set_algo_name(self):
        self.algo_name = "swap2elements"

    def init_vocab(self):
        #We need 2 registers
        self.assembly.registers = ["%%eax", "%%ebx"]

        #Memory addresers for 2 arrays of length 2.
        self.assembly.target_mem_locs = ["(%1)", "4(%1)"]
        self.assembly.input_mem_locs = ["(%0)", "4(%0)"]

        #All assembly instructions required for swapping 2 elements
        self.assembly.vocab = ["movl REG, REG",
        "movl IMEM, REG",
        "movl REG, TMEM",
        "END"
        ]

        self.assembly.calculate_vocab_size()
        self.set_illegal_moves()
        self.max_lines = 7
        self.min_lines = 5

class MatrixMultiplication(AssemblyGame):
    def generate_test_cases(self):
        #Has to be modified based on the target algorithm
        self.test_cases = []
        self.targets = []
        for _ in range(3):
            test_case1 = torch.randint(0,20,(2,2))
            test_case2 = torch.randint(0,20,(2,2))
            self.test_cases.append([test_case1, test_case2])
            target = torch.matmul(test_case1, test_case2)
            self.targets.append([target])
    
    def set_illegal_moves(self):
        dim = self.get_num_actions()
        self.illegal_moves_matrix = torch.ones((dim,dim))
        #Maps registers to instructions which have register as dest
        reg_dest_map = {}
        #Maps targets to instructions which have targets as dest
        target_dest_map = {}
        for action in range(dim):
            words = self.assembly.decode(action).split(" ")
            if words[0] == "movl" and words[3] in self.assembly.registers:
                if words[3] in reg_dest_map:
                    reg_dest_map[words[3]][action] = 1
                else:
                    reg_dest_map[words[3]] = torch.zeros(dim)
                    reg_dest_map[words[3]][action] = 1
            if words[0] == "movl" and words[3] in self.assembly.target_mem_locs:
                if words[3] in target_dest_map:
                    target_dest_map[words[3]][action] = -1
                else:
                    target_dest_map[words[3]] = torch.zeros(dim)
                    target_dest_map[words[3]][action] = -1
        #Reg_dest_map has to be known in get_legal_moves
        self.reg_dest_map = reg_dest_map
        for action in range(dim):
            words = self.assembly.decode(action).split(" ")
            #Moving with the same src and dest is never allowed
            if words[0] == "movl" and words[1] == words[3]:
                self.illegal_moves_matrix[action, :] = 0
            
            #Don't allow to mov un handled registers
            elif words[0] == "movl" and words[1] in self.assembly.registers:
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]]
            
            elif words[0] == "imull":
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]] * 0.5 + reg_dest_map[words[3]] * 0.5

            elif words[0] == "add":
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]] * 0.5 + reg_dest_map[words[3]] * 0.5

            #Don't allow multiple mov's to the same target

            #It's only allowed to move to a target
            #If the src reg has been filled AND we haven't allocated to this space before
            #If you think about it this works
            if words[0] == "movl" and words[3] in self.assembly.target_mem_locs:
                self.illegal_moves_matrix[action, :] = reg_dest_map[words[1]]*2 + target_dest_map[words[3]]  * (2*sum(reg_dest_map[words[1]]) + 1)



    
    def set_algo_name(self):
        self.algo_name = "matmul"
    
    def init_vocab(self):
        #We need 4 registers
        self.assembly.registers = ["%%eax", "%%ebx", "%%ecx", "%%edx"]

        #Memory addresers for input0, input1 and target. All being 2x2 matricies.
        self.assembly.target_mem_locs = ["(%2)", "4(%2)", "8(%2)", "12(%2)"]
        
        self.assembly.input_mem_locs = ["(%0)", "4(%0)", "8(%0)", "12(%0)", "(%1)", "4(%1)", "8(%1)", "12(%1)"]
            
        #All assembly instructions required for matrix multiplication
        self.assembly.vocab = ["movl REG, REG",
            "movl IMEM, REG",
            "movl REG, TMEM",
            "imull REG, REG",
            "add REG, REG",
            "END"
        ]
        self.assembly.calculate_vocab_size()

        self.set_illegal_moves()
        
        self.max_lines = 40
        self.min_lines = 29


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
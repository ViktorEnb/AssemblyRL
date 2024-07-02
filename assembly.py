class Assembly:
    def __init__(self):
        self.init_registers()
        self.init_memory_location()
        self.init_vocab()
        self.calculate_vocab_size()
    
    def init_vocab(self):
        #Describes all allowed assembly lines  
        self.vocab = ["mov REG REG",
        "mov MEM REG",
        "mov REG MEM",
        "cmp REG REG",
        "cmovl REG REG",
        ]

    def init_registers(self):
        #All allowed registers
        self.registers = ["%rax", "%rbx", "%rcx"]

    def init_memory_location(self):
        #All allowed memory locations
        self.mem_locs = ["-0x0(%rbp)", "-0x4(%rbp)", "-0x8(%rbp)"]

    def calculate_vocab_size(self):
        #Calculates the number of allowed assembly-lines
        self.size = 0
        self.line_sizes = []
        for line in self.vocab:
            possible_lines = 1
            words = line.split(" ")
            for word in words:
                if word == "REG":
                    possible_lines *= len(self.registers)
                elif word == "MEM":
                    possible_lines *= len(self.mem_locs)
            self.line_sizes.append(possible_lines)
            self.size += possible_lines
    
    def instruction_to_num(self, instruction):
        #Converts an assembly instruction in the vocabulary to a number
        self.previous_line_sizes = 0
        for (i, line) in enumerate(self.vocab):
            words_in_line = line.split(" ")
            words_in_instruction = instruction.split(" ")
            current_line_index = 0
            remaining = self.line_sizes[i]
            for (j, word_in_line) in enumerate(words_in_line):
                word_in_instruction = words_in_instruction[j]
                if word_in_line == "REG" and word_in_instruction in self.registers:
                    remaining /= len(self.registers)
                    current_line_index += remaining * self.registers.index(word_in_instruction)
                elif word_in_line == "MEM" and word_in_instruction in self.mem_locs:
                    remaining /= len(self.mem_locs)
                    current_line_index += remaining * self.mem_locs.index(word_in_instruction)
                elif word_in_line == word_in_instruction:
                    pass
                else:
                    break

                if j == len(words_in_line) - 1:
                    return self.previous_line_sizes + current_line_index
                
            self.previous_line_sizes += self.line_sizes[i]
    
    def num_to_instruction(self, num):
        cumulative = 0
        line_nr = -1
        while cumulative < num:
            line_nr += 1
            cumulative += self.line_sizes[num]
        remaining = num - self.line_sizes[num]
        for word in self.vocab[line_nr]:
            if word == "REG":
                remaining /= len(self.registers)
            elif word == "MEM":
                remaining /= len(self.mem_locs)

if __name__ == "__main__":
    a = Assembly()
    print(a.instruction_to_num("mov %rbx %rax"))

import ctypes

# Function to encode mov instructions manually
def encode_mov(destination, source):
    opcode_map = {
        ('eax', 'ebx'): b'\x89\xD8',
        ('eax', 'ecx'): b'\x89\xC8',
        ('ebx', 'eax'): b'\x89\xC3',
        ('ebx', 'ecx'): b'\x89\xCB',
        ('ecx', 'eax'): b'\x89\xC1',
        ('ecx', 'ebx'): b'\x89\xD9'
    }
    
    key = (destination, source)
    
    if key in opcode_map:
        return opcode_map[key]
    else:
        raise ValueError(f"Unsupported mov instruction: mov {destination}, {source}")

# Construct machine code for the program
def create_program():
    program = b''
    
    # mov eax, 42 (manually encoded)
    program += b'\xB8\x2A\x00\x00\x00'  # mov eax, 42
    
    # mov ebx, eax
    program += encode_mov('ebx', 'eax')
    
    # mov ecx, ebx
    program += encode_mov('ecx', 'ebx')
    
    # ret (return the value in eax/ecx)
    program += b'\xC3'
    
    return program

# Allocate executable memory and copy the machine code
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
PAGE_EXECUTE_READWRITE = 0x40

program_code = create_program()
print("program_code: " + str(program_code))
ctypes.windll.kernel32.VirtualAlloc.argtypes = (
    ctypes.c_void_p,  # LPVOID
    ctypes.c_size_t,  # SIZE_T
    ctypes.c_long,    # DWORD
    ctypes.c_long,    # DWORD
)

ctypes.windll.kernel32.VirtualAlloc.restype = ctypes.c_void_p  # LPVOID

memory_buffer = ctypes.windll.kernel32.VirtualAlloc(
    0,                         # lpAddress - NULL
    len(program_code),         # dwSize
    MEM_COMMIT | MEM_RESERVE,  # flAllocationType
    PAGE_EXECUTE_READWRITE     # flProtect
)

if not memory_buffer:
    print("VirtualAlloc call failed. Error code:", ctypes.GetLastError())
    exit(-1)

# Copy machine code into the allocated memory
ctypes.windll.kernel32.RtlMoveMemory.argtypes = (
    ctypes.c_void_p,  # VOID*
    ctypes.c_void_p,  # VOID*
    ctypes.c_size_t   # SIZE_T
)

ctypes.windll.kernel32.RtlMoveMemory(
    memory_buffer,         # Destination
    ctypes.c_char_p(program_code),  # Source
    len(program_code)      # Length
)

# Cast memory to function and execute
f = ctypes.cast(
    memory_buffer,
    ctypes.CFUNCTYPE(ctypes.c_int)  # function returns an int
)

# Execute the program and print the result
result = f()
print(f"Result: {result}")
from peachpy import *
from peachpy.x86_64 import *

x = Argument(int32_t)
y = Argument(int32_t)

with Function("Add", (x, y), (int32_t, int32_t)) as asm_function:
    reg_x = GeneralPurposeRegister32()
    reg_y = GeneralPurposeRegister32()

    LOAD.ARGUMENT(reg_x, x)
    LOAD.ARGUMENT(reg_y, y)

    # ADD(reg_x, reg_y)
    # MOV(reg_x, reg_y)
    # MOV(reg_x, y)

    y = reg_x
    RETURN(reg_x, reg_y)

python_function = asm_function.finalize(abi.detect()).encode().load()

y = 1
print(python_function(5, y)) # -> prints "4"
print(y)
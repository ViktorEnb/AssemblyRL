#include <stdio.h>
#include <stdlib.h>
void swap(int* buffer){ 
    // Inline assembly to swap buffer[0] and buffer[1]
    __asm__ (
        "mov (%0), %%eax;"    // Move buffer[0] into eax
        "mov 4(%0), %%ebx;"   // Move buffer[1] into ebx
        "mov %%ebx, (%0);"    // Move ebx (original buffer[1]) into buffer[0]
        "mov %%eax, 4(%0);"   // Move eax (original buffer[0]) into buffer[1]
        :                      // No output operands
        : "r"(buffer)          // Input operand: buffer pointer
        : "%eax", "%ebx"       // Clobbered registers
    );
} 
void test(int* buffer){ 
    // Inline assembly to swap buffer[0] and buffer[1]
    __asm__ (
        "movl 4(%0) , %%ebx;"    // Move buffer[0] into eax
        " movl %%ebx , (%0);"   // Move buffer[1] into ebx
        "movl (%0) , %%eax;"    // Move ebx (original buffer[1]) into buffer[0]
        :                      // No output operands
        : "r"(buffer)          // Input operand: buffer pointer
        : "%eax", "%ebx"       // Clobbered registers
    );
} 
int main(int argc, char* argv[]){ 
    int *buffer = malloc(sizeof(int) * 2);
    buffer[0] = 1;
    buffer[1] = 2;

    test(buffer);
    printf("%d \n", buffer[0]);
    printf("%d", buffer[1]);
    return 0;
} 

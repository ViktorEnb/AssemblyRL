#include <stdio.h>


void matmul(int *mat){
    *(mat + 1) = 1;
}
int main() {

    // __asm__ (
    //             "mov $10, %rax;"
    //             "mov %rax, -0x30(%rbp);"
    //             "mov -0x30(%rbp), %rcx;"
    //             "imul %rcx, %rax;"
    // );  

    return 0;
}
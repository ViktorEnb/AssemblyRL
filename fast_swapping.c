#include <stdio.h> 
#include <stdlib.h> 
void swap(int* input0,int* target0){ 
__asm__ ( 
"movl (%0) , %%eax;" 
"movl 4(%0) , %%ebx;" 
"movl %%ebx , (%1);" 
"movl %%eax , 4(%1);" 
: 
: "r"(input0),"r"(target0)
 : "%eax", "%ebx" 
); 
} 
int main(int argc, char* argv[]){ 
int* input0 = malloc(sizeof(int) * 2); 
input0[0] = 7; 
input0[1] = 6; 
int* target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 7; 
input0[1] = 1; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 3; 
input0[1] = 8; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 9; 
input0[1] = 2; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 2; 
input0[1] = 2; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 7; 
input0[1] = 3; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 9; 
input0[1] = 3; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 7; 
input0[1] = 8; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 4; 
input0[1] = 5; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
input0 = malloc(sizeof(int) * 2); 
input0[0] = 1; 
input0[1] = 1; 
target0 = malloc(sizeof(int) * 2); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
} 

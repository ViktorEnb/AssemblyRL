#include <stdio.h> 
#include <stdlib.h> 
void swap(int* buffer){ 
__asm__ ( 
"movl %%ebx , %%eax;" 
"movl (%0) , %%eax;" 
"movl 4(%0) , %%ebx;" 
"movl %%eax , 4(%0);" 
"movl %%ebx , (%0);" 
"movl %%eax , 4(%0);" 
"movl (%0) , %%eax;" 
: 
: "r"(buffer) 
 : "%eax", "%ebx" 
); 
} 
int main(int argc, char* argv[]){ 
int *buffer = malloc(sizeof(int) * 2); 
buffer[0] = 1; 
buffer[1] = 2; 
swap(buffer); 
printf("%d \n", buffer[0]); 
printf("%d ", buffer[1]); 
} 
//META INFO 
//Reward: 2
//Iteration: 0
//Time since start: 380

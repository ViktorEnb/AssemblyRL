#include <stdio.h> 
#include <stdlib.h> 
void swap(int* input0,int* target0){ 
__asm__ ( 
"movl %%eax , %%ebx;" 
"movl %%ebx , %%eax;" 
: 
: "r"(input0,target0)
 : "%eax", "%ebx" 
); 
} 
int main(int argc, char* argv[]){ 
int* input0 = malloc(sizeof(int) * 6); 
input0[0] = 8; 
input0[1] = 2; 
input0[2] = 0; 
input0[3] = 6; 
input0[4] = 0; 
input0[5] = 0; 
int* target0 = malloc(sizeof(int) * 30); 
swap(input0,target0); 
printf("%d", target0[0]); 
printf("%d", target0[1]); 
printf("%d", target0[2]); 
printf("%d", target0[3]); 
printf("%d", target0[4]); 
printf("%d", target0[5]); 
printf("%d", target0[6]); 
printf("%d", target0[7]); 
printf("%d", target0[8]); 
printf("%d", target0[9]); 
printf("%d", target0[10]); 
printf("%d", target0[11]); 
printf("%d", target0[12]); 
printf("%d", target0[13]); 
printf("%d", target0[14]); 
printf("%d", target0[15]); 
printf("%d", target0[16]); 
printf("%d", target0[17]); 
printf("%d", target0[18]); 
printf("%d", target0[19]); 
printf("%d", target0[20]); 
printf("%d", target0[21]); 
printf("%d", target0[22]); 
printf("%d", target0[23]); 
printf("%d", target0[24]); 
printf("%d", target0[25]); 
printf("%d", target0[26]); 
printf("%d", target0[27]); 
printf("%d", target0[28]); 
printf("%d", target0[29]); 
} 

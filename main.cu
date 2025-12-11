#include "kernel_CPU.c"
#include "kernel.cu"
#include "support.h"
#include <stdio.h>


int main() {
    int clients  = 4;
    int periods  = 3;
    int changes[] = {
        1,  2,  3,  4,
        1, -1,  0,  2,
        2,  2, -3,  1
    };

    int size_changes = clients * periods;
    int size_account = clients * periods;
    int size_sum     = periods;
 
    int *sum = (int*)malloc(size_sum);

    int *d_changes, *d_account, *d_sum;
    cudaMalloc((void**)&d_changes, size_changes * sizeof(int));
    cudaMalloc((void**)&d_account, size_account * sizeof(int));
    cudaMalloc((void**)&d_sum, size_sum * sizeof(int));

    cudaMemcpy(d_changes, changes, size_changes * sizeof(int) , cudaMemcpyHostToDevice);

    solveGPU(d_changes, d_account, d_sum, clients, periods);
    
    cudaMemcpy(sum, d_sum, size_sum * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Sum by periode :\n");
    for (int j = 0; j < periods; j++)
        printf("periode %d = %d\n", j, sum[j]);

    // --- free GPU
    cudaFree(d_changes);
    cudaFree(d_account);
    cudaFree(d_sum);

    free(sum);

    return 0;
}
#include <stdio.h>

/***** Accounts computation thread *****/ 
__global__ void kernelComputeAccounts(const int *changes, int *account,int clients, int periods) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= clients) return;

    account[i] = changes[i]; // the first change is copied

    for (int j = 1; j < periods; j++) {
        account[j * clients + i] = account[(j - 1) * clients + i] + changes[j * clients + i];
    } 
}


/***** Sums computation thread *****/ 
__global__ void kernelSum(const int *account, int *sum, int clients, int periods) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= periods) return;

    int s = 0;
    for (int i = 0; i < clients; i++) {
        s += account[j * clients + i];
    }
    sum[j] = s;
}

void solveGPU(int *d_changes, int *d_account, int *d_sum, int clients, int periods) {
    int block_size = 256;

    int nb_blocks_clients = (clients + block_size - 1) / block_size;
    kernelComputeAccounts<<<nb_blocks_clients, block_size>>>(d_changes, d_account, clients, periods);

    int nb_blocks_periods = (periods + block_size - 1) / block_size;
    kernelSum<<<nb_blocks_periods, block_size>>>(d_account, d_sum, clients, periods);
}


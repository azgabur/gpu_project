#include <stdio.h>

#include "kernel.h"

// kernel implementations assume row-major storage of 2D data
// each client is represented by a column, each period by a row
// TODO: optimize kernels using shared memory and/or other techniques


// kernel to compute accumulating account balance per client over periods
__global__ void kernel_account_balance(const int *account_changes_d, int *account_balance_d, int clients_num, int periods_num) {
    int client_col = threadIdx.x + blockIdx.x * blockDim.x;
    if (client_col >= clients_num) return;

    // the first change is copied directly
    account_balance_d[client_col] = account_changes_d[client_col]; 

    for (int period_row = 1; period_row < periods_num; period_row++) {
        int entry_idx = period_row * clients_num + client_col;
        int prev_period_entry_idx = (period_row - 1) * clients_num + client_col;
        account_balance_d[entry_idx] = account_balance_d[prev_period_entry_idx] + account_changes_d[entry_idx];
    } 
}

// kernel to compute sums per period across all clients
__global__ void kernel_sums_per_period(const int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    
    for(int period_row = blockIdx.x; period_row < periods_num; period_row += gridDim.x) {
        int sum = 0;

        for(int client_col = tid; client_col < clients_num; client_col += blockDim.x){
            sum += account_balance_d[period_row * clients_num + client_col];
        }

        sdata[tid] = sum;
        __syncthreads();

        for(int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid == 0)
            sums_per_period_d[period_row] = sdata[0];
        __syncthreads();
    }
}

void launch_sums_per_period_kernel(const int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
    int period_blocks_num = min(periods_num, 1024);
    size_t shmem = BLOCK_SIZE * sizeof(int);
    kernel_sums_per_period<<<period_blocks_num, BLOCK_SIZE, shmem>>>(account_balance_d, sums_per_period_d, clients_num, periods_num);
}

void launch_account_balance_kernel(const int *account_changes_d, int *account_balance_d, int clients_num, int periods_num) {
    int clients_blocks_num = (clients_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_account_balance<<<clients_blocks_num, BLOCK_SIZE>>>(account_changes_d, account_balance_d, clients_num, periods_num);
}

#include <stdio.h>

#include "kernel.h"

// kernel implementations assume row-major storage of 2D data
// each client is represented by a column, each period by a row
// TODO: optimize kernels using shared memory and/or other techniques
// TODO : reunite the kernels, allocate arrays aligned to the gpu memory (other than cudamalloc) suitable 2d metrics (cudamallocpitch)
// read rows instead of columns or opposite (1thread = 1client)
// mention in the readme the possible size for parameters
// 1st kernel on columns and 2nd kernel on rows
// reducing global memory reading
// save the whole column at the end and not at every entry
// maybe not necessary the shared memory in second kernel

// // kernel to compute accumulating account balance per client over periods
// __global__ void kernel_account_balance(const int *account_changes_d, int *account_balance_d, int clients_num, int periods_num) {
//     int client_col = threadIdx.x + blockIdx.x * blockDim.x;
//     if (client_col >= clients_num) return;

//     // the first change is copied directly
//     account_balance_d[client_col] = account_changes_d[client_col]; 

//     for (int period_row = 1; period_row < periods_num; period_row++) {
//         int entry_idx = period_row * clients_num + client_col;
//         int prev_period_entry_idx = (period_row - 1) * clients_num + client_col;
//         account_balance_d[entry_idx] = account_balance_d[prev_period_entry_idx] + account_changes_d[entry_idx];
//     } 
// }

// // kernel to compute sums per period across all clients
// __global__ void kernel_sums_per_period(const int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
//     int period_row = threadIdx.x + blockIdx.x * blockDim.x;
//     if (period_row >= periods_num) return;

//     int period_sum = 0;
//     for (int client_col = 0; client_col < clients_num; client_col++) {
//         int entry_idx = period_row * clients_num + client_col;
//         period_sum += account_balance_d[entry_idx];
//     }
//     sums_per_period_d[period_row] = period_sum;
// }

// void launch_sums_per_period_kernel(const int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
//     int period_blocks_num = (periods_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     kernel_sums_per_period<<<period_blocks_num, BLOCK_SIZE>>>(account_balance_d, sums_per_period_d, clients_num, periods_num);
// }

// void launch_account_balance_kernel(const int *account_changes_d, int *account_balance_d, int clients_num, int periods_num) {
//     int clients_blocks_num = (clients_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     kernel_account_balance<<<clients_blocks_num, BLOCK_SIZE>>>(account_changes_d, account_balance_d, clients_num, periods_num);
// }


// whole rewriting

__global__ void kernel_balance_and_sums(const int *account_changes_d, int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
    //  One block = one period
    int period_row = blockIdx.x;
    if (period_row >= periods_num) return;

    // Shared memory stores balances for this period
    extern __shared__ int shared_balances[];

    int client_col = threadIdx.x;
    int balance = 0;

    // Each thread computes balance for its client up to this period
    if (client_col < clients_num) {
        for (int r = 0; r <= period_row; r++) {
            balance += account_changes_d[r * clients_num + client_col];
        }
        shared_balances[client_col] = balance;
    }

    __syncthreads();

    if (client_col == 0) {
        int period_sum = 0;
        for (int c = 0; c < clients_num; c++) {
            period_sum += shared_balances[c];
        }
        sums_per_period_d[period_row] = period_sum;
    }

    // Save final balances only for the last period
    if (period_row == periods_num - 1 && client_col < clients_num) {
        account_balance_d[client_col] = balance;
    }

}

void launch_account_balance_and_sums_kernel (const int* account_changes_d, int * account_balance_d, int *sums_per_period_d, int clients_num, int periods_num) {
    dim3 blocks(periods_num);
    dim3 threads( max(clients_num, 1024) );
    kernel_balance_and_sums<<<blocks, threads, clients_num*sizeof(int)>>>(account_changes_d, account_balance_d, sums_per_period_d, clients_num, periods_num);
}
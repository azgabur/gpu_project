#include <cuda_runtime.h>
#include <stdio.h>

#include "kernel.h"

// ============================================================
// Kernel 1 (ON GARDE L’ORIGINAL — déjà optimal)
// Each thread = 1 client
// Sequential scan over periods (excellent for GPU)
// ============================================================

__global__ void kernel_account_balance(
    const int * __restrict__ account_changes_d,
    int * __restrict__ account_balance_d,
    int clients_num,
    int periods_num)
{
    int client_col = threadIdx.x + blockIdx.x * blockDim.x;
    if (client_col >= clients_num) return;

    // First period
    account_balance_d[client_col] = account_changes_d[client_col];

    for (int period_row = 1; period_row < periods_num; period_row++) {
        int idx = period_row * clients_num + client_col;
        int prev = (period_row - 1) * clients_num + client_col;
        account_balance_d[idx] =
            account_balance_d[prev] + account_changes_d[idx];
    }
}

// ============================================================
// Kernel 2 (OPTIMISÉ) — reduction par période
// Warp-level primitives, très peu de synchronisations
// ============================================================

__global__ void kernel_sums_per_period_fast(
    const int * __restrict__ account_balance_d,
    int * __restrict__ sums_per_period_d,
    int clients_num)
{
    int period = blockIdx.x;
    int tid = threadIdx.x;

    int sum = 0;

    // Strided access (coalescé)
    for (int i = tid; i < clients_num; i += blockDim.x) {
        sum += account_balance_d[period * clients_num + i];
    }

    // Reduction intra-warp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // One value per warp
    __shared__ int warp_sums[32]; // max 1024 threads

    if ((tid & 31) == 0)
        warp_sums[tid >> 5] = sum;

    __syncthreads();

    // Final reduction by first warp
    if (tid < 32) {
        sum = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (tid == 0)
            sums_per_period_d[period] = sum;
    }
}

// ============================================================
// Kernel launchers (API IDENTIQUE À TON CODE)
// ============================================================

void launch_account_balance_kernel(
    const int *account_changes_d,
    int *account_balance_d,
    int clients_num,
    int periods_num)
{
    int blocks = (clients_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_account_balance<<<blocks, BLOCK_SIZE>>>(
        account_changes_d,
        account_balance_d,
        clients_num,
        periods_num
    );
}

void launch_sums_per_period_kernel(
    const int *account_balance_d,
    int *sums_per_period_d,
    int clients_num,
    int periods_num)
{
    const int THREADS = 256;
    kernel_sums_per_period_fast<<<periods_num, THREADS>>>(
        account_balance_d,
        sums_per_period_d,
        clients_num
    );
}

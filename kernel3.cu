#include <cuda_runtime.h>
#include <stdio.h>

#include "kernel.h"

/*
Memory layout:
- account_changes_d[period * clients + client]
- sums_per_period_d[period]
*/

// ============================================================
// Kernel 1 (ORIGINAL — déjà optimal)
// ============================================================

__global__ void kernel_account_balance(
    const int * __restrict__ account_changes_d,
    int * __restrict__ account_balance_d,
    int clients_num,
    int periods_num)
{
    int client = blockIdx.x * blockDim.x + threadIdx.x;
    if (client >= clients_num) return;

    int acc = account_changes_d[client];
    account_balance_d[client] = acc;

    for (int period = 1; period < periods_num; ++period) {
        int idx = period * clients_num + client;
        acc += account_changes_d[idx];
        account_balance_d[idx] = acc;
    }
}

// ============================================================
// Kernel 2 FUSIONNÉ — balance + sum
// 1 bloc = plusieurs clients
// réduction par bloc + atomic global MINIMAL
// ============================================================

__global__ void kernel_balance_and_sum_fused(
    const int * __restrict__ account_changes_d,
    int * __restrict__ sums_per_period_d,
    int clients_num,
    int periods_num)
{
    int client = blockIdx.x * blockDim.x + threadIdx.x;
    if (client >= clients_num) return;

    int acc = 0;

    // Shared buffer pour réduction intra-bloc
    extern __shared__ int shmem[];

    for (int period = 0; period < periods_num; ++period) {

        acc += account_changes_d[period * clients_num + client];

        // chaque thread écrit sa balance courante
        shmem[threadIdx.x] = acc;
        __syncthreads();

        // réduction parallèle dans le bloc
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                shmem[threadIdx.x] += shmem[threadIdx.x + stride];
            __syncthreads();
        }

        // un seul atomicAdd par bloc et par période
        if (threadIdx.x == 0)
            atomicAdd(&sums_per_period_d[period], shmem[0]);

        __syncthreads();
    }
}

// ============================================================
// Launchers
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
    // Version fusionnée → on ignore account_balance_d
    // sums_per_period_d DOIT être initialisé à 0 avant le launch

    const int THREADS = 256;
    int blocks = (clients_num + THREADS - 1) / THREADS;
    size_t shmem = THREADS * sizeof(int);

    kernel_balance_and_sum_fused<<<blocks, THREADS, shmem>>>(
        account_balance_d,   // reinterpreté comme account_changes_d
        sums_per_period_d,
        clients_num,
        periods_num
    );
}

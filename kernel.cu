#include <cstddef>
#include <stdio.h>

#include "kernel.h"

// -----------------------------------------------------------------------------
// Assumptions and data layout
// -----------------------------------------------------------------------------
// - 2D data is stored in row-major order
// - Rows correspond to periods
// - Columns correspond to clients
// - Linear index: idx = period * clients_num + client
//
// All kernels rely on this layout for correct indexing and memory coalescing.
// -----------------------------------------------------------------------------


// =============================================================================
// NAIVE KERNELS
// =============================================================================

// -----------------------------------------------------------------------------
// kernel_account_balance_naive
// -----------------------------------------------------------------------------
// Computes cumulative account balance per client across all periods.
//
// Execution model:
// - One thread per client (column-wise processing)
// - Each thread iterates sequentially over all periods
//
// Characteristics:
// - Long loop-carried dependency (balance depends on previous period)
// - Repeated global memory accesses
// - Memory-bandwidth bound
// - Limited instruction-level parallelism
// -----------------------------------------------------------------------------
__global__ void kernel_account_balance_naive (
    const int* account_changes_d,   // input: per-period account changes
    int* account_balance_d,          // output: cumulative balances
    size_t clients_num,              // number of clients (columns)
    size_t periods_num               // number of periods (rows)
) {
    // Global client index handled by this thread
    size_t client_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: ignore threads beyond client count
    if (client_col >= clients_num) return;

    // Initialize balance with first period
    account_balance_d[client_col] = account_changes_d[client_col];

    // Sequential accumulation over periods
    for (size_t period_row = 1; period_row < periods_num; period_row++) {
        size_t entry_idx = period_row * clients_num + client_col;
        size_t prev_period_entry_idx = (period_row - 1) * clients_num + client_col;

        // Read previous balance from global memory and add current change
        account_balance_d[entry_idx] =
            account_balance_d[prev_period_entry_idx] + account_changes_d[entry_idx];
    }
}


// -----------------------------------------------------------------------------
// kernel_sums_per_period_naive
// -----------------------------------------------------------------------------
// Computes total account balance per period by summing over all clients.
//
// Execution model:
// - One thread per period
// - Each thread iterates sequentially over all clients
//
// Characteristics:
// - Fully serialized reduction per period
// - No parallel reduction
// - Poor memory coalescing across threads
// - Very low memory bandwidth utilization
// -----------------------------------------------------------------------------
__global__ void kernel_sums_per_period_naive (
    const int *account_balance_d,    // input: per-client balances
    int *sums_per_period_d,          // output: per-period sums
    size_t clients_num,
    size_t periods_num
) {
    // Global period index handled by this thread
    size_t period_row = threadIdx.x + blockIdx.x * blockDim.x;

    // Bounds check
    if (period_row >= periods_num) return;

    int period_sum = 0;

    // Sequential sum over all clients
    for (size_t client_col = 0; client_col < clients_num; client_col++) {
        size_t idx = period_row * clients_num + client_col;
        period_sum += account_balance_d[idx];
    }

    // Write result
    sums_per_period_d[period_row] = period_sum;
}


// =============================================================================
// OPTIMIZED KERNELS
// =============================================================================

// -----------------------------------------------------------------------------
// kernel_account_balance_optimalized
// -----------------------------------------------------------------------------
// Optimized version of cumulative balance computation.
//
// Optimizations applied:
// - Eliminate redundant global memory reads by keeping balance in a register
// - Use __restrict__ to remove pointer aliasing assumptions
// - Period tiling to reduce loop overhead and expose limited ILP
// - Loop unrolling within the tile
//
// Execution model:
// - One thread per client
// - Each thread processes periods sequentially, but in small tiles
//
// Bottleneck:
// - Memory bandwidth (confirmed by Nsight Compute)
// -----------------------------------------------------------------------------
template <size_t PERIOD_TILE>
__global__ void kernel_account_balance_optimalized (
    const int* __restrict__ account_changes_d,
    int* __restrict__ account_balance_d,
    size_t clients_num,
    size_t periods_num)
{
    size_t client_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (client_col >= clients_num) return;

    // Load initial balance (period 0) into a register
    int balance = account_changes_d[client_col];
    account_balance_d[client_col] = balance;

    size_t period = 1;

    // Main tiled loop over periods
    for (; period + PERIOD_TILE - 1 < periods_num; period += PERIOD_TILE) {

        // Unroll to reduce loop overhead and expose ILP
        #pragma unroll
        for (int i = 0; i < PERIOD_TILE; ++i) {
            size_t idx = (period + (size_t)i) * clients_num + client_col;

            // Accumulate using register-held balance
            balance += account_changes_d[idx];
            account_balance_d[idx] = balance;
        }
    }

    // Handle remaining periods (if periods_num not divisible by tile size)
    for (; period < periods_num; ++period) {
        size_t idx = period * clients_num + client_col;
        balance += account_changes_d[idx];
        account_balance_d[idx] = balance;
    }
}


// -----------------------------------------------------------------------------
// kernel_sums_per_period_optimalized
// -----------------------------------------------------------------------------
// Optimized per-period reduction using block-parallel reduction.
//
// Optimizations applied:
// - Block-level parallelism: one block per period
// - Coalesced global memory accesses across threads
// - Shared-memory tree reduction
//
// Execution model:
// - One block computes the sum for one period
// - Threads cooperate to reduce over clients
//
// Bottleneck:
// - Memory bandwidth (near peak DRAM utilization)
// -----------------------------------------------------------------------------
template<size_t BLOCK_SIZE>
__global__ void kernel_sums_per_period_optimalized (
    const int* __restrict__ account_balance_d,
    int* __restrict__ sums_per_period_d,
    size_t clients_num,
    size_t periods_num)
{
    // Each block handles one period
    int period = blockIdx.x;

    if (period >= periods_num) return;

    int tid = threadIdx.x;
    int local_sum = 0;

    // Stride over clients: each thread processes multiple clients
    for (size_t client = tid; client < clients_num; client += BLOCK_SIZE) {
        size_t idx = (size_t)period * clients_num + client;
        local_sum += account_balance_d[idx];
    }

    // Shared memory for block-level reduction
    __shared__ int sdata[BLOCK_SIZE];
    sdata[tid] = local_sum;
    __syncthreads();

    // Tree-based reduction in shared memory
    for (size_t offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum for this period
    if (tid == 0) {
        sums_per_period_d[period] = sdata[0];
    }
}


// ============================================================================
// KERNEL LAUNCHERS
// ============================================================================

// Launch optimized per-period reduction kernel
void launch_sums_per_period_kernel_optimalized(
    const int* account_balance_d,
    int* sums_per_period_d,
    size_t clients_num,
    size_t periods_num)
{
    constexpr int BLOCK_SIZE = 256;

    // One block per period
    dim3 grid(periods_num);
    dim3 block(BLOCK_SIZE);

    kernel_sums_per_period_optimalized<BLOCK_SIZE>
        <<<grid, block>>>(
            account_balance_d,
            sums_per_period_d,
            clients_num,
            periods_num
        );
}


// Launch optimized balance kernel
void launch_account_balance_kernel_optimalized(
    const int* account_changes_d,
    int* account_balance_d,
    size_t clients_num,
    size_t periods_num)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int PERIOD_TILE = 8;

    dim3 grid((clients_num + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    kernel_account_balance_optimalized<PERIOD_TILE>
        <<<grid, block>>>(
            account_changes_d,
            account_balance_d,
            clients_num,
            periods_num
        );
}


// Launch naive per-period sum kernel
void launch_sums_per_period_kernel_naive(
    const int *account_balance_d,
    int *sums_per_period_d,
    size_t clients_num,
    size_t periods_num
) {
    constexpr int BLOCK_SIZE = 256;

    dim3 grid((periods_num + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    kernel_sums_per_period_naive
        <<<grid, block>>>(
            account_balance_d,
            sums_per_period_d,
            clients_num,
            periods_num
        );
}


// Launch naive balance kernel
void launch_account_balance_kernel_naive(
    const int *account_changes_d,
    int *account_balance_d,
    size_t clients_num,
    size_t periods_num
) {
    constexpr int BLOCK_SIZE = 256;

    dim3 grid((clients_num + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    kernel_account_balance_naive
        <<<grid, block>>>(
            account_changes_d,
            account_balance_d,
            clients_num,
            periods_num
        );
}

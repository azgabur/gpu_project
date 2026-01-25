#!/bin/bash
# BOTTLENECK ANALYSIS WITH NSIGHT COMPUTE

set -e  # exit on error

mkdir -p ncu_reports

###############################################################################
# Naive kernels
###############################################################################

# kernel_account_balance_naive
ncu --clock-control none \
    --target-processes all \
    --launch-count 1 \
    --set roofline \
    --kernel-name kernel_account_balance_naive \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --export ncu_reports/kernel_balance_naive \
    ./account_savings -t -n

ncu --import ncu_reports/kernel_balance_naive.ncu-rep \
    > ncu_reports/kernel_balance_naive.txt


# kernel_sums_per_period_naive
ncu --clock-control none \
    --target-processes all \
    --launch-count 1 \
    --set roofline \
    --kernel-name kernel_sums_per_period_naive \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --export ncu_reports/kernel_sums_naive \
    ./account_savings -t -n

ncu --import ncu_reports/kernel_sums_naive.ncu-rep \
    > ncu_reports/kernel_sums_naive.txt


###############################################################################
# Optimized kernels
###############################################################################

# kernel_account_balance_optimalized
ncu --clock-control none \
    --target-processes all \
    --launch-count 1 \
    --set roofline \
    --kernel-name kernel_account_balance_optimalized \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --export ncu_reports/kernel_balance_optimized \
    ./account_savings -t

ncu --import ncu_reports/kernel_balance_optimized.ncu-rep \
    > ncu_reports/kernel_balance_optimized.txt


# kernel_sums_per_period_optimalized
ncu --clock-control none \
    --target-processes all \
    --launch-count 1 \
    --set roofline \
    --kernel-name kernel_sums_per_period_optimalized \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --export ncu_reports/kernel_sums_optimized \
    ./account_savings -t

ncu --import ncu_reports/kernel_sums_optimized.ncu-rep \
    > ncu_reports/kernel_sums_optimized.txt

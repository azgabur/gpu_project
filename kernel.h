#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>


// Launch account balance kernel
// input:
//  - account_changes_d: device pointer to account changes array
//  - clients_num: number of clients
//  - periods_num: number of periods
// output:
//  - account_balance_d: device pointer to account balance array
void launch_account_balance_kernel_naive (
    const int *account_changes_d, 
    int *account_balance_d, 
    size_t clients_num, 
    size_t periods_num
);

// Launch sums per period kernel
// requires account_balance_d to be already computed
// input: 
//  - account_balance_d: device pointer to account balances array
//  - clients_num: number of clients
//  - periods_num: number of periods
// output:
//  - sums_per_period_d: device pointer to sums per period array
void launch_sums_per_period_kernel_naive (
    const int *account_balance_d, 
    int *sums_per_period_d, 
    size_t clients_num, 
    size_t periods_num
);

// optimized version 
void launch_account_balance_kernel_optimalized (
    const int* account_changes_d,
    int* account_balance_d,
    size_t clients_num,
    size_t periods_num
);

// optimized version
void launch_sums_per_period_kernel_optimalized (
    const int* account_changes_d,
    int* account_balance_d,
    size_t clients_num,
    size_t periods_num
);
    
#endif // KERNEL_H
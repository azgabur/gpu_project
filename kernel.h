#ifndef KERNEL_H
#define KERNEL_H


#define BLOCK_SIZE 256


// Launch account balance kernel
// input:
//  - account_changes_d: device pointer to account changes array
//  - clients_num: number of clients
//  - periods_num: number of periods
// output:
//  - account_balance_d: device pointer to account balance array
// void launch_account_balance_kernel(const int *account_changes_d, int *account_balance_d, int clients_num, int periods_num);

// Launch sums per period kernel
// requires account_balance_d to be already computed
// input: 
//  - account_balance_d: device pointer to account balances array
//  - clients_num: number of clients
//  - periods_num: number of periods
// output:
//  - sums_per_period_d: device pointer to sums per period array
// void launch_sums_per_period_kernel(const int *account_balance_d, int *sums_per_period_d, int clients_num, int periods_num);

// new one
void launch_account_balance_and_sums_kernel (const int* account_changes_d, int * account_balance_d, int *sums_per_period_d, int client_num, int periods_num);



#endif // KERNEL_H
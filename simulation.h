#ifndef SIMULATION_H
#define SIMULATION_H


#define VERIFICATION_FAILURE 0
#define VERIFICATION_SUCCESS 1

#define VERIFICATION_SUCCESS_MSG "Verification:                       SUCCESS\n"
#define VERIFICATION_FAILURE_MSG "Verification:                       FAILURE\n"

// Function to generate saving accounts data
//  saving_accounts - array to store generated saving accounts data
//  clients_num - number of clients
//  periods_num - number of periods
void generate_saving_accounts_array(int *saving_accounts, int clients_num, int periods_num);

// Function to verify results computed by GPU against CPU results
//  account_changes - original account changes generated or loaded from file
//  gpu_account_balance - computeed by GPU account balances
//  gpu_sums_per_period - computed by GPU sums per period
//  clients_num - number of clients
//  periods_num - number of periods
int verify_results_with_CPU(const int *account_changes, const int *gpu_account_balance, 
    const int *gpu_sums_per_period, int clients_num, int periods_num);

// CPU algorithm for solving the problem
void solve_CPU(const int *account_changes, int *cpu_account_balance, int *cpu_sums_per_period, int clients_num, int periods_num);


#endif // SIMULATION_H
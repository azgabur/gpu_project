#include <stdlib.h>
#include <stdio.h>

#include "simulation.h"

// TODO: Replace with actual data generation logic as needed
void generate_saving_accounts_array(int *account_changes, int clients_num, int periods_num) {
    for (int i = 0; i < clients_num * periods_num; i++) {
        // Basic implementation: random integers between -10000 and 10000
        account_changes[i] = rand() % 20000 - 10000;
    }
}

// TODO: Implement actual verification logic
int verify_results_with_CPU(const int *account_changes, const int *gpu_account_balance, 
    const int *gpu_sums_per_period, int clients_num, int periods_num) {
    
    // Allocate memory for CPU results
    int *cpu_account_balance = (int *) malloc(sizeof(int) * (size_t) clients_num * (size_t) periods_num);
    int *cpu_sums_per_period = (int *) malloc(sizeof(int) * (size_t) periods_num);

    if (cpu_account_balance == NULL || cpu_sums_per_period == NULL) {
        fprintf(stderr, "CPU memory allocation failed during verification.\n");
        return VERIFICATION_FAILURE;
    }

    // Solve on CPU
    solve_CPU(account_changes, cpu_account_balance, cpu_sums_per_period, clients_num, periods_num);
    
    // Verify results of account balance
    for (int i = 0; i < clients_num * periods_num; i++) {
        if (cpu_account_balance[i] != gpu_account_balance[i]) {
            free(cpu_account_balance);
            free(cpu_sums_per_period);
            return VERIFICATION_FAILURE;
        }
    }

    // Verify results of sums per period
    for (int i = 0; i < periods_num; i++) {
        if (cpu_sums_per_period[i] != gpu_sums_per_period[i]) {
            free(cpu_account_balance);
            free(cpu_sums_per_period);
            return VERIFICATION_FAILURE;
        }
    }

    free(cpu_account_balance);
    free(cpu_sums_per_period);

    return VERIFICATION_SUCCESS;
}


void solve_CPU(const int *account_changes, int *cpu_account_balance, int *cpu_sums_per_period, int clients_num, int periods_num) {
    for (int i = 0; i < clients_num; i++)
        // the first change is copied directly
        cpu_account_balance[i] = account_changes[i];
    for (int j = 1; j < periods_num; j++) {
        for (int i = 0; i < clients_num; i++) {
            cpu_account_balance[j*clients_num + i] = cpu_account_balance[(j-1)*clients_num + i] 
                + account_changes[j*clients_num + i];
        }
    }

    for (int j = 0; j < periods_num; j++) {
        int s = 0;
        for (int i = 0; i < clients_num; i++) {
            s += cpu_account_balance[j*clients_num + i];
        }
        cpu_sums_per_period[j] = s;
    }
}


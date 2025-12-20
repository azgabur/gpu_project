#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <getopt.h>

#include "cuda_support.h"
#include "simulation.h"
#include "benchmark.h"
#include "kernel.h"
#include "csv.h"


int main(int argc, char *argv[]) {

    /* --- --- --- --- --- ------------------------ --- --- --- --- --- */
    /* --- --- --- --- --- VARIABLES INITIALIZATION --- --- --- --- --- */

    // value of CPU verification of the GPU results 
    int verification_result;

    // timer to be used for performance measurement
    Timer timer;

    // Variables sizes
    size_t account_changes_size;
    size_t account_balance_size;
    size_t sums_per_period_size;

    // Host variables
    int *account_changes_h = NULL;
    int *account_balance_h = NULL;
    int *sums_per_period_h = NULL;

    // Device variables
    int *account_changes_d = NULL;
    int *account_balance_d = NULL;
    int *sums_per_period_d = NULL;

    // Default testing problem size
    bool test_generate = false;
    int clients_num = 1000;
    int periods_num = 1000;

    // Parse command line arguments
    int opt;
    static struct option long_options[] = {
        {"test-input",  no_argument, 0, 't'},
        {"test-clients-num", required_argument, 0, 'c'},
        {"test-periods-num", required_argument, 0, 'p'},
    };
    while ((opt = getopt_long(argc, argv, "vd:tc:p:h?", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                // enable results verification
                // enable_verification is declared in benchmark.h
                enable_verification = 1;
                break;
            case 'd':
                // set time debug level
                // time_debug_level is declared in benchmark.h
                time_debug_level = atoi(optarg);
                break;
            case 't':
                // Generate testing random input and output csv file
                test_generate = true;
                break;
            case 'c':
                // Sets custom clients number for random generation
                clients_num = atoi(optarg);
                break;
            case 'p':
                // Sets custom periods number for random generation
                periods_num = atoi(optarg);
                break;
            case 'h': case '?': default:
                fprintf(stderr, "Usage: %s [-v] [-d level] [-c clients] [-p periods] [-t]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }


    if (!test_generate && optind >= argc){
        fprintf(stderr, "Use -t to generate random input or provide path to input csv file. See -h for help.\n");
        exit(EXIT_FAILURE);
    }

    print_entry_label(PROBLEM_SETUP_START_MSG);
    set_start_time(&timer, PROBLEM_SETUP_TIME);
    if (test_generate) {  // Generate random input
        // Set sizes to be generated
        account_changes_size = sizeof(int) * (size_t) clients_num * (size_t) periods_num;
        account_balance_size = sizeof(int) * (size_t) clients_num * (size_t) periods_num;
        sums_per_period_size = sizeof(int) * (size_t) periods_num;

        // Allocate host memory
        account_changes_h = (int *) malloc(account_changes_size);
        account_balance_h = (int *) malloc(account_balance_size);
        sums_per_period_h = (int *) malloc(sums_per_period_size);

        // Check host memory allocation (malloc returns NULL if allocation fails)
        if (account_changes_h == NULL || sums_per_period_h == NULL || account_balance_h == NULL) {
            fprintf(stderr, HOST_ALLOC_ERR_MSG);
            exit(EXIT_FAILURE);
        }

        // Initialize host memory with generated data
        generate_saving_accounts_array(account_changes_h, clients_num, periods_num);
        if (save_csv(TESTING_MATRIX_PATH, account_changes_h, periods_num, clients_num)) {
            fprintf(stderr, "Saving of randomly generated input matrix failed.\n");
        }
    }
    else {  // Load csv from file
        char* csv_file_path = argv[optind];
        if (load_csv(
            csv_file_path,
            &account_changes_size,
            &account_balance_size,
            &sums_per_period_size,
            &account_changes_h,
            &account_balance_h,
            &sums_per_period_h,
            &clients_num,
            &periods_num
        )){
            fprintf(stderr, "Loading of '%s' csv file failed.\n", argv[optind]);
            exit(EXIT_FAILURE);
        }
    }

    set_end_time(&timer, PROBLEM_SETUP_TIME);
    print_elapsed_time(&timer, PROBLEM_SETUP_TIME, OPERATION_COMPLETED_MSG);

    
    /* --- --- --- --- --- ------------------------ --- --- --- --- --- */
    /* --- --- --- --- --- DEVICE MEMORY ALLOCATION --- --- --- --- --- */

    print_entry_label(DEVICE_ALLOC_START_MSG);
    set_start_time(&timer, DEVICE_ALLOC_TIME);

    // Allocate device memory
    CUDA_SAFE_CALL( cudaMalloc((void**) &account_changes_d, account_changes_size), DEVICE_ALLOC_ERR_MSG );
    CUDA_SAFE_CALL( cudaMalloc((void**) &account_balance_d, account_balance_size), DEVICE_ALLOC_ERR_MSG );
    CUDA_SAFE_CALL( cudaMalloc((void**) &sums_per_period_d, sums_per_period_size), DEVICE_ALLOC_ERR_MSG );

    set_end_time(&timer, DEVICE_ALLOC_TIME);
    print_elapsed_time(&timer, DEVICE_ALLOC_TIME, OPERATION_COMPLETED_MSG);

    
    /* --- --- --- --- --- ----------------------- --- --- --- --- --- */
    /* --- --- --- --- --- DATA TRANSFER TO DEVICE --- --- --- --- --- */

    print_entry_label(H2D_TRANSFER_START_MSG);
    set_start_time(&timer, H2D_TRANSFER_TIME);

    // Copy data from host to device
    CUDA_SAFE_CALL( cudaMemcpy( account_changes_d, 
                                account_changes_h,
                                account_changes_size, 
                                cudaMemcpyHostToDevice ), 
                    H2D_TRANSFER_ERR_MSG );

    // Synchronize device after data transfer
    CUDA_SAFE_CALL( cudaDeviceSynchronize(), SYNCHRONIZE_ERR_MSG );

    set_end_time(&timer, H2D_TRANSFER_TIME);
    print_elapsed_time(&timer, H2D_TRANSFER_TIME, OPERATION_COMPLETED_MSG);

    
    /* --- --- --- --- --- ----------------- --- --- --- --- --- */
    /* --- --- --- --- --- KERNELS LUANCHING --- --- --- --- --- */

    print_entry_label(KERNEL_1_EXEC_START_MSG);
    set_start_time(&timer, KERNEL_1_EXEC_TIME);

    // First launch account balance kernel and check for errors
    launch_account_balance_kernel(account_changes_d, account_balance_d, clients_num, periods_num);
    CUDA_SAFE_CALL( cudaGetLastError(), KERNEL_1_EXEC_ERR_MSG );

    // Synchronize device before launching next kernel
    CUDA_SAFE_CALL( cudaDeviceSynchronize(), SYNCHRONIZE_ERR_MSG );

    set_end_time(&timer, KERNEL_1_EXEC_TIME);
    print_elapsed_time(&timer, KERNEL_1_EXEC_TIME, OPERATION_COMPLETED_MSG);

    print_entry_label(KERNEL_2_EXEC_START_MSG);
    set_start_time(&timer, KERNEL_2_EXEC_TIME);

    // Second launch sums per period kernel and check for errors
    launch_sums_per_period_kernel(account_balance_d, sums_per_period_d, clients_num, periods_num);
    CUDA_SAFE_CALL( cudaGetLastError(), KERNEL_2_EXEC_ERR_MSG );

    // Synchronize device after launching kernels
    CUDA_SAFE_CALL( cudaDeviceSynchronize(), SYNCHRONIZE_ERR_MSG );

    set_end_time(&timer, KERNEL_2_EXEC_TIME);
    print_elapsed_time(&timer, KERNEL_2_EXEC_TIME, OPERATION_COMPLETED_MSG);

    
    /* --- --- --- --- --- --------------------- --- --- --- --- --- */
    /* --- --- --- --- --- DATA TRANSFER TO HOST --- --- --- --- --- */

    print_entry_label(D2H_TRANSFER_START_MSG);
    set_start_time(&timer, D2H_TRANSFER_TIME);

    // Copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( sums_per_period_h,
                                sums_per_period_d,
                                sums_per_period_size,
                                cudaMemcpyDeviceToHost ), 
                    D2H_TRANSFER_ERR_MSG );
    CUDA_SAFE_CALL( cudaMemcpy( account_balance_h, 
                                account_balance_d, 
                                account_balance_size, 
                                cudaMemcpyDeviceToHost ), 
                    D2H_TRANSFER_ERR_MSG );

    set_end_time(&timer, D2H_TRANSFER_TIME);
    print_elapsed_time(&timer, D2H_TRANSFER_TIME, OPERATION_COMPLETED_MSG);

    /* --- --- --- --- --- --------------- --- --- --- --- --- */
    /* --- --- --- --- --- OUTPUT RESULTS  --- --- --- --- --- */
    // TODO do not hardcode output names 
    // TODO maybe move after verification? But that broke the logic of benchmark...

    print_entry_label(CSV_SAVE_START_MSG);
    set_start_time(&timer, CSV_SAVE_TIME);

    save_csv("out_balance.csv", account_balance_h, periods_num, clients_num);
    save_csv("out_sums.csv", sums_per_period_h, 1, periods_num);
    
    set_end_time(&timer, CSV_SAVE_TIME);
    print_elapsed_time(&timer, CSV_SAVE_TIME, OPERATION_COMPLETED_MSG);

    /* --- --- --- --- --- -------------------- --- --- --- --- --- */
    /* --- --- --- --- --- RESULTS VERIFICATION --- --- --- --- --- */

    if (enable_verification) {

        print_entry_label(VERIFICATION_START_MSG);
        set_start_time(&timer, VERIFICATION_TIME);

        // Rsult verification with CPU computation
        verification_result = verify_results_with_CPU( account_changes_h, 
                                                       account_balance_h, 
                                                       sums_per_period_h, 
                                                       clients_num, 
                                                       periods_num );

        set_end_time(&timer, VERIFICATION_TIME);
        print_elapsed_time(&timer, VERIFICATION_TIME, OPERATION_COMPLETED_MSG);

        printf(verification_result
            ? VERIFICATION_SUCCESS_MSG
            : VERIFICATION_FAILURE_MSG);

    }
    
    /* --- --- --- --- --- --------------- --- --- --- --- --- */
    /* --- --- --- --- --- MEMORY CLEAN UP --- --- --- --- --- */

    // Free device memory
    if (account_changes_d)
        CUDA_SAFE_CALL( cudaFree(account_changes_d), DEVICE_FREE_ERR_MSG );
    if (account_balance_d)
        CUDA_SAFE_CALL( cudaFree(account_balance_d), DEVICE_FREE_ERR_MSG );
    if (sums_per_period_d)
        CUDA_SAFE_CALL( cudaFree(sums_per_period_d), DEVICE_FREE_ERR_MSG );
    
    // Free host memory
    free(account_changes_h);
    free(sums_per_period_h);
    free(account_balance_h);


    /* --- --- --- --- --- -------------------- --- --- --- --- --- */
    /* --- --- --- --- --- MAIN FUNCTION RETURN --- --- --- --- --- */

    print_total_time(&timer, TOTAL_EXECUTION_TIME_MSG);

    return EXIT_SUCCESS;
}

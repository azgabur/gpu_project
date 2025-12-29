#ifndef BENCHMARK_H
#define BENCHMARK_H


#include <sys/time.h>


// Time debug levels
#define TIME_DEBUG_LEVEL_NONE 0
#define TIME_DEBUG_LEVEL_SIMPLE 1
#define TIME_DEBUG_LEVEL_DETAILED 2


// Logging messages
#define PROBLEM_SETUP_START_MSG  "Problem setup started...            "
#define DEVICE_ALLOC_START_MSG   "Device memory allocation started... "
#define H2D_TRANSFER_START_MSG   "Data transfer to device started...  "
#define KERNEL_1_EXEC_START_MSG  "1st kernel execution started...     "
#define KERNEL_2_EXEC_START_MSG  "2nd kernel execution started...     "
#define D2H_TRANSFER_START_MSG   "Data transfer to host started...    "
#define VERIFICATION_START_MSG   "Results verification started...     "
#define CSV_SAVE_START_MSG       "Saving the results to disk...       "

#define OPERATION_COMPLETED_MSG  "completed in %f seconds.\n"
#define TOTAL_EXECUTION_TIME_MSG "Total execution time:               %f seconds.\n"


// Number of rounds
#define TEST_ROUNDS 6

// Timer entries indexes
#define PROBLEM_SETUP_TIME 0
#define DEVICE_ALLOC_TIME  1
#define H2D_TRANSFER_TIME  2
#define KERNEL_1_EXEC_TIME 3
#define KERNEL_2_EXEC_TIME 4
#define D2H_TRANSFER_TIME  5
#define CSV_SAVE_TIME      6
#define VERIFICATION_TIME  7

#define CPU_PERF_TIME_START 8
#define GPU_PERF_TIME_START CPU_PERF_TIME_START + TEST_ROUNDS
#define TIMER_ENTRIES_NUM  8 + TEST_ROUNDS * 2


// External variable for time debug level
extern int time_debug_level;

// External variable to enable/disable results verification
extern int enable_verification;

// Timer structure
typedef struct {
    struct timeval start_time[TIMER_ENTRIES_NUM];
    struct timeval end_time[TIMER_ENTRIES_NUM];
} Timer;

// set current time as start time of given entry
void set_start_time(Timer* timer, int time_entry);

// set current time as end time of given entry
void set_end_time(Timer* timer, int time_entry);

// calculate elapsed time in seconds of given entry
double get_elapsed_time(Timer *timer, int time_entry);

// calculate total time in seconds
double get_total_time(Timer *timer);

// print elapsed time of given entry with message based on debug level
void print_elapsed_time(Timer* timer, int entry, const char* msg);

// print total time with message based on debug level
void print_total_time(Timer* timer, const char* msg);

// print entry label based on debug level
void print_entry_label(const char* msg);

// print perofrmance information
void print_performance_results(Timer* timer, size_t input_bytes);

#endif // BENCHMARK_H

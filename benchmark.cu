#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "benchmark.h"


// External variable for time debug level (declared in benchmark.h)
int time_debug_level = TIME_DEBUG_LEVEL_NONE; // default value

// External variable to enable/disable results verification (declared in benchmark.h)
int enable_verification = 0; // by default disabled


void set_start_time(Timer* timer, int time_entry) {
    gettimeofday(&(timer->start_time[time_entry]), NULL);
}

void set_end_time(Timer* timer, int time_entry) {
    gettimeofday(&(timer->end_time[time_entry]), NULL);
}

double get_elapsed_time(Timer *timer, int time_entry) {
    struct timeval start = timer->start_time[time_entry];
    struct timeval end = timer->end_time[time_entry];

    double seconds = end.tv_sec - start.tv_sec;
    double microsec = end.tv_usec - start.tv_usec;
    
    return seconds + (microsec * 1.0e-6);
}

double get_total_time(Timer *timer) {
    double elapsed_time = 0;
    for (int i = 0; i < TIMER_ENTRIES_NUM; i++) {
        elapsed_time += get_elapsed_time(timer, i);
    }
    return elapsed_time;
}

// Internal function to print time based on debug level
void _print_time(double time, const char* msg) {
    if (time_debug_level == TIME_DEBUG_LEVEL_SIMPLE) {
        printf("%f\n", time);
    } else if (time_debug_level == TIME_DEBUG_LEVEL_DETAILED) {
        printf(msg, time);
    }
    // if time_debug_level == TIME_DOBUG_LEVEL_NONE, do nothing
}

void print_total_time(Timer* timer, const char* msg) {
    _print_time(get_total_time(timer), msg);
}

void print_elapsed_time(Timer* timer, int entry, const char* msg) {
    _print_time(get_elapsed_time(timer, entry), msg);
}

void print_entry_label(const char* msg) {
    if (time_debug_level == TIME_DEBUG_LEVEL_DETAILED) {
        printf(msg);
        fflush(stdout);
    }
    // if time_debug_level == TIME_DEBUG_LEVEL_SIMPLE -> do nothing
    // if time_debug_level == TIME_DOBUG_LEVEL_NONE -> do nothing
}

void print_performance_results(Timer* timer, size_t input_bytes) {
    double cpu_average = 0;
    double gpu_average = 0;
    double cpu_bytes_per_time = 0;
    double gpu_bytes_per_time = 0;

    for (int i = 0; i < TEST_ROUNDS; i++) {
        cpu_average += get_elapsed_time(timer, CPU_PERF_TIME_START + i);
        gpu_average += get_elapsed_time(timer, GPU_PERF_TIME_START + i);
    }
    cpu_average = cpu_average / TEST_ROUNDS;
    gpu_average = gpu_average / TEST_ROUNDS;

    cpu_bytes_per_time = input_bytes / cpu_average;
    gpu_bytes_per_time = input_bytes / gpu_average;

    printf("\n------ Performance ------ (over %d rounds and %lu megabytes)\n", TEST_ROUNDS, input_bytes / (1024*1024));
    printf("The average cpu computation time is: %fms\n", cpu_average * 1e3);
    printf("The megabytes per second calculation speed for cpu is: %f\n", cpu_bytes_per_time / (1024*1024));
    printf("The average GPU computation time is: %fms\n", gpu_average * 1e3);
    printf("The megabytes per second calculation speed for GPU is: %f\n", gpu_bytes_per_time / (1024*1024));
}

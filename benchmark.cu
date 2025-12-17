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
    int last_entry_idx = TIMER_ENTRIES_NUM - 1;
    // skip verification time if disabled
    if (!enable_verification) last_entry_idx -= 1;

    struct timeval start = timer->start_time[0];
    struct timeval end = timer->end_time[last_entry_idx];
    
    double seconds = end.tv_sec - start.tv_sec;
    double microsec = end.tv_usec - start.tv_usec;
    
    return seconds + (microsec * 1.0e-6);
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
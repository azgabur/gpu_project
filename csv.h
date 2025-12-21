#ifndef CSV_H
#define CSV_H
#include <stdlib.h>

#define TESTING_MATRIX_PATH "./testing.csv"

// Saves 2D array as CSV file to path.
// Returns 0 if successful, 1 otherwise
int save_csv(const char* path, int* matrix, int rows, int cols);

// Loads input array from path, allocates arrays, and sets their sizes
// Returns 0 if successful, 1 otherwise
int load_csv(
    const char* path,
    size_t* account_changes_size,
    size_t* account_balance_size,
    size_t* sums_per_period_size, 
    int** account_changes_h, 
    int** account_balance_h, 
    int** sums_per_period_h,
    int* clients_num,
    int* periods_num
);


#endif // CSV_H
#include <stdio.h>

#include "csv.h"

int save_csv(const char* path, int* matrix, int rows, int cols) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL){
        return 1;
    }
    
    // First line will store the size of the array
    fprintf(fp, "%d,%d\n", rows, cols);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int element = matrix[row * rows + col];
            bool last_element = col >= cols - 1;
            if (last_element) {
                fprintf(fp, "%d", element);
            }
            else {
                fprintf(fp, "%d,", element);
            }
        }
        fputs("\n", fp);
    }

    fclose(fp);
    return 0;
}

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
) {
    int err = 0;
    FILE *fp = fopen(path, "r");
    if (fp == NULL){
        return 1;
    }
    
    char *line = NULL;
    size_t line_size = 0;
    int input_rows = 0;
    int input_cols = 0;
    int current_row = 0;
    int current_cols = 0;

    // Get the dimensions from first line
    if (getline(&line, &line_size, fp) == -1){
        err = 1;
        goto cleanup;
    }
    
        
    if (sscanf(line, "%d,%d", &input_rows, &input_cols) != 2) {
        err = 1;
        goto cleanup;
    }

    // Set dimensions
    *clients_num = input_cols;
    *periods_num = input_rows;

    // Allocate arrays
    *account_changes_size = (size_t) input_rows * (size_t) input_cols * sizeof(int);
    *account_balance_size = (size_t) input_rows * (size_t) input_cols * sizeof(int);
    *sums_per_period_size = (size_t) input_rows * sizeof(int);
    *account_changes_h = NULL;
    *account_balance_h = NULL;
    *sums_per_period_h = NULL;
    *account_changes_h = (int *) malloc(*account_changes_size);
    *account_balance_h = (int *) malloc(*account_balance_size);
    *sums_per_period_h = (int *) malloc(*sums_per_period_size);

    if (*account_changes_h == NULL || *account_balance_h == NULL || *sums_per_period_h == NULL){
        err = 1;
        goto cleanup;
    }


    while (getline(&line, &line_size, fp) != -1) {
        current_cols = 0;
        char *saveptr;
        char *token = strtok_r(line, ",\n", &saveptr);

        while (token != NULL) {
            
            // transform token to int, save to account_changes_h
            int element = atoi(token);
            (*account_changes_h)[current_row * input_cols + current_cols] = element;
            current_cols++;
            token = strtok_r(NULL, ",\n", &saveptr);
        }
        if (input_cols != current_cols) {
            err = 1;
            goto cleanup;
        }
        current_row++;
    }

    if (input_rows != current_row) {
        err = 1;
        goto cleanup;
    }

    cleanup:
    if (err) {
        fprintf(stderr, "Error reading line %d from input file!\n", current_row);
        free(*account_changes_h);
        free(*account_balance_h);
        free(*sums_per_period_h);
    }
    free(line);
    fclose(fp);
    return err;
}

#ifndef CUDA_SUPPORT_H
#define CUDA_SUPPORT_H


// Macro wrap for CUDA api calls, ensures error checking
//  ret - return value of the CUDA call
//  msg - error message to be printed in case of failure
#define CUDA_SAFE_CALL(ret, msg) do { \
    if (ret != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s: %s (%s:%d)\n", cudaGetErrorString(ret), msg, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Error messages for the macro wrap
#define DEVICE_ALLOC_ERR_MSG  "Unable to allocate device memory\n"
#define DEVICE_FREE_ERR_MSG   "Unable to free device memory\n"
#define HOST_ALLOC_ERR_MSG    "Unable to allocate host memory\n"
#define H2D_TRANSFER_ERR_MSG  "Unable to copy data from host to device\n"
#define D2H_TRANSFER_ERR_MSG  "Unable to copy data from device to host\n"
#define KERNEL_1_EXEC_ERR_MSG "Unable to launch 1st kernel\n"
#define KERNEL_2_EXEC_ERR_MSG "Unable to launch 2nd kernel\n"
#define SYNCHRONIZE_ERR_MSG   "Unable to synchronize device\n"


#endif // CUDA_SUPPORT_H
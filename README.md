# Gpu accelerated savings account

CUDA parallel program which takes 2D input array of (account x period) saving account changes and outputs 2D array of accumulated balance from previous periods (account x period) and 1D array of final sum of all changes accross all accounts per period.
The 0 period is an initial state of saving accounts.

## Current stage of development

non-optimised implementation:

+ compilation - ok
+ porgram run - ok
+ verification - ok

usage:

- account_savings [-v] [-d level] [clients] [periods]
- program input: 
  - parameters: client and periods number
  - flag -v: enables verification (disabled by default)
  - flag -d x: set time debug level (0 - none, 1 - simple, 2 - detailed)
  - flag -h: pritns usage info

development stage details:
- safe cuda api calls (via wraps)
- safe dynamic memory management
- unified error messages
- non-optimised gpu kernels
- non-optimised methods for running kernels
- CPU solver and verifier
- generator of random array
- simplified Makefile


## System architecture

Program should be able to work in two modes. Production and testing. 
Production takes the input csv file and returns output csv files. 
Testing can create random input csv files and can run test framework while outputing performance data to stdout (maybe also structured file). Peformance data like time elapsed, number of operations of computation or loading of memory.

1 iteration:
- Input of csv file as input
  - Creating random input for testing (export random csv for testing and/or directly populating input array)
  - Should the input data be completely random? Only positive?
- Methods for allocating memeory on host machine and device (gpu) machine
- Methods for transfering memory from host machine to device machine and back
- Method for running kernel
- Unoptimalized but correct kernel
- ! Method for checking the result against the reference 
- Cleanup at the end of main

2 iteration:
- Some Kernel optimalizations
- Testing framework
  - Multiple runs, calculating average performance values and outputing
  - Keep checking the result against the reference
  - Have one large input csv generated which takes non-trivial time to complete, compute the correct output and in testing framwork check against the correct outputs instead of computing on CPU
  - Via #define macros add operations counting in kernel.cu
    - counting additions, global memory loads, shared memory loads (if applicable)

3 iteration:
- More optimalisations of kernel
...

## Additional outputs
- Performance report (file docs/performance.txt)
- Bottleneck analysis (file docs/bottleneck.txt
- What kernel optimalisations were used (file docs/optimalisations.txt)

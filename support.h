

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#define FATAL(...) \
    do {\
        fprintf(stderr, "[%s:%d] ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        exit(-1); \
    } while(0)


#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif

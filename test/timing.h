#ifndef _TIMING_H
#define _TIMING_H

#include <time.h>


struct timer {
    struct timespec start, end;
};

#define start_timing(ts) \
    __asm__ __volatile__("":::"memory"); \
    clock_gettime(CLOCK_MONOTONIC, &(ts)->start); \
    __asm__ __volatile__("":::"memory")

#define end_timing(ts) \
    __asm__ __volatile__("":::"memory"); \
    clock_gettime(CLOCK_MONOTONIC, &(ts)->end); \
    __asm__ __volatile__("":::"memory")



static double get_time(struct timer * ts) {
    return (ts->end.tv_sec - ts->start.tv_sec) +
        (ts->end.tv_nsec - ts->start.tv_nsec) / 1000000000.;
}


#endif /* _TIMING_H */

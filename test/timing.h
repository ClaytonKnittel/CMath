#ifndef _TIMING_H
#define _TIMING_H

#include <time.h>


struct timer {
    struct timespec start, end;
};

#define start_timing(ts) \
    clock_gettime(CLOCK_MONOTONIC, &(ts)->start)

#define end_timing(ts) \
    clock_gettime(CLOCK_MONOTONIC, &(ts)->end)



static double get_time(struct timer * ts) {
    return (ts->end.tv_sec - ts->start.tv_sec) +
        (ts->end.tv_nsec - ts->start.tv_nsec) / 1000000000.;
}


#endif /* _TIMING_H */


/*
 * test vector addition implementation
 */

#include <pmath/util.h>
#include <pmath/vec.h>

#include "timing.h"


static void __attribute__((noinline)) naive_add(float * dst, float * v1, float * v2, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = v1[i] + v2[i];
    }
}


int main(int argc, char *argv[]) {
    struct timer t;
#define L1 8
    float v1[L1] = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f
    };
    float v2[L1] = {
        7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f
    };

    float res[L1];

    start_timing(&t);
    for (int i = 0; i < 10000000; i++) {
        _fvec_add(res, v1, v2, L1);
    }
    end_timing(&t);
    printf("Took %f s\n", get_time(&t));

    start_timing(&t);
    for (int i = 0; i < 10000000; i++) {
        naive_add(res, v1, v2, L1);
    }
    end_timing(&t);
    printf("naive Took %f s\n", get_time(&t));

    MATH_ASSERT(res[0] == 8.f);
    MATH_ASSERT(res[1] == 8.f);
    MATH_ASSERT(res[2] == 8.f);
    MATH_ASSERT(res[3] == 8.f);
    MATH_ASSERT(res[4] == 8.f);
    MATH_ASSERT(res[5] == 8.f);
    MATH_ASSERT(res[6] == 8.f);
    MATH_ASSERT(res[7] == 8.f);

    return 0;
}



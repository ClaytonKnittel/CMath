
#include <assert.h>
#include <stdio.h>

#include <pmath/cl.h>
#include <pmath/vec.h>

#include "timing.h"
#include "util.h"


//#define DP


#ifdef DP
typedef double num_t;
#else
typedef float num_t;
#endif


int main(int argc, char *argv[]) {
    struct timer t;

#ifdef DP
    _load_vec_add();
#else
    _load_fvec_add();
#endif


#define N_SIZES 5

    const size_t sizes[N_SIZES] = {
        0x200000,
        0x400000,
        0x800000,
        0x1000000,
        0x2000000
    };

    size_t size;

    for (int i = 0; i < N_SIZES; i++) {
        size = sizes[i];

        num_t * a_del = (num_t*) malloc(64 + size * sizeof(num_t));
        num_t * b_del = (num_t*) malloc(64 + size * sizeof(num_t));
        num_t * c_del = (num_t*) malloc(64 + size * sizeof(num_t));

        assert(a_del != NULL && b_del != NULL && c_del != NULL);

        num_t * a = (num_t *) ((((uint64_t) a_del) + 63) & ~(64 - 1));
        num_t * b = (num_t *) ((((uint64_t) b_del) + 63) & ~(64 - 1));
        num_t * c = (num_t *) ((((uint64_t) c_del) + 63) & ~(64 - 1));

        for (int i = 0; i < size; i++) {
            a[i] = (i + 1) % 128;
            b[i] = (i + 2) % 128;
        }

        mem_flush(a, size * sizeof(num_t));
        mem_flush(b, size * sizeof(num_t));
        mem_flush(c, size * sizeof(num_t));
        printf("start\n");

        start_timing(&t);
#ifdef DP
        _vec_add(c, a, b, size);
#else
        _fvec_add(c, a, b, size);
#endif
        end_timing(&t);

        printf("%lu\tTotal time: %lf\n", size, get_time(&t));
        
        for (int i = 0; i < size; i++) {
            assert(c[i] == a[i] + b[i]);
        }

        free(c_del);
        free(b_del);
        free(a_del);
    }

    return 0;
}


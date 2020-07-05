
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

#include <pmath/cl.h>
#include <pmath/util.h>
#include <pmath/vec.h>


// number of packed single-precision floating point numbers per AVX register
#define PS_PER_REG 8
#define LOG_PS_PER_REG 3

#define FADD_BASE_LEN 64


/*
 * add double vectors v1 and v2 together, storing the result in dst
 */
void _vec_add(double * dst, double * restrict v1, double * restrict v2,
        size_t len) {
}

void _load_vec_add() {

}


/*
 * add remaining parts of vectors together
 */
static void _fvec_add_base_rem(float * dst, float * restrict v1,
        float * restrict v2, size_t len) {

    MATH_ASSERT(len < PS_PER_REG);

#if 0
    switch (len) {
        case 7:
            dst[6] = v1[6] + v2[6];
        case 6:
            dst[5] = v1[5] + v2[5];
        case 5:
            dst[4] = v1[4] + v2[4];
        case 4:
            dst[3] = v1[3] + v2[3];
        case 3:
            dst[2] = v1[2] + v2[2];
        case 2:
            dst[1] = v1[1] + v2[1];
        case 1:
            dst[0] = v1[0] + v2[0];
        case 0:;
    }
#else
    switch (len) {
        case 0:
            break;
        case 1:
            dst[0] = v1[0] + v2[0];
            break;
        case 2:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            break;
        case 3:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            dst[2] = v1[2] + v2[2];
            break;
        case 4:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            dst[2] = v1[2] + v2[2];
            dst[3] = v1[3] + v2[3];
            break;
        case 5:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            dst[2] = v1[2] + v2[2];
            dst[3] = v1[3] + v2[3];
            dst[4] = v1[4] + v2[4];
            break;
        case 6:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            dst[2] = v1[2] + v2[2];
            dst[3] = v1[3] + v2[3];
            dst[4] = v1[4] + v2[4];
            dst[5] = v1[5] + v2[5];
            break;
        case 7:
            dst[0] = v1[0] + v2[0];
            dst[1] = v1[1] + v2[1];
            dst[2] = v1[2] + v2[2];
            dst[3] = v1[3] + v2[3];
            dst[4] = v1[4] + v2[4];
            dst[5] = v1[5] + v2[5];
            dst[6] = v1[6] + v2[6];
            break;
    }
#endif
}


static void _fvec_add_base(float * dst, float * restrict v1,
        float * restrict v2, size_t len) {

    uint32_t i;
    uint8_t  remainder;

    MATH_ASSERT(len <= FADD_BASE_LEN);

    remainder = len & (PS_PER_REG - 1);

    for (i = 0; i < len; i += PS_PER_REG) {
        __m256d m1 = *(__m256d *) &v1[i];
        __m256d m2 = *(__m256d *) &v2[i];
        __m256d *d =  (__m256d *) &dst[i];
        *d = _mm256_add_ps(m1, m2);
    }

    _fvec_add_base_rem(&dst[i], &v1[i], &v2[i], remainder);
}



/*
 * add float vectors v1 and v2 together, storing the result in dst
 */
void _fvec_add(float * dst, float * restrict v1, float * restrict v2,
        size_t len) {

    if (len <= FADD_BASE_LEN) {
        _fvec_add_base(dst, v1, v2, len);
        return;
    }

    abort();
}



void _load_fvec_add() {
    cl_load_op(fvec_add, "kernels/fvec_add.cl", "fvec_add");
}



#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

#include <pmath/cl.h>
#include <pmath/util.h>
#include <pmath/vec.h>


// number of packed double-precision floating point numbers per AVX register
#define PD_PER_REG 4
#define LOG_PD_PER_REG 2
// number of packed single-precision floating point numbers per AVX register
#define PS_PER_REG 8
#define LOG_PS_PER_REG 3

#define ADD_BASE_LEN 0x10000
#define FADD_BASE_LEN 0x58000


/*
 * add remaining parts of vectors together
 */
static void _vec_add_base_rem(double * dst, double * restrict v1,
        double * restrict v2, size_t len) {

    MATH_ASSERT(len < PD_PER_REG);

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
    }
}


static void _vec_add_base(double * dst, double * restrict v1,
        double * restrict v2, size_t len) {

    uint64_t i;
    uint8_t  remainder;

    MATH_ASSERT(len <= ADD_BASE_LEN);

    remainder = len & (PD_PER_REG - 1);

    for (i = 0; i < len; i += PD_PER_REG) {
        __m256d m1 = *(__m256d *) &v1[i];
        __m256d m2 = *(__m256d *) &v2[i];
        __m256d *d =  (__m256d *) &dst[i];
        *d = _mm256_add_pd(m1, m2);
    }

    _vec_add_base_rem(&dst[i], &v1[i], &v2[i], remainder);
}




void _vec_add_gpu(cl_mem dst, cl_mem v1, cl_mem v2, size_t len) {
    cl_set_param(vec_add, 0, sizeof(dst), &dst);
    cl_set_param(vec_add, 1, sizeof(v1),  &v1);
    cl_set_param(vec_add, 2, sizeof(v2),  &v2);

    cl_execute_op(vec_add, len / 16);
}



static void __attribute__((always_inline)) __vec_add_gpu(double * dst,
        double * restrict v1, double * restrict v2, size_t len) {

    cl_mem v1_cl, v2_cl, dst_cl;

    v1_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(double), v1);
    v2_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(double), v2);

    dst_cl = cl_create_buffer(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(double), dst);

    _vec_add_gpu(dst_cl, v1_cl, v2_cl, len);

    cl_finish();

    cl_delete_buffer(dst_cl);
    cl_delete_buffer(v2_cl);
    cl_delete_buffer(v1_cl);
}





/*
 * add double vectors v1 and v2 together, storing the result in dst
 */
void _vec_add(double * dst, double * restrict v1, double * restrict v2,
        size_t len) {

    if (len <= ADD_BASE_LEN) {
        _vec_add_base(dst, v1, v2, len);
        return;
    }

    __vec_add_gpu(dst, v1, v2, len);
}

void _load_vec_add() {
    cl_load_op(vec_add, "kernels/vec_add.cl", "vec_add", NULL);
}


/*
 * add remaining parts of vectors together
 */
static void _fvec_add_base_rem(float * dst, float * restrict v1,
        float * restrict v2, size_t len) {

    MATH_ASSERT(len < PS_PER_REG);

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
}


static void _fvec_add_base(float * dst, float * restrict v1,
        float * restrict v2, size_t len) {

    uint64_t i;
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




void _fvec_add_gpu(cl_mem dst, cl_mem v1, cl_mem v2, size_t len) {
    cl_set_param(fvec_add, 0, sizeof(dst), &dst);
    cl_set_param(fvec_add, 1, sizeof(v1),  &v1);
    cl_set_param(fvec_add, 2, sizeof(v2),  &v2);

    cl_execute_op(fvec_add, len);
}



static void __attribute__((always_inline)) __fvec_add_gpu(float * dst,
        float * restrict v1, float * restrict v2, size_t len) {

    cl_mem v1_cl, v2_cl, dst_cl;

    v1_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(float), v1);
    v2_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(float), v2);

    dst_cl = cl_create_buffer(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            len * sizeof(float), dst);

    _fvec_add_gpu(dst_cl, v1_cl, v2_cl, len);

    cl_finish();

    cl_delete_buffer(dst_cl);
    cl_delete_buffer(v2_cl);
    cl_delete_buffer(v1_cl);
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

    __fvec_add_gpu(dst, v1, v2, len);
}



void _load_fvec_add() {
    cl_load_op(fvec_add, "kernels/fvec_add.cl", "fvec_add",
            "-cl-std=CL1.2 -Werror -cl-no-signed-zeros "
            "-cl-uniform-work-group-size");
}


#ifndef _PMATH_MAT_H
#define _PMATH_MAT_H


#include <pmath/cl.h>


void _fmat_add_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t h, size_t w);


void _fmat_add(float * dst, float * restrict m1, float * restrict m2,
        size_t h, size_t w);


void _fmat_sub_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t h, size_t w);


void _fmat_sub(float * dst, float * restrict m1, float * restrict m2,
        size_t h, size_t w);



void _fmat_mul_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t m1_h, size_t m1_w,
        size_t m2_w);

void _fmat_mul(float * dst, float * m1, float * m2, size_t m1_h, size_t m1_w,
        size_t m2_w);

void _load_fmat_ops();


#endif /* _PMATH_MAT_H */

#ifndef _PMATH_VEC_H
#define _PMATH_VEC_H

#include <stddef.h>

#include <pmath/cl.h>


/*
 * call the vec_add kernel with given arguments
 */
void _vec_add_gpu(cl_mem dst, cl_mem v1, cl_mem v2, size_t len);

/*
 * add double vectors v1 and v2 together, storing the result in dst
 */
void _vec_add(double * dst, double * restrict v1, double * restrict v2,
        size_t len);

void _load_vec_add();


/*
 * call the fvec_add kernel with given arguments
 */
void _fvec_add_gpu(cl_mem dst, cl_mem v1, cl_mem v2, size_t len);


/*
 * add float vectors v1 and v2 together, storing the result in dst
 */
void _fvec_add(float * dst, float * restrict v1, float * restrict v2,
        size_t len);

void _load_fvec_add();


#endif /* _PMATH_VEC_H */

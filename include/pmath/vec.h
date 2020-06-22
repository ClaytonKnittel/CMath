#ifndef _PMATH_VEC_H
#define _PMATH_VEC_H

#include <stddef.h>


/*
 * add double vectors v1 and v2 together, storing the result in dst
 */
void _vec_add(double * dst, double * restrict v1, double * restrict v2,
        size_t len);


/*
 * add float vectors v1 and v2 together, storing the result in dst
 */
void _fvec_add(float * dst, float * restrict v1, float * restrict v2,
        size_t len);


#endif /* _PMATH_VEC_H */


#include <pmath/mat.h>
#include <pmath/vec.h>



void _fmat_add_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t h, size_t w) {
    _fvec_add_gpu(dst, m1, m2, h * w);
}


void _fmat_add(float * dst, float * restrict m1, float * restrict m2,
        size_t h, size_t w) {
    _fvec_add(dst, m1, m2, h * w);
}

void _load_fmat_add() {
    // fmat_add is just an alias for fvec add, since they work the exact same way
    _load_fvec_add();
}



void _fmat_mul_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t m1_h, size_t m1_w,
        size_t m2_w) {

    uint32_t _m1_w = m1_w;

    cl_set_param(fmat_mul, 0, sizeof(dst), &dst);
    cl_set_param(fmat_mul, 1, sizeof(m1), &m1);
    cl_set_param(fmat_mul, 2, sizeof(m2), &m2);
    cl_set_param(fmat_mul, 3, sizeof(_m1_w), &_m1_w);

    size_t g_sizes[2] = {
        m1_h,
        m2_w / 16
    };

    size_t l_sizes[2] = {
        64, 1
    };

    cl_execute_op(fmat_mul, 2, g_sizes, l_sizes);
}


void _fmat_mul(float * dst, float * m1, float * m2, size_t m1_h, size_t m1_w,
        size_t m2_w) {

    cl_mem m1_cl, m2_cl, dst_cl;

    m1_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            m1_h * m1_w * sizeof(float), m1);
    m2_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            m1_w * m2_w * sizeof(float), m2);

    dst_cl = cl_create_buffer(CL_MEM_WRITE_ONLY, m1_h * m2_w * sizeof(float), NULL);

    _fmat_mul_gpu(dst_cl, m1_cl, m2_cl, m1_h, m1_w, m2_w);

    cl_read_buffer(dst_cl, 0, m1_h * m2_w * sizeof(float), dst);

    cl_delete_buffer(dst_cl);
    cl_delete_buffer(m2_cl);
    cl_delete_buffer(m1_cl);
}


void _load_fmat_mul() {
    cl_load_op(fmat_mul, "kernels/fmat_mul.cl", "fmat_mul",
            "-cl-std=CL1.2 -Werror -cl-no-signed-zeros");
}



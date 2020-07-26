
#include <pmath/mat.h>
#include <pmath/util.h>
#include <pmath/vec.h>



#define ADD_DESIRED_KERNEL_W 64



#define MMUL_BLOCK_H 32
#define MMUL_BLOCK_W 16



void _fmat_add_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t h, size_t w) {

    size_t n_els = h * w;
    cl_uint width = ADD_DESIRED_KERNEL_W / 16;

    MATH_ASSERT((n_els & 0xff) == 0);

    cl_set_param(fmat_add, 0, sizeof(dst), &dst);
    cl_set_param(fmat_add, 1, sizeof(m1), &m1);
    cl_set_param(fmat_add, 2, sizeof(m2), &m2);
    //cl_set_param(fmat_add, 3, sizeof(width), &width);

    size_t g_sizes[1] = {
        n_els / 16 / width,
    };

    size_t l_sizes[1] = {
        16
    };

    cl_execute_op(fmat_add, 1, g_sizes, l_sizes);
}


void _fmat_add(float * dst, float * restrict m1, float * restrict m2,
        size_t h, size_t w) {

    cl_mem m1_cl, m2_cl, dst_cl;

    m1_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            h * w * sizeof(float), m1);
    m2_cl = cl_create_buffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            h * w * sizeof(float), m2);

    dst_cl = cl_create_buffer(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            h * w * sizeof(float), dst);

    _fmat_add_gpu(dst_cl, m1_cl, m2_cl, h, w);

    cl_finish();

    cl_delete_buffer(dst_cl);
    cl_delete_buffer(m2_cl);
    cl_delete_buffer(m1_cl);
}



void _fmat_mul_gpu(cl_mem dst, cl_mem m1, cl_mem m2, size_t m1_h, size_t m1_w,
        size_t m2_w) {

    MATH_ASSERT((m1_h & (MMUL_BLOCK_H - 1)) == 0);
    MATH_ASSERT((m1_w & (MMUL_BLOCK_W - 1)) == 0);

    uint32_t _m1_w = m1_w;

    cl_set_param(fmat_mul, 0, sizeof(dst), &dst);
    cl_set_param(fmat_mul, 1, sizeof(m1), &m1);
    cl_set_param(fmat_mul, 2, sizeof(m2), &m2);
    cl_set_param(fmat_mul, 3, sizeof(_m1_w), &_m1_w);

    size_t g_sizes[2] = {
        m1_h,
        m2_w / MMUL_BLOCK_W
    };

    size_t l_sizes[2] = {
        MMUL_BLOCK_H, 1
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


void _load_fmat_ops() {
    cl_load_op(fmat_mul, "kernels/fmat_mul.cl", "fmat_mul",
            "-cl-std=CL1.2 -Werror -cl-no-signed-zeros "
            "-D GROUP_HEIGHT=" STR(MMUL_BLOCK_H) " "
            "-D GROUP_STRIDE=" STR(MMUL_BLOCK_W));

    cl_load_op(fmat_add, "kernels/mat/add.cl", "add",
            "-cl-std=CL1.2 -Werror -cl-no-signed-zeros");

    cl_load_op(fmat_sub, "kernels/mat/sub.cl", "sub",
            "-cl-std=CL1.2 -Werror -cl-no-signed-zeros");
}



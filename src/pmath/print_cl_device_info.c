
#include <stdio.h>

#include <OpenCL/opencl.h>

#include <pmath/cl.h>

void cl_print_device_info(cl_device_id id) {
    char name[128];
    char vendor[128];
    char version[128];
    char d_version[128];
    char c_version[128];

    cl_uint compute_units;
    cl_uint max_work_item_dims;
    size_t* max_work_item_sizes;
    size_t max_work_group_size;

    cl_uint max_clock_freq;

    cl_uint addr_bits;
    cl_ulong mem_alloc_size;

    cl_device_mem_cache_type cache_type;
    cl_uint cacheline_size;
    cl_ulong cache_size;
    cl_ulong glob_mem_size;

    cl_device_local_mem_type loc_mem;
    cl_ulong loc_mem_size;

    cl_bool endian;

    cl_device_exec_capabilities exec_cap;

    cl_device_fp_config sp, dp;

    char builtin_kernels[128];

    clGetDeviceInfo(id, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(id, CL_DEVICE_VENDOR, 128, vendor, NULL);
    clGetDeviceInfo(id, CL_DRIVER_VERSION, 128, version, NULL);
    clGetDeviceInfo(id, CL_DEVICE_VERSION, 128, d_version, NULL);
    clGetDeviceInfo(id, CL_DEVICE_OPENCL_C_VERSION, 128, c_version, NULL);
    printf("%s : %s (%s)\n", vendor, name, version);
    printf("\tdevice version: %s\n\tc version: %s\n", d_version, c_version);

    clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, 128, &compute_units, NULL);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 128, &max_work_item_dims, NULL);
    max_work_item_sizes = (size_t *) malloc(max_work_item_dims * sizeof(size_t));
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 128, max_work_item_sizes, NULL);
    clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, 128, &max_work_group_size, NULL);

    printf("\tcompute units: %u\n\tmax work group size: %lu\n\twork item dims: (",
            compute_units, max_work_group_size);
    for (cl_uint i = 0; i < max_work_item_dims; i++) {
        printf("%lu", max_work_item_sizes[i]);
        if (i != max_work_item_dims - 1) {
            printf(", ");
        }
    }
    printf(")\n");

    free(max_work_item_sizes);

    clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, 128, &max_clock_freq, NULL);
    printf("\tmax clock freq: %u MHz\n", max_clock_freq);

    clGetDeviceInfo(id, CL_DEVICE_ADDRESS_BITS, 128, &addr_bits, NULL);
    clGetDeviceInfo(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 128, &mem_alloc_size, NULL);

    printf("\taddress bits: %u\n\tmax mem alloc size: %llu\n", addr_bits, mem_alloc_size);



    clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, 128, &cache_type, NULL);
    switch (cache_type) {
        case CL_NONE:
            printf("\tno cache\n");
            break;
        case CL_READ_ONLY_CACHE:
            printf("\tread only cache\n");
            break;
        case CL_READ_WRITE_CACHE:
            printf("\tread write cache\n");
            break;
    }
    if (cache_type != CL_NONE) {
        clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, 128, &cacheline_size, NULL);
        clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, 128, &cache_size, NULL);
        printf("\tcache size: %llu (line %u)\n", cache_size, cacheline_size);
    }
    clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, 128, &glob_mem_size, NULL);
    printf("\tglobal memory: %llu\n", glob_mem_size);

    clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_TYPE, 128, &loc_mem, NULL);
    clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, 128, &loc_mem_size, NULL);

    printf("\tlocal memory: ");
    if (loc_mem == CL_LOCAL) {
        printf("local, size %llu\n", loc_mem_size);
    }
    else if (loc_mem == CL_GLOBAL) {
        printf("global, size %llu\n", loc_mem_size);
    }
    else {
        printf("none\n");
    }

    clGetDeviceInfo(id, CL_DEVICE_ENDIAN_LITTLE, 128, &endian, NULL);
    if (endian) {
        printf("\tlittle endian\n");
    }
    else {
        printf("\tbig endian\n");
    }

    clGetDeviceInfo(id, CL_DEVICE_EXECUTION_CAPABILITIES, 128, &exec_cap, NULL);
    switch (exec_cap) {
        case CL_EXEC_KERNEL:
            printf("\tcan execute OpenCL kernels\n");
            break;
        case CL_EXEC_NATIVE_KERNEL:
            printf("\tcan execute OpenCL native kernels\n");
            break;
    }

    clGetDeviceInfo(id, CL_DEVICE_BUILT_IN_KERNELS, 128, &builtin_kernels, NULL);
    printf("\tbuiltin kernels: %s\n", builtin_kernels);


    clGetDeviceInfo(id, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(sp), &sp, NULL);
    printf("\tsingle precision capabilities:\n");
    if (sp & CL_FP_DENORM) {
        printf("\t\tdenormalized\n");
    }
    if (sp & CL_FP_INF_NAN) {
        printf("\t\tINF and quiet NaNs\n");
    }
    if (sp & CL_FP_ROUND_TO_NEAREST) {
        printf("\t\tround to nearest even\n");
    }
    if (sp & CL_FP_ROUND_TO_ZERO) {
        printf("\t\tround to zero\n");
    }
    if (sp & CL_FP_ROUND_TO_INF) {
        printf("\t\tround to infinity\n");
    }
    if (sp & CL_FP_FMA) {
        printf("\t\tIEEE754-2008 FMA\n");
    }
    if (sp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
        printf("\t\tdivide and sqrt are correctly rounded\n");
    }
    if (sp & CL_FP_SOFT_FLOAT) {
        printf("\t\tbasic floating ops implemented in software\n");
    }


    clGetDeviceInfo(id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(dp), &dp, NULL);
    printf("\tdouble precision capabilities:\n");
    if (dp & CL_FP_DENORM) {
        printf("\t\tdenormalized\n");
    }
    if (dp & CL_FP_INF_NAN) {
        printf("\t\tINF and quiet NaNs\n");
    }
    if (dp & CL_FP_ROUND_TO_NEAREST) {
        printf("\t\tround to nearest even\n");
    }
    if (dp & CL_FP_ROUND_TO_ZERO) {
        printf("\t\tround to zero\n");
    }
    if (dp & CL_FP_ROUND_TO_INF) {
        printf("\t\tround to infinity\n");
    }
    if (dp & CL_FP_FMA) {
        printf("\t\tIEEE754-2008 FMA\n");
    }
    if (dp & CL_FP_SOFT_FLOAT) {
        printf("\t\tbasic floating ops implemented in software\n");
    }

}


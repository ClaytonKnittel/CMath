
#include <stdio.h>

#include <OpenCL/opencl.h>


#include <pmath/cl.h>


// universal flag
#define INITIALIZED 0x1


struct __int_op {
    cl_program prog;
    cl_kernel kernel;
    int flags;
};


struct __int_cl_context {
    cl_context context;

    cl_uint num_devices;
    cl_device_id * device_list;

    cl_command_queue command_queue;

    int flags;

    struct __int_op ops[n_operations];
};


/*
 * default global context, to be initialized once by the first API call
 */
struct __int_cl_context __global_context = {
    .flags = 0
};





static void print_device_info(cl_device_id id) {
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

    cl_bool endian;

    cl_device_exec_capabilities exec_cap;

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

}




/*
 * initializes given cl_context, called automatically by all of the below
 * functions if the global context has not yet been initialized
 */
static int _cl_init(struct __int_cl_context * c) {

    cl_uint num_platforms;
    cl_platform_id * platforms = NULL;

    //Set up the platform
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms =
        (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the device you want to run on
    cl_uint num_devices;
    cl_device_id * device_list = NULL;

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
    device_list = (cl_device_id *)
        malloc(sizeof(cl_device_id)*num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices,
            device_list, NULL);

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platforms[0],
        0
    };
    context = clCreateContext(props, num_devices, device_list, NULL, NULL, &err);

    if (context == NULL) {
        fprintf(stderr, "Unable to initialize OpenCL context, ");
        switch (err) {
            case CL_INVALID_PLATFORM:
                fprintf(stderr, "platform value is invalid\n");
                break;
            case CL_INVALID_VALUE:
                fprintf(stderr, "no devices could be found\n");
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                fprintf(stderr, "device not available\n");
                break;
            case CL_OUT_OF_HOST_MEMORY:
                fprintf(stderr, "the host was unable to allocate OpenCL "
                        "resources\n");
                break;
        }
        free(device_list);
        return -1;
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "failed to create command queue");
        clReleaseContext(context);
        free(device_list);
        return -1;
    }

    c->context = context;
    c->num_devices = num_devices;
    c->device_list = device_list;
    c->command_queue = command_queue;
    c->flags = INITIALIZED;
    __builtin_memset(c->ops, 0, n_operations * sizeof(struct __int_op));

    for (int i = 0; i < num_devices; i++) {
        print_device_info(device_list[i]);
    }

    return 0;
}


struct __int_cl_context * __cl_get_global_context() {
    struct __int_cl_context * gc = &__global_context;

    if (__builtin_expect(!(gc->flags & INITIALIZED), 0)) {
        _cl_init(gc);
    }

    return gc;
}



int __int_cl_load_op(operation_t op_idx, const char * program_name,
        const char * kernel_name, struct __int_cl_context * ctxt) {

    struct __int_op * op = &ctxt->ops[op_idx];

    if (__builtin_expect(!(op->flags & INITIALIZED), 0)) {
        // TODO cb & set initialized flag
        int err = 0;

        op->prog = clCreateProgramWithSource(ctxt->context, 1, &program_name,
                NULL, &err);

        if (err) {
            fprintf(stderr, "Error loading program \"%s\"\n", program_name);
            return err;
        }

        err = clBuildProgram(op->prog, 1, ctxt->device_list, NULL, NULL, NULL);

        if (err != CL_SUCCESS) {
            char buf[4096];
            size_t len;

            clGetProgramBuildInfo(op->prog, ctxt->device_list[0],
                    CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &len);

            fprintf(stderr, "Error building program \"%s\", reason: %s\n",
                    program_name, buf);
            return err;
        }

        op->kernel = clCreateKernel(op->prog, kernel_name, &err);

        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create kernel \"%s\" in program "
                    "\"%s\"\n", kernel_name, program_name);
            clReleaseProgram(op->prog);
            return err;
        }

        op->flags |= INITIALIZED;
    }

    // if op was already initialized, don't need to do anything
    return 0;
}


void cl_set_param(operation_t op_idx, uint32_t param_idx, size_t arg_size,
        const void * arg) {
    struct __int_cl_context * global_context;
    struct __int_op * op;

    global_context = __cl_get_global_context();
    op = &global_context->ops[op_idx];

    clSetKernelArg(op->kernel, param_idx, arg_size, arg);
}


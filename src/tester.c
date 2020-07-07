
#include <OpenCL/opencl.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include <tester.h>

#define SIZE 67108864


//OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    float a = A[index];                    \n"
"    float b = B[index];                    \n"
"    float res = alpha * a + b * b;         \n"
"    float res2 = res * res + alpha * a * b;\n"
"    C[index] = alpha * res2 + res * res;   \n"
"}                                          \n";



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

int test() {

    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)
        malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint           num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
    device_list = (cl_device_id *)
        malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    printf("Num devices: %d\n", num_devices);

    for (int i = 0; i < num_devices; i++) {
        print_device_info(device_list[i]);
    }

    // Create one OpenCL context for each device in the platform
    cl_context context;
    cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platforms[0],
        0
    };
    context = clCreateContext( props, num_devices, device_list, NULL, NULL, &clStatus);

    if (context == NULL) {
        fprintf(stderr, "Unable to initialize OpenCL context, ");
        switch (clStatus) {
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
                fprintf(stderr, "the host was unable to allocate OpenCL resources\n");
                break;
        }
        return -1;
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    float alpha = 2.f;
    float * a = (float*) malloc(SIZE * sizeof(float));
    float * b = (float*) malloc(SIZE * sizeof(float));
    float * c = (float*) malloc(SIZE * sizeof(float));

    // Create memory buffers on the device for each vector
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  SIZE * sizeof(float), a, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  SIZE * sizeof(float), b, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &clStatus);

    for (int i = 0; i < SIZE; i++) {
        a[i] = fmod(i, 127.f);
        b[i] = fmod(i * 2, 133.f);
    }

    // Copy the Buffer A and B to the device
    //clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, SIZE * sizeof(float), a, 0, NULL, NULL);
    //clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, SIZE * sizeof(float), b, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    if (clStatus != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");

        char buf[4096];
        size_t len;

        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &len);

        fprintf(stderr, "%s\n", buf);
        return -1;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    if (clStatus != CL_SUCCESS) {
        fprintf(stderr, "Failed to create kernel\n");
        return -1;
    }

    // Set the arguments of the kernel
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

    // Execute the OpenCL kernel on the list
    size_t global_size = SIZE; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    struct timespec start, end;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL /*&local_size*/, 0, NULL, NULL);

    if (clStatus != CL_SUCCESS) {
        fprintf(stderr, "failed to enqueue kernel: ");
        switch (clStatus) {
            case CL_INVALID_PROGRAM_EXECUTABLE:
                fprintf(stderr, "no executable has been built in the program object for the device associated with the command queue\n");
                break;
            case CL_INVALID_COMMAND_QUEUE:
                fprintf(stderr, "the command queue is not valid\n");
                break;
            case CL_INVALID_KERNEL:
                fprintf(stderr, "the kernel object is not valid\n");
                break;
            case CL_INVALID_CONTEXT:
                fprintf(stderr, "the command queue and kernel are not associated with the same context\n");
                break;
            case CL_INVALID_KERNEL_ARGS:
                fprintf(stderr, "kernel arguments have not been set\n");
                break;
            case CL_INVALID_WORK_DIMENSION:
                fprintf(stderr, "the dimension is not between 1 and 3\n");
                break;
            case CL_INVALID_GLOBAL_WORK_SIZE:
                fprintf(stderr, "the global work size is NULL or exceeds the range supported by the compute device\n");
                break;
            case CL_INVALID_WORK_GROUP_SIZE:
                fprintf(stderr, "the local work size is not evenly divisible with the global work size or the value specified exceeds "
                        "the range supported by the compute device\n");
                break;
            case CL_INVALID_WORK_ITEM_SIZE:
                fprintf(stderr, "the number of work-items specified in any of local_work_size[0], ... local_work_size[work_dim - 1] "
                        "is greater than the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ..., "
                        "CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1]\n");
                break;
            case CL_INVALID_GLOBAL_OFFSET:
                fprintf(stderr, "the reserved global offset parameter is not set to NULL");
                break;
            case CL_INVALID_EVENT_WAIT_LIST:
                fprintf(stderr, "the events list is empty (NULL) but the number of events is greater than 0; or number of events is 0 "
                        "but the event list is not NULL; or the events list contains invalid event objects\n");
                break;
            case CL_OUT_OF_HOST_MEMORY:
                fprintf(stderr, "the host is unable to allocate OpenCL resources\n");
                break;
            case CL_OUT_OF_RESOURCES:
                fprintf(stderr, "insufficient resources to execute the kernel\n");
                break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                fprintf(stderr, "there was a failure to allocate memory for data store associated with image or buffer objects "
                        "specified as arguments to the kernel\n");
                break;
        }
        return -1;
    }

    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_FALSE, 0, SIZE * sizeof(float), c, 0, NULL, NULL);

    clock_gettime(CLOCK_MONOTONIC, &start);
    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    clock_gettime(CLOCK_MONOTONIC, &end);

    float time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.f;
    fprintf(stderr, "GPU time: %f\n", time);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < SIZE; i++) {
        float res = alpha * a[i] + b[i] * b[i];
        float res2 = res * res + alpha * a[i] * b[i];
        float diff = (alpha * res2 + res * res) - c[i];
        assert(diff > -1e-5 && diff < 1e-5);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1000000000.f;
    fprintf(stderr, "CPU time: %f\n", time);

    // Display the result to the screen
    //for(int i = 0; i < SIZE; i++)
    //    printf("%f * %f + %f = %f\n", alpha, a[i], b[i], c[i]);

    // Finally release all OpenCL allocated objects and host buffers.
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(a);
    free(b);
    free(c);
    free(platforms);
    free(device_list);


    return 0;
}


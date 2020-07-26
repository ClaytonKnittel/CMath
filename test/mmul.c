
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <pmath/cl.h>
#include <pmath/mat.h>
#include <pmath/print_colors.h>

#include "timing.h"
#include "util.h"


static void fmat_mul_test(float * dst, float * m1, float * m2, size_t m1_h,
        size_t m1_w, size_t m2_w) {

    for (size_t i = 0; i < m1_h; i++) {
        for (size_t j = 0; j < m2_w; j++) {
            dst[i * m2_w + j] = 0;
            for (int k = 0; k < m1_w; k++) {
                dst[i * m2_w + j] += m1[i * m1_w + k] * m2[k * m2_w + j];
            }
        }
    }
}


static void sub(float * dst, float * m1, float * m2, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = m1[i] - m2[i];
    }
}

#define BLOCK_SIZE 16
static void fast_fmat_mul_test(float * dst, float * m1, float * m2, size_t m1_h,
        size_t m1_w, size_t m2_w) {

    for (size_t _i = 0; _i < m1_h; _i += BLOCK_SIZE) {
        for (size_t _j = 0; _j < m2_w; _j += BLOCK_SIZE) {
            for (size_t i = _i; i < _i + BLOCK_SIZE; i++) {
                for (size_t j = _j; j < _j + BLOCK_SIZE; j++) {
                    dst[i * m2_w + j] = 0;
                }
            }
            for (size_t _k = 0; _k < m1_w; _k += BLOCK_SIZE) {
                for (size_t i = _i; i < _i + BLOCK_SIZE; i++) {
                    for (size_t j = _j; j < _j + BLOCK_SIZE; j++) {
                        for (int k = _k; k < _k + BLOCK_SIZE; k++) {
                            dst[i * m2_w + j] += m1[i * m1_w + k] * m2[k * m2_w + j];
                        }
                    }
                }
            }
        }
    }
}


int run_subp(const unsigned char * buf, size_t buf_len) {
    FILE * file;
    file = fopen("test.krn", "w");

    fwrite(buf, 1, buf_len, file);

    fclose(file);

    pid_t child;

    if ((child = fork()) == 0) {
        dup(STDOUT_FILENO);
        dup(STDERR_FILENO);

        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
        fclose(stdout);
        fclose(stderr);

        // child
        char * const args[] = {
            "-d",
            "test.krn",
            "-p=9",
            NULL
        };
        execve("profilers/iga64", args, NULL);

        printf("error: %s\n", strerror(errno));

        exit(-2);
    }
    else {
        int ret_stat;
        waitpid(child, &ret_stat, 0);

        if (WIFSIGNALED(ret_stat)) {
            return -1;
        }

        return WEXITSTATUS(ret_stat);
    }

}



int main(int argc, char *argv[]) {
    struct timer t;
    int print = 0, check = 0;
    int opt;
    char * buf;

    ssize_t m1_h = -1, m1_w = -1, m2_w = -1;

    while ((opt = getopt(argc, argv, "cpi:j:k:n:")) != -1) {
        switch (opt) {
            case 'c':
                check = 1;
                break;
            case 'i':
                m1_h = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                break;
            case 'j':
                m2_w = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                break;
            case 'k':
                m1_w = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                break;
            case 'n':
                m1_h = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                m1_w = m1_h;
                m2_w = m1_h;
                break;
            case 'p':
                print = 1;
                break;
            case '?':
            default:
                return -1;
        }
    }

    if (m1_h == -1 || m1_w == -1 || m2_w == -1) {
        fprintf(stderr, "All three dimensions (i, j, k) must be specified\n"
                "The multiplication is: A (i x k) * B (k * j) = C (i * j)\n");
        return -1;
    }

    _load_fmat_ops();

    /*size_t siz;
    unsigned char buf2[16678];

    cl_get_op_binary(fmat_mul, buf2, sizeof(buf2), &siz);

    FILE * f = fopen("test2.krn", "w");

    fwrite(buf2, 1, siz, f);

    fclose(f);*/

    /*

#define START 0
    for (int i = 0; i < sizeof(buf2); i++) {
        if (i % 100 == 0) {
            printf("i = %d\n", i);
        }
        int ret = run_subp(buf2 + i, siz - i);
        if (ret == 0) {
            printf("worked on %d (%d)\n", i, ret);
        }
    }*/

    //return 0;

    float * a = (float *) malloc(m1_h * m1_w * sizeof(float));
    float * b = (float *) malloc(m1_w * m2_w * sizeof(float));
    float * c = (float *) malloc(m1_h * m2_w * sizeof(float));
    float * test;

    if (check) {
        test = (float *) malloc(m1_h * m2_w * sizeof(float));
    }

#define RANGE 2

    for (int i = 0; i < m1_h * m1_w; i++) {
        a[i] = floor(rand() / (RAND_MAX / ((float) RANGE)));
    }
    for (int i = 0; i < m1_w * m2_w; i++) {
        b[i] = floor(rand() / (RAND_MAX / ((float) RANGE)));
    }


    start_timing(&t);
    _fmat_mul(c, a, b, m1_h, m1_w, m2_w);
    end_timing(&t);

    printf("Total GPU time: %lf\n", get_time(&t));

    if (check) {
        start_timing(&t);
        fast_fmat_mul_test(test, a, b, m1_h, m1_w, m2_w);
        end_timing(&t);

        printf("Total CPU time: %lf\n", get_time(&t));
    }

    if (print) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                printf("%3.0f", b[i * m1_w + j]);
            }
            printf("\n");
        }

        printf("GPU:\n");
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m2_w; j++) {
                if (check && c[i * m2_w + j] != test[i * m2_w + j]) {
                    printf(P_RED);
                }
                printf("%3.0f", c[i * m2_w + j]);
                if (check && c[i * m2_w + j] != test[i * m2_w + j]) {
                    printf(P_DEFAULT);
                }
            }
            printf("\n");
        }

        if (check) {
            printf("CPU:\n");
            for (int i = 0; i < m1_h; i++) {
                for (int j = 0; j < m2_w; j++) {
                    printf("%3.0f", test[i * m2_w + j]);
                }
                printf("\n");
            }
        }
    }

    if (check) {
        for (int i = 0; i < m1_h * m2_w; i++) {
            assert(c[i] == test[i]);
        }

        free(test);
    }
    free(c);
    free(b);
    free(a);

    return 0;
}


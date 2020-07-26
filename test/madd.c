
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



static void add(float * dst, float * m1, float * m2, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dst[i] = m1[i] + m2[i];
    }
}




int main(int argc, char *argv[]) {
    struct timer t;
    int print = 0, check = 0;
    int opt;
    char * buf;

    ssize_t h = -1, w = -1;

    while ((opt = getopt(argc, argv, "cpi:j:n:")) != -1) {
        switch (opt) {
            case 'c':
                check = 1;
                break;
            case 'i':
                h = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                break;
            case 'j':
                w = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                break;
            case 'n':
                h = strtoul(optarg, &buf, 10);
                if (*optarg == '\0' || *buf != '\0') {
                    // invalid number
                    fprintf(stderr, "%s is not a valid base 10 unsigned "
                            "number\n", optarg);
                    return -1;
                }
                w = h;
                break;
            case 'p':
                print = 1;
                break;
            case '?':
            default:
                return -1;
        }
    }

    if (h == -1 || w == -1) {
        fprintf(stderr, "All two dimensions (i, j) must be specified\n");
        return -1;
    }

    _load_fmat_ops();

    float * a = (float *) malloc(h * w * sizeof(float));
    float * b = (float *) malloc(h * w * sizeof(float));
    float * c = (float *) malloc(h * w * sizeof(float));
    float * test;

    if (check) {
        test = (float *) malloc(h * w * sizeof(float));
    }

#define RANGE 50

    for (int i = 0; i < h * w; i++) {
        a[i] = floor(rand() / (RAND_MAX / ((float) RANGE)));
    }
    for (int i = 0; i < h * w; i++) {
        b[i] = floor(rand() / (RAND_MAX / ((float) RANGE)));
    }


    start_timing(&t);
    _fmat_add(c, a, b, h, w);
    end_timing(&t);

    printf("Total GPU time: %lf\n", get_time(&t));

    if (check) {
        start_timing(&t);
        add(test, a, b, h * w);
        end_timing(&t);

        printf("Total CPU time: %lf\n", get_time(&t));
    }

    if (print) {

        printf("GPU:\n");
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (check && c[i * w + j] != test[i * w + j]) {
                    printf(P_RED);
                }
                printf("%3.0f", c[i * w + j]);
                if (check && c[i * w + j] != test[i * w + j]) {
                    printf(P_DEFAULT);
                }
            }
            printf("\n");
        }

        if (check) {
            printf("CPU:\n");
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    printf("%3.0f", test[i * w + j]);
                }
                printf("\n");
            }
        }
    }

    if (check) {
        for (int i = 0; i < h * w; i++) {
            assert(c[i] == test[i]);
        }

        free(test);
    }
    free(c);
    free(b);
    free(a);

    return 0;
}


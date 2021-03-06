
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <pmath/print_colors.h>

#define P_FILE_LINE P_LGREEN __FILE__ P_DEFAULT ":" P_LCYAN "%d" P_DEFAULT


#define IS_ALIGNED(ptr, alignment) ((((size_t) (ptr)) & ((alignment) - 1)) == 0)

#define _STR(expr) #expr
#define STR(expr) _STR(expr)


#define MATH_ASSERT(expr) \
    do { \
        if (__builtin_expect(!(expr), 0)) { \
            fprintf(stderr, P_FILE_LINE " " P_LRED "assert " P_LYELLOW "\"" \
                    #expr "\"" P_LRED " failed" P_RESET "\n", __LINE__); \
            assert(0); \
        } \
    } while (0)


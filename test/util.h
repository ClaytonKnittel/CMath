#ifndef _TEST_UTIL_H
#define _TEST_UTIL_H


static void mem_flush(const void *p, size_t allocation_size){
    const size_t cache_line = 64;
    const char *cp = (const char *)p;
    size_t i = 0;

    if (p == NULL || allocation_size <= 0)
            return;

    for (i = 0; i < allocation_size; i += cache_line) {
            asm volatile("clflush (%0)\n\t"
                         :
                         : "r"(&cp[i])
                         : "memory");
    }

    asm volatile("sfence\n\t"
                 :
                 :
                 : "memory");
}


#endif /* _TEST_UTIL_H */

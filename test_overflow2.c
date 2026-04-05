#define SLA_IMPLEMENTATION
#include "sla.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
    size_t rows = 4;
    size_t cols = 4;
    size_t num_diags = SIZE_MAX / 2; // this times max_diag_len (4) will overflow

    // SIZE_MAX / 2 = 0x7FFFFFFFFFFFFFFF
    // * 4 = 0x1FFFFFFFFFFFFFFFC (wraps to 0xFFFFFFFFFFFFFFFC in 64-bit size_t? Wait, 64-bit size_t max is 0xFFFFFFFFFFFFFFFF)
    // Actually, (SIZE_MAX / 2) * 4 > SIZE_MAX, so it overflows.
    // Let's use SIZE_MAX / 3.

    num_diags = SIZE_MAX / 3;
    printf("Attempting to allocate with num_diags=%zu\n", num_diags);
    sla_f32_dia_mat m = sla_f32_dia_mat_alloc(rows, cols, num_diags);
    if (m.data == NULL) {
        printf("Overflow successfully caught or calloc failed normally\n");
    } else {
        printf("Overflow allowed! data=%p\n", m.data);
    }
    return 0;
}

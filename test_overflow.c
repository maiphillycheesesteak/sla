#define SLA_IMPLEMENTATION
#include "sla.h"
#include <stdio.h>
#include <stdint.h>

int main() {
    size_t rows = SIZE_MAX / 2;
    size_t cols = SIZE_MAX / 2;
    size_t num_diags = 4;
    sla_f32_dia_mat m = sla_f32_dia_mat_alloc(rows, cols, num_diags);
    if (m.data == NULL) {
        printf("Overflow successfully caught\n");
    } else {
        printf("Overflow allowed! data=%p\n", m.data);
    }
    return 0;
}

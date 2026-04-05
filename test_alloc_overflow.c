#define SLA_IMPLEMENTATION
#include "sla.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
    // Test dense mat overflow
    size_t dense_rows = SIZE_MAX / 2;
    size_t dense_cols = 4;
    sla_f32_dense_mat m_dense = sla_f32_dense_mat_alloc(dense_rows, dense_cols);
    if (m_dense.data == NULL) {
        printf("Dense matrix overflow successfully caught.\n");
    } else {
        printf("Dense matrix overflow allowed! data=%p\n", m_dense.data);
    }

    // Test dia mat overflow
    size_t dia_rows = 4;
    size_t dia_cols = 4;
    size_t dia_num_diags = SIZE_MAX / 3;
    sla_f32_dia_mat m_dia = sla_f32_dia_mat_alloc(dia_rows, dia_cols, dia_num_diags);
    if (m_dia.data == NULL && m_dia.offsets == NULL) {
        printf("DIA matrix overflow successfully caught.\n");
    } else {
        printf("DIA matrix overflow allowed! data=%p\n", m_dia.data);
    }

    return 0;
}

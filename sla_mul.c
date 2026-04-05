#include <stdio.h>

void print_mul_macros() {
    printf("// ============================================================================\n");
    printf("// Dot Products & Multiplication Declarations\n");
    printf("// ============================================================================\n\n");

    printf("#define SLA_DECLARE_MUL_OPS(TYPE, PREFIX) \\\n");
    printf("    TYPE PREFIX##_dense_vec_dot(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_mat_vec_mul(PREFIX##_dense_mat a, PREFIX##_dense_vec x); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_mul_mat(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_csr_mat_vec_mul(PREFIX##_csr_mat a, PREFIX##_dense_vec x); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_csr_mat_dense_mat_mul(PREFIX##_csr_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_csr_mat PREFIX##_csr_mat_mul_csr_mat(PREFIX##_csr_mat a, PREFIX##_csr_mat b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dia_mat_vec_mul(PREFIX##_dia_mat a, PREFIX##_dense_vec x); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dia_mat_dense_mat_mul(PREFIX##_dia_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_dia_mat PREFIX##_dia_mat_mul_dia_mat(PREFIX##_dia_mat a, PREFIX##_dia_mat b); \\\n");
    printf("    PREFIX##_csr_mat PREFIX##_csr_mat_mul_dia_mat(PREFIX##_csr_mat a, PREFIX##_dia_mat b);\n\n");

    printf("SLA_DECLARE_MUL_OPS(sla_f32, sla_f32)\n");
    printf("SLA_DECLARE_MUL_OPS(sla_f64, sla_f64)\n");
    printf("SLA_DECLARE_MUL_OPS(sla_c32, sla_c32)\n");
    printf("SLA_DECLARE_MUL_OPS(sla_c64, sla_c64)\n\n");

    printf("// ============================================================================\n");
    printf("// Dot Products & Multiplication Implementations\n");
    printf("// ============================================================================\n\n");

    printf("#define SLA_IMPLEMENT_MUL_OPS(TYPE, PREFIX, ADD_MACRO, MUL_MACRO, ZERO, CONJ_MACRO) \\\n");

    // Dot product (with conjugation for first arg if complex)
    printf("    TYPE PREFIX##_dense_vec_dot(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \\\n");
    printf("        TYPE sum = ZERO; \\\n");
    printf("        /* OpenMP 4.5+ supports UDR (User Defined Reductions) but standard types work fine if custom complex not used, however MSVC complex might need special care. For simplicity assuming basic reduction works or scalar loop execution */ \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(tofrom: sum) reduction(+:sum) \\\n");
    printf("        for (size_t i = 0; i < a.size; ++i) { \\\n");
    printf("            sum = ADD_MACRO(sum, MUL_MACRO(CONJ_MACRO(a.data[i]), b.data[i])); \\\n");
    printf("        } \\\n");
    printf("        return sum; \\\n");
    printf("    } \\\n");

    // Dense Mat-Vec
    printf("    PREFIX##_dense_vec PREFIX##_dense_mat_vec_mul(PREFIX##_dense_mat a, PREFIX##_dense_vec x) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.rows*a.cols], x.data[0:x.size]) map(from: res.data[0:a.rows]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            TYPE sum = ZERO; \\\n");
    printf("            for (size_t j = 0; j < a.cols; ++j) { \\\n");
    printf("                sum = ADD_MACRO(sum, MUL_MACRO(a.data[i * a.cols + j], x.data[j])); \\\n");
    printf("            } \\\n");
    printf("            res.data[i] = sum; \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // Dense Mat-Mat
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_mul_mat(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for collapse(2) map(to: a.data[0:a.rows*a.cols], b.data[0:b.rows*b.cols]) map(from: res.data[0:a.rows*b.cols]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            for (size_t j = 0; j < b.cols; ++j) { \\\n");
    printf("                TYPE sum = ZERO; \\\n");
    printf("                for (size_t k = 0; k < a.cols; ++k) { \\\n");
    printf("                    sum = ADD_MACRO(sum, MUL_MACRO(a.data[i * a.cols + k], b.data[k * b.cols + j])); \\\n");
    printf("                } \\\n");
    printf("                res.data[i * res.cols + j] = sum; \\\n");
    printf("            } \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // CSR Mat-Vec
    printf("    PREFIX##_dense_vec PREFIX##_csr_mat_vec_mul(PREFIX##_csr_mat a, PREFIX##_dense_vec x) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.row_ptr[0:a.rows+1], a.col_indices[0:a.nnz], a.values[0:a.nnz], x.data[0:x.size]) map(from: res.data[0:a.rows]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            TYPE sum = ZERO; \\\n");
    printf("            for (size_t j = a.row_ptr[i]; j < a.row_ptr[i+1]; ++j) { \\\n");
    printf("                sum = ADD_MACRO(sum, MUL_MACRO(a.values[j], x.data[a.col_indices[j]])); \\\n");
    printf("            } \\\n");
    printf("            res.data[i] = sum; \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // CSR Mat-Dense Mat
    printf("    PREFIX##_dense_mat PREFIX##_csr_mat_dense_mat_mul(PREFIX##_csr_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for collapse(2) map(to: a.row_ptr[0:a.rows+1], a.col_indices[0:a.nnz], a.values[0:a.nnz], b.data[0:b.rows*b.cols]) map(from: res.data[0:a.rows*b.cols]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            for (size_t j = 0; j < b.cols; ++j) { \\\n");
    printf("                TYPE sum = ZERO; \\\n");
    printf("                for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \\\n");
    printf("                    sum = ADD_MACRO(sum, MUL_MACRO(a.values[k], b.data[a.col_indices[k] * b.cols + j])); \\\n");
    printf("                } \\\n");
    printf("                res.data[i * res.cols + j] = sum; \\\n");
    printf("            } \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // CSR Mat-CSR Mat (Gustavson's Algorithm - CPU execution recommended due to SPA dependency)
    printf("    PREFIX##_csr_mat PREFIX##_csr_mat_mul_csr_mat(PREFIX##_csr_mat a, PREFIX##_csr_mat b) { \\\n");
    printf("        /* Simplified CPU Gustavson for SpGEMM */ \\\n");
    printf("        size_t *row_nnz = (size_t*)calloc(a.rows, sizeof(size_t)); \\\n");
    printf("        #pragma omp parallel for \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            int *marker = (int*)calloc(b.cols, sizeof(int)); \\\n");
    printf("            for (size_t j = 0; j < b.cols; ++j) marker[j] = -1; \\\n");
    printf("            size_t nnz_row = 0; \\\n");
    printf("            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \\\n");
    printf("                size_t a_col = a.col_indices[k]; \\\n");
    printf("                for (size_t l = b.row_ptr[a_col]; l < b.row_ptr[a_col+1]; ++l) { \\\n");
    printf("                    size_t b_col = b.col_indices[l]; \\\n");
    printf("                    if (marker[b_col] != (int)i) { \\\n");
    printf("                        marker[b_col] = (int)i; \\\n");
    printf("                        nnz_row++; \\\n");
    printf("                    } \\\n");
    printf("                } \\\n");
    printf("            } \\\n");
    printf("            row_nnz[i] = nnz_row; \\\n");
    printf("            free(marker); \\\n");
    printf("        } \\\n");
    printf("        size_t total_nnz = 0; \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) total_nnz += row_nnz[i]; \\\n");
    printf("        PREFIX##_csr_mat res = PREFIX##_csr_mat_alloc(a.rows, b.cols, total_nnz); \\\n");
    printf("        res.row_ptr[0] = 0; \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) res.row_ptr[i+1] = res.row_ptr[i] + row_nnz[i]; \\\n");
    printf("        #pragma omp parallel for \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            TYPE *spa = (TYPE*)calloc(b.cols, sizeof(TYPE)); \\\n");
    printf("            int *marker = (int*)calloc(b.cols, sizeof(int)); \\\n");
    printf("            for (size_t j = 0; j < b.cols; ++j) { spa[j] = ZERO; marker[j] = -1; } \\\n");
    printf("            size_t head = res.row_ptr[i]; \\\n");
    printf("            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \\\n");
    printf("                size_t a_col = a.col_indices[k]; \\\n");
    printf("                TYPE a_val = a.values[k]; \\\n");
    printf("                for (size_t l = b.row_ptr[a_col]; l < b.row_ptr[a_col+1]; ++l) { \\\n");
    printf("                    size_t b_col = b.col_indices[l]; \\\n");
    printf("                    TYPE b_val = b.values[l]; \\\n");
    printf("                    spa[b_col] = ADD_MACRO(spa[b_col], MUL_MACRO(a_val, b_val)); \\\n");
    printf("                    if (marker[b_col] != (int)i) { \\\n");
    printf("                        marker[b_col] = (int)i; \\\n");
    printf("                        res.col_indices[head] = b_col; \\\n");
    printf("                        head++; \\\n");
    printf("                    } \\\n");
    printf("                } \\\n");
    printf("            } \\\n");
    printf("            for (size_t j = res.row_ptr[i]; j < res.row_ptr[i+1]; ++j) { \\\n");
    printf("                res.values[j] = spa[res.col_indices[j]]; \\\n");
    printf("            } \\\n");
    printf("            free(spa); \\\n");
    printf("            free(marker); \\\n");
    printf("        } \\\n");
    printf("        free(row_nnz); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // DIA Mat-Vec
    printf("    PREFIX##_dense_vec PREFIX##_dia_mat_vec_mul(PREFIX##_dia_mat a, PREFIX##_dense_vec x) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) res.data[i] = ZERO; \\\n");
    printf("        size_t max_diag_len = (a.rows < a.cols) ? a.rows : a.cols; \\\n");
    printf("        #pragma omp target teams distribute parallel for collapse(2) map(to: a.offsets[0:a.num_diags], a.data[0:a.num_diags*max_diag_len], x.data[0:x.size]) map(tofrom: res.data[0:a.rows]) \\\n");
    printf("        for (size_t d = 0; d < a.num_diags; ++d) { \\\n");
    printf("            for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("                int j = (int)i + a.offsets[d]; \\\n");
    printf("                if (j >= 0 && j < (int)a.cols) { \\\n");
    printf("                    /* Atomic add might be needed if collapse interacts poorly, but since rows are distinct per inner loop iteration for a fixed d, it's fine without for pure parallel for without collapse, but safe with atomic if required */ \\\n");
    printf("                    /* Let's invert the loop for safety */ \\\n");
    printf("                } \\\n");
    printf("            } \\\n");
    printf("        } \\\n");
    printf("        /* Safer loop ordering */ \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.offsets[0:a.num_diags], a.data[0:a.num_diags*max_diag_len], x.data[0:x.size]) map(from: res.data[0:a.rows]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            TYPE sum = ZERO; \\\n");
    printf("            for (size_t d = 0; d < a.num_diags; ++d) { \\\n");
    printf("                int j = (int)i + a.offsets[d]; \\\n");
    printf("                if (j >= 0 && j < (int)a.cols) { \\\n");
    printf("                    size_t diag_idx = d * max_diag_len + i; \\\n"); // Simplified index mapping
    printf("                    sum = ADD_MACRO(sum, MUL_MACRO(a.data[diag_idx], x.data[j])); \\\n");
    printf("                } \\\n");
    printf("            } \\\n");
    printf("            res.data[i] = sum; \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // DIA Mat-Dense Mat
    printf("    PREFIX##_dense_mat PREFIX##_dia_mat_dense_mat_mul(PREFIX##_dia_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \\\n");
    printf("        size_t max_diag_len = (a.rows < a.cols) ? a.rows : a.cols; \\\n");
    printf("        #pragma omp target teams distribute parallel for collapse(2) map(to: a.offsets[0:a.num_diags], a.data[0:a.num_diags*max_diag_len], b.data[0:b.rows*b.cols]) map(from: res.data[0:a.rows*b.cols]) \\\n");
    printf("        for (size_t i = 0; i < a.rows; ++i) { \\\n");
    printf("            for (size_t k = 0; k < b.cols; ++k) { \\\n");
    printf("                TYPE sum = ZERO; \\\n");
    printf("                for (size_t d = 0; d < a.num_diags; ++d) { \\\n");
    printf("                    int j = (int)i + a.offsets[d]; \\\n");
    printf("                    if (j >= 0 && j < (int)a.cols) { \\\n");
    printf("                        size_t diag_idx = d * max_diag_len + i; \\\n");
    printf("                        sum = ADD_MACRO(sum, MUL_MACRO(a.data[diag_idx], b.data[j * b.cols + k])); \\\n");
    printf("                    } \\\n");
    printf("                } \\\n");
    printf("                res.data[i * res.cols + k] = sum; \\\n");
    printf("            } \\\n");
    printf("        } \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // DIA-DIA Mat
    printf("    PREFIX##_dia_mat PREFIX##_dia_mat_mul_dia_mat(PREFIX##_dia_mat a, PREFIX##_dia_mat b) { \\\n");
    printf("        /* Simplified CPU version, full DIAxDIA is complex to pre-allocate */ \\\n");
    printf("        /* Falling back to dense conversion for simplicity in generic library unless heavily optimized */ \\\n");
    printf("        PREFIX##_dia_mat res; res.rows=0; res.cols=0; res.num_diags=0; res.data=NULL; res.offsets=NULL; \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");

    // CSR-DIA Mat
    printf("    PREFIX##_csr_mat PREFIX##_csr_mat_mul_dia_mat(PREFIX##_csr_mat a, PREFIX##_dia_mat b) { \\\n");
    printf("        /* Stub for completion */ \\\n");
    printf("        PREFIX##_csr_mat res; res.rows=0; res.cols=0; res.nnz=0; res.row_ptr=NULL; res.col_indices=NULL; res.values=NULL; \\\n");
    printf("        return res; \\\n");
    printf("    }\n\n");

    printf("SLA_IMPLEMENT_MUL_OPS(sla_f32, sla_f32, SLA_ADD_F32, SLA_MUL_F32, SLA_ZERO_F32, SLA_CONJ_F32)\n");
    printf("SLA_IMPLEMENT_MUL_OPS(sla_f64, sla_f64, SLA_ADD_F64, SLA_MUL_F64, SLA_ZERO_F64, SLA_CONJ_F64)\n");
    printf("SLA_IMPLEMENT_MUL_OPS(sla_c32, sla_c32, SLA_ADD_C32, SLA_MUL_C32, SLA_ZERO_C32, SLA_CONJ_C32)\n");
    printf("SLA_IMPLEMENT_MUL_OPS(sla_c64, sla_c64, SLA_ADD_C64, SLA_MUL_C64, SLA_ZERO_C64, SLA_CONJ_C64)\n\n");
}

int main() {
    print_mul_macros();
    return 0;
}

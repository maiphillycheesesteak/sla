#include <stdio.h>

void print_elem_macros() {
    printf("// ============================================================================\n");
    printf("// Element-wise Operations Declarations\n");
    printf("// ============================================================================\n\n");

    printf("#define SLA_DECLARE_ELEM_OPS(TYPE, PREFIX) \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_add(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_sub(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_mul(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_div(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_add(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_sub(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_mul(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_div(PREFIX##_dense_mat a, PREFIX##_dense_mat b);\n\n");

    printf("SLA_DECLARE_ELEM_OPS(sla_f32, sla_f32)\n");
    printf("SLA_DECLARE_ELEM_OPS(sla_f64, sla_f64)\n");
    printf("SLA_DECLARE_ELEM_OPS(sla_c32, sla_c32)\n");
    printf("SLA_DECLARE_ELEM_OPS(sla_c64, sla_c64)\n\n");

    printf("// ============================================================================\n");
    printf("// Element-wise Operations Implementations\n");
    printf("// ============================================================================\n\n");

    printf("#define SLA_IMPLEMENT_ELEM_OPS(TYPE, PREFIX, ADD_MACRO, SUB_MACRO, MUL_MACRO, DIV_MACRO) \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_add(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \\\n");
    printf("        for (size_t i = 0; i < a.size; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_sub(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \\\n");
    printf("        for (size_t i = 0; i < a.size; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_mul(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \\\n");
    printf("        for (size_t i = 0; i < a.size; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_vec PREFIX##_dense_vec_div(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \\\n");
    printf("        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \\\n");
    printf("        for (size_t i = 0; i < a.size; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_add(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        size_t n = a.rows * a.cols; \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \\\n");
    printf("        for (size_t i = 0; i < n; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_sub(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        size_t n = a.rows * a.cols; \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \\\n");
    printf("        for (size_t i = 0; i < n; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_mul(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        size_t n = a.rows * a.cols; \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \\\n");
    printf("        for (size_t i = 0; i < n; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    } \\\n");
    printf("    PREFIX##_dense_mat PREFIX##_dense_mat_div(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \\\n");
    printf("        size_t n = a.rows * a.cols; \\\n");
    printf("        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \\\n");
    printf("        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \\\n");
    printf("        for (size_t i = 0; i < n; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \\\n");
    printf("        return res; \\\n");
    printf("    }\n\n");

    printf("SLA_IMPLEMENT_ELEM_OPS(sla_f32, sla_f32, SLA_ADD_F32, SLA_SUB_F32, SLA_MUL_F32, SLA_DIV_F32)\n");
    printf("SLA_IMPLEMENT_ELEM_OPS(sla_f64, sla_f64, SLA_ADD_F64, SLA_SUB_F64, SLA_MUL_F64, SLA_DIV_F64)\n");
    printf("SLA_IMPLEMENT_ELEM_OPS(sla_c32, sla_c32, SLA_ADD_C32, SLA_SUB_C32, SLA_MUL_C32, SLA_DIV_C32)\n");
    printf("SLA_IMPLEMENT_ELEM_OPS(sla_c64, sla_c64, SLA_ADD_C64, SLA_SUB_C64, SLA_MUL_C64, SLA_DIV_C64)\n\n");
}

int main() {
    print_elem_macros();
    return 0;
}

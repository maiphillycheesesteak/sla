// ============================================================================
// Element-wise Operations Declarations
// ============================================================================

#define SLA_DECLARE_ELEM_OPS(TYPE, PREFIX) \
    PREFIX##_dense_vec PREFIX##_dense_vec_add(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \
    PREFIX##_dense_vec PREFIX##_dense_vec_sub(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \
    PREFIX##_dense_vec PREFIX##_dense_vec_mul(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \
    PREFIX##_dense_vec PREFIX##_dense_vec_div(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \
    PREFIX##_dense_mat PREFIX##_dense_mat_add(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \
    PREFIX##_dense_mat PREFIX##_dense_mat_sub(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \
    PREFIX##_dense_mat PREFIX##_dense_mat_mul(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \
    PREFIX##_dense_mat PREFIX##_dense_mat_div(PREFIX##_dense_mat a, PREFIX##_dense_mat b);

SLA_DECLARE_ELEM_OPS(sla_f32, sla_f32)
SLA_DECLARE_ELEM_OPS(sla_f64, sla_f64)
SLA_DECLARE_ELEM_OPS(sla_c32, sla_c32)
SLA_DECLARE_ELEM_OPS(sla_c64, sla_c64)

// ============================================================================
// Element-wise Operations Implementations
// ============================================================================

#define SLA_IMPLEMENT_ELEM_OPS(TYPE, PREFIX, ADD_MACRO, SUB_MACRO, MUL_MACRO, DIV_MACRO) \
    PREFIX##_dense_vec PREFIX##_dense_vec_add(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_sub(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_mul(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_div(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(from: res.data[0:a.size]) \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_add(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \
        for (size_t i = 0; i < n; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_sub(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \
        for (size_t i = 0; i < n; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_mul(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \
        for (size_t i = 0; i < n; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_div(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        #pragma omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(from: res.data[0:n]) \
        for (size_t i = 0; i < n; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \
        return res; \
    }

SLA_IMPLEMENT_ELEM_OPS(sla_f32, sla_f32, SLA_ADD_F32, SLA_SUB_F32, SLA_MUL_F32, SLA_DIV_F32)
SLA_IMPLEMENT_ELEM_OPS(sla_f64, sla_f64, SLA_ADD_F64, SLA_SUB_F64, SLA_MUL_F64, SLA_DIV_F64)
SLA_IMPLEMENT_ELEM_OPS(sla_c32, sla_c32, SLA_ADD_C32, SLA_SUB_C32, SLA_MUL_C32, SLA_DIV_C32)
SLA_IMPLEMENT_ELEM_OPS(sla_c64, sla_c64, SLA_ADD_C64, SLA_SUB_C64, SLA_MUL_C64, SLA_DIV_C64)

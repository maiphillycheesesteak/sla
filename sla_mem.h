// ============================================================================
// Memory Management Declarations
// ============================================================================

#define SLA_DECLARE_MEM_FUNCTIONS(TYPE, PREFIX) \
    PREFIX##_dense_vec PREFIX##_dense_vec_alloc(size_t size); \
    void PREFIX##_dense_vec_free(PREFIX##_dense_vec *v); \
    PREFIX##_coo_vec PREFIX##_coo_vec_alloc(size_t size, size_t nnz); \
    void PREFIX##_coo_vec_free(PREFIX##_coo_vec *v); \
    PREFIX##_dense_mat PREFIX##_dense_mat_alloc(size_t rows, size_t cols); \
    void PREFIX##_dense_mat_free(PREFIX##_dense_mat *m); \
    PREFIX##_csr_mat PREFIX##_csr_mat_alloc(size_t rows, size_t cols, size_t nnz); \
    void PREFIX##_csr_mat_free(PREFIX##_csr_mat *m); \
    PREFIX##_dia_mat PREFIX##_dia_mat_alloc(size_t rows, size_t cols, size_t num_diags); \
    void PREFIX##_dia_mat_free(PREFIX##_dia_mat *m); \
    void PREFIX##_dense_vec_map_to_device(PREFIX##_dense_vec *v); \
    void PREFIX##_dense_vec_update_from_device(PREFIX##_dense_vec *v); \
    void PREFIX##_dense_vec_unmap_from_device(PREFIX##_dense_vec *v); \
    void PREFIX##_coo_vec_map_to_device(PREFIX##_coo_vec *v); \
    void PREFIX##_coo_vec_update_from_device(PREFIX##_coo_vec *v); \
    void PREFIX##_coo_vec_unmap_from_device(PREFIX##_coo_vec *v); \
    void PREFIX##_dense_mat_map_to_device(PREFIX##_dense_mat *m); \
    void PREFIX##_dense_mat_update_from_device(PREFIX##_dense_mat *m); \
    void PREFIX##_dense_mat_unmap_from_device(PREFIX##_dense_mat *m); \
    void PREFIX##_csr_mat_map_to_device(PREFIX##_csr_mat *m); \
    void PREFIX##_csr_mat_update_from_device(PREFIX##_csr_mat *m); \
    void PREFIX##_csr_mat_unmap_from_device(PREFIX##_csr_mat *m); \
    void PREFIX##_dia_mat_map_to_device(PREFIX##_dia_mat *m); \
    void PREFIX##_dia_mat_update_from_device(PREFIX##_dia_mat *m); \
    void PREFIX##_dia_mat_unmap_from_device(PREFIX##_dia_mat *m);

SLA_DECLARE_MEM_FUNCTIONS(sla_f32, sla_f32)
SLA_DECLARE_MEM_FUNCTIONS(sla_f64, sla_f64)
SLA_DECLARE_MEM_FUNCTIONS(sla_c32, sla_c32)
SLA_DECLARE_MEM_FUNCTIONS(sla_c64, sla_c64)

// ============================================================================
// Memory Management Implementations
// ============================================================================

#define SLA_IMPLEMENT_MEM_FUNCTIONS(TYPE, PREFIX) \
    PREFIX##_dense_vec PREFIX##_dense_vec_alloc(size_t size) { \
        PREFIX##_dense_vec v; \
        v.size = size; \
        v.data = (TYPE*)calloc(size, sizeof(TYPE)); \
        return v; \
    } \
    void PREFIX##_dense_vec_free(PREFIX##_dense_vec *v) { \
        if (v->data) free(v->data); \
        v->data = NULL; \
        v->size = 0; \
    } \
    PREFIX##_coo_vec PREFIX##_coo_vec_alloc(size_t size, size_t nnz) { \
        PREFIX##_coo_vec v; \
        v.size = size; \
        v.nnz = nnz; \
        v.values = (TYPE*)calloc(nnz, sizeof(TYPE)); \
        v.indices = (size_t*)calloc(nnz, sizeof(size_t)); \
        return v; \
    } \
    void PREFIX##_coo_vec_free(PREFIX##_coo_vec *v) { \
        if (v->values) free(v->values); \
        if (v->indices) free(v->indices); \
        v->values = NULL; \
        v->indices = NULL; \
        v->size = 0; \
        v->nnz = 0; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_alloc(size_t rows, size_t cols) { \
        PREFIX##_dense_mat m; \
        m.rows = rows; \
        m.cols = cols; \
        m.data = (TYPE*)calloc(rows * cols, sizeof(TYPE)); \
        return m; \
    } \
    void PREFIX##_dense_mat_free(PREFIX##_dense_mat *m) { \
        if (m->data) free(m->data); \
        m->data = NULL; \
        m->rows = 0; \
        m->cols = 0; \
    } \
    PREFIX##_csr_mat PREFIX##_csr_mat_alloc(size_t rows, size_t cols, size_t nnz) { \
        PREFIX##_csr_mat m; \
        m.rows = rows; \
        m.cols = cols; \
        m.nnz = nnz; \
        m.values = (TYPE*)calloc(nnz, sizeof(TYPE)); \
        m.col_indices = (size_t*)calloc(nnz, sizeof(size_t)); \
        m.row_ptr = (size_t*)calloc(rows + 1, sizeof(size_t)); \
        return m; \
    } \
    void PREFIX##_csr_mat_free(PREFIX##_csr_mat *m) { \
        if (m->values) free(m->values); \
        if (m->col_indices) free(m->col_indices); \
        if (m->row_ptr) free(m->row_ptr); \
        m->values = NULL; \
        m->col_indices = NULL; \
        m->row_ptr = NULL; \
        m->rows = 0; \
        m->cols = 0; \
        m->nnz = 0; \
    } \
    PREFIX##_dia_mat PREFIX##_dia_mat_alloc(size_t rows, size_t cols, size_t num_diags) { \
        PREFIX##_dia_mat m; \
        m.rows = rows; \
        m.cols = cols; \
        m.num_diags = num_diags; \
        size_t max_diag_len = (rows < cols) ? rows : cols; \
        m.data = (TYPE*)calloc(num_diags * max_diag_len, sizeof(TYPE)); \
        m.offsets = (int*)calloc(num_diags, sizeof(int)); \
        return m; \
    } \
    void PREFIX##_dia_mat_free(PREFIX##_dia_mat *m) { \
        if (m->data) free(m->data); \
        if (m->offsets) free(m->offsets); \
        m->data = NULL; \
        m->offsets = NULL; \
        m->rows = 0; \
        m->cols = 0; \
        m->num_diags = 0; \
    } \
    void PREFIX##_dense_vec_map_to_device(PREFIX##_dense_vec *v) { \
        #pragma omp target enter data map(to: v[0], v->data[0:v->size]) \
    } \
    void PREFIX##_dense_vec_update_from_device(PREFIX##_dense_vec *v) { \
        #pragma omp target update from(v->data[0:v->size]) \
    } \
    void PREFIX##_dense_vec_unmap_from_device(PREFIX##_dense_vec *v) { \
        #pragma omp target exit data map(from: v->data[0:v->size], v[0]) \
    } \
    void PREFIX##_coo_vec_map_to_device(PREFIX##_coo_vec *v) { \
        #pragma omp target enter data map(to: v[0], v->values[0:v->nnz], v->indices[0:v->nnz]) \
    } \
    void PREFIX##_coo_vec_update_from_device(PREFIX##_coo_vec *v) { \
        #pragma omp target update from(v->values[0:v->nnz]) \
    } \
    void PREFIX##_coo_vec_unmap_from_device(PREFIX##_coo_vec *v) { \
        #pragma omp target exit data map(from: v->values[0:v->nnz], v->indices[0:v->nnz], v[0]) \
    } \
    void PREFIX##_dense_mat_map_to_device(PREFIX##_dense_mat *m) { \
        #pragma omp target enter data map(to: m[0], m->data[0:m->rows*m->cols]) \
    } \
    void PREFIX##_dense_mat_update_from_device(PREFIX##_dense_mat *m) { \
        #pragma omp target update from(m->data[0:m->rows*m->cols]) \
    } \
    void PREFIX##_dense_mat_unmap_from_device(PREFIX##_dense_mat *m) { \
        #pragma omp target exit data map(from: m->data[0:m->rows*m->cols], m[0]) \
    } \
    void PREFIX##_csr_mat_map_to_device(PREFIX##_csr_mat *m) { \
        #pragma omp target enter data map(to: m[0], m->values[0:m->nnz], m->col_indices[0:m->nnz], m->row_ptr[0:m->rows+1]) \
    } \
    void PREFIX##_csr_mat_update_from_device(PREFIX##_csr_mat *m) { \
        #pragma omp target update from(m->values[0:m->nnz]) \
    } \
    void PREFIX##_csr_mat_unmap_from_device(PREFIX##_csr_mat *m) { \
        #pragma omp target exit data map(from: m->values[0:m->nnz], m->col_indices[0:m->nnz], m->row_ptr[0:m->rows+1], m[0]) \
    } \
    void PREFIX##_dia_mat_map_to_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        #pragma omp target enter data map(to: m[0], m->data[0:m->num_diags * max_diag_len], m->offsets[0:m->num_diags]) \
    } \
    void PREFIX##_dia_mat_update_from_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        #pragma omp target update from(m->data[0:m->num_diags * max_diag_len]) \
    } \
    void PREFIX##_dia_mat_unmap_from_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        #pragma omp target exit data map(from: m->data[0:m->num_diags * max_diag_len], m->offsets[0:m->num_diags], m[0]) \
    }

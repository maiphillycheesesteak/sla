// ============================================================================
// C11 _Generic Unified API Interface
// ============================================================================

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L

#define sla_alloc_dense_vec(type, size) _Generic((type)0, \
    sla_f32: sla_f32_dense_vec_alloc, \
    sla_f64: sla_f64_dense_vec_alloc, \
    sla_c32: sla_c32_dense_vec_alloc, \
    sla_c64: sla_c64_dense_vec_alloc)(size)

#define sla_alloc_coo_vec(type, size, nnz) _Generic((type)0, \
    sla_f32: sla_f32_coo_vec_alloc, \
    sla_f64: sla_f64_coo_vec_alloc, \
    sla_c32: sla_c32_coo_vec_alloc, \
    sla_c64: sla_c64_coo_vec_alloc)(size, nnz)

#define sla_alloc_dense_mat(type, rows, cols) _Generic((type)0, \
    sla_f32: sla_f32_dense_mat_alloc, \
    sla_f64: sla_f64_dense_mat_alloc, \
    sla_c32: sla_c32_dense_mat_alloc, \
    sla_c64: sla_c64_dense_mat_alloc)(rows, cols)

#define sla_alloc_csr_mat(type, rows, cols, nnz) _Generic((type)0, \
    sla_f32: sla_f32_csr_mat_alloc, \
    sla_f64: sla_f64_csr_mat_alloc, \
    sla_c32: sla_c32_csr_mat_alloc, \
    sla_c64: sla_c64_csr_mat_alloc)(rows, cols, nnz)

#define sla_alloc_dia_mat(type, rows, cols, num_diags) _Generic((type)0, \
    sla_f32: sla_f32_dia_mat_alloc, \
    sla_f64: sla_f64_dia_mat_alloc, \
    sla_c32: sla_c32_dia_mat_alloc, \
    sla_c64: sla_c64_dia_mat_alloc)(rows, cols, num_diags)

#define sla_free(x) _Generic((x), \
    sla_f32_dense_vec*: sla_f32_dense_vec_free, \
    sla_f64_dense_vec*: sla_f64_dense_vec_free, \
    sla_c32_dense_vec*: sla_c32_dense_vec_free, \
    sla_c64_dense_vec*: sla_c64_dense_vec_free, \
    sla_f32_coo_vec*: sla_f32_coo_vec_free, \
    sla_f64_coo_vec*: sla_f64_coo_vec_free, \
    sla_c32_coo_vec*: sla_c32_coo_vec_free, \
    sla_c64_coo_vec*: sla_c64_coo_vec_free, \
    sla_f32_dense_mat*: sla_f32_dense_mat_free, \
    sla_f64_dense_mat*: sla_f64_dense_mat_free, \
    sla_c32_dense_mat*: sla_c32_dense_mat_free, \
    sla_c64_dense_mat*: sla_c64_dense_mat_free, \
    sla_f32_csr_mat*: sla_f32_csr_mat_free, \
    sla_f64_csr_mat*: sla_f64_csr_mat_free, \
    sla_c32_csr_mat*: sla_c32_csr_mat_free, \
    sla_c64_csr_mat*: sla_c64_csr_mat_free, \
    sla_f32_dia_mat*: sla_f32_dia_mat_free, \
    sla_f64_dia_mat*: sla_f64_dia_mat_free, \
    sla_c32_dia_mat*: sla_c32_dia_mat_free, \
    sla_c64_dia_mat*: sla_c64_dia_mat_free)(x)

#define sla_map_to_device(x) _Generic((x), \
    sla_f32_dense_vec*: sla_f32_dense_vec_map_to_device, \
    sla_f64_dense_vec*: sla_f64_dense_vec_map_to_device, \
    sla_c32_dense_vec*: sla_c32_dense_vec_map_to_device, \
    sla_c64_dense_vec*: sla_c64_dense_vec_map_to_device, \
    sla_f32_coo_vec*: sla_f32_coo_vec_map_to_device, \
    sla_f64_coo_vec*: sla_f64_coo_vec_map_to_device, \
    sla_c32_coo_vec*: sla_c32_coo_vec_map_to_device, \
    sla_c64_coo_vec*: sla_c64_coo_vec_map_to_device, \
    sla_f32_dense_mat*: sla_f32_dense_mat_map_to_device, \
    sla_f64_dense_mat*: sla_f64_dense_mat_map_to_device, \
    sla_c32_dense_mat*: sla_c32_dense_mat_map_to_device, \
    sla_c64_dense_mat*: sla_c64_dense_mat_map_to_device, \
    sla_f32_csr_mat*: sla_f32_csr_mat_map_to_device, \
    sla_f64_csr_mat*: sla_f64_csr_mat_map_to_device, \
    sla_c32_csr_mat*: sla_c32_csr_mat_map_to_device, \
    sla_c64_csr_mat*: sla_c64_csr_mat_map_to_device, \
    sla_f32_dia_mat*: sla_f32_dia_mat_map_to_device, \
    sla_f64_dia_mat*: sla_f64_dia_mat_map_to_device, \
    sla_c32_dia_mat*: sla_c32_dia_mat_map_to_device, \
    sla_c64_dia_mat*: sla_c64_dia_mat_map_to_device)(x)

#define sla_update_from_device(x) _Generic((x), \
    sla_f32_dense_vec*: sla_f32_dense_vec_update_from_device, \
    sla_f64_dense_vec*: sla_f64_dense_vec_update_from_device, \
    sla_c32_dense_vec*: sla_c32_dense_vec_update_from_device, \
    sla_c64_dense_vec*: sla_c64_dense_vec_update_from_device, \
    sla_f32_coo_vec*: sla_f32_coo_vec_update_from_device, \
    sla_f64_coo_vec*: sla_f64_coo_vec_update_from_device, \
    sla_c32_coo_vec*: sla_c32_coo_vec_update_from_device, \
    sla_c64_coo_vec*: sla_c64_coo_vec_update_from_device, \
    sla_f32_dense_mat*: sla_f32_dense_mat_update_from_device, \
    sla_f64_dense_mat*: sla_f64_dense_mat_update_from_device, \
    sla_c32_dense_mat*: sla_c32_dense_mat_update_from_device, \
    sla_c64_dense_mat*: sla_c64_dense_mat_update_from_device, \
    sla_f32_csr_mat*: sla_f32_csr_mat_update_from_device, \
    sla_f64_csr_mat*: sla_f64_csr_mat_update_from_device, \
    sla_c32_csr_mat*: sla_c32_csr_mat_update_from_device, \
    sla_c64_csr_mat*: sla_c64_csr_mat_update_from_device, \
    sla_f32_dia_mat*: sla_f32_dia_mat_update_from_device, \
    sla_f64_dia_mat*: sla_f64_dia_mat_update_from_device, \
    sla_c32_dia_mat*: sla_c32_dia_mat_update_from_device, \
    sla_c64_dia_mat*: sla_c64_dia_mat_update_from_device)(x)

#define sla_unmap_from_device(x) _Generic((x), \
    sla_f32_dense_vec*: sla_f32_dense_vec_unmap_from_device, \
    sla_f64_dense_vec*: sla_f64_dense_vec_unmap_from_device, \
    sla_c32_dense_vec*: sla_c32_dense_vec_unmap_from_device, \
    sla_c64_dense_vec*: sla_c64_dense_vec_unmap_from_device, \
    sla_f32_coo_vec*: sla_f32_coo_vec_unmap_from_device, \
    sla_f64_coo_vec*: sla_f64_coo_vec_unmap_from_device, \
    sla_c32_coo_vec*: sla_c32_coo_vec_unmap_from_device, \
    sla_c64_coo_vec*: sla_c64_coo_vec_unmap_from_device, \
    sla_f32_dense_mat*: sla_f32_dense_mat_unmap_from_device, \
    sla_f64_dense_mat*: sla_f64_dense_mat_unmap_from_device, \
    sla_c32_dense_mat*: sla_c32_dense_mat_unmap_from_device, \
    sla_c64_dense_mat*: sla_c64_dense_mat_unmap_from_device, \
    sla_f32_csr_mat*: sla_f32_csr_mat_unmap_from_device, \
    sla_f64_csr_mat*: sla_f64_csr_mat_unmap_from_device, \
    sla_c32_csr_mat*: sla_c32_csr_mat_unmap_from_device, \
    sla_c64_csr_mat*: sla_c64_csr_mat_unmap_from_device, \
    sla_f32_dia_mat*: sla_f32_dia_mat_unmap_from_device, \
    sla_f64_dia_mat*: sla_f64_dia_mat_unmap_from_device, \
    sla_c32_dia_mat*: sla_c32_dia_mat_unmap_from_device, \
    sla_c64_dia_mat*: sla_c64_dia_mat_unmap_from_device)(x)

#define sla_add(a, b) _Generic((a), \
    sla_f32_dense_vec: sla_f32_dense_vec_add, \
    sla_f64_dense_vec: sla_f64_dense_vec_add, \
    sla_c32_dense_vec: sla_c32_dense_vec_add, \
    sla_c64_dense_vec: sla_c64_dense_vec_add, \
    sla_f32_dense_mat: sla_f32_dense_mat_add, \
    sla_f64_dense_mat: sla_f64_dense_mat_add, \
    sla_c32_dense_mat: sla_c32_dense_mat_add, \
    sla_c64_dense_mat: sla_c64_dense_mat_add)(a, b)

#define sla_sub(a, b) _Generic((a), \
    sla_f32_dense_vec: sla_f32_dense_vec_sub, \
    sla_f64_dense_vec: sla_f64_dense_vec_sub, \
    sla_c32_dense_vec: sla_c32_dense_vec_sub, \
    sla_c64_dense_vec: sla_c64_dense_vec_sub, \
    sla_f32_dense_mat: sla_f32_dense_mat_sub, \
    sla_f64_dense_mat: sla_f64_dense_mat_sub, \
    sla_c32_dense_mat: sla_c32_dense_mat_sub, \
    sla_c64_dense_mat: sla_c64_dense_mat_sub)(a, b)

#define sla_div(a, b) _Generic((a), \
    sla_f32_dense_vec: sla_f32_dense_vec_div, \
    sla_f64_dense_vec: sla_f64_dense_vec_div, \
    sla_c32_dense_vec: sla_c32_dense_vec_div, \
    sla_c64_dense_vec: sla_c64_dense_vec_div, \
    sla_f32_dense_mat: sla_f32_dense_mat_div, \
    sla_f64_dense_mat: sla_f64_dense_mat_div, \
    sla_c32_dense_mat: sla_c32_dense_mat_div, \
    sla_c64_dense_mat: sla_c64_dense_mat_div)(a, b)

#define sla_dot(a, b) _Generic((a), \
    sla_f32_dense_vec: sla_f32_dense_vec_dot, \
    sla_f64_dense_vec: sla_f64_dense_vec_dot, \
    sla_c32_dense_vec: sla_c32_dense_vec_dot, \
    sla_c64_dense_vec: sla_c64_dense_vec_dot)(a, b)

#define sla_mul(a, b) _Generic((a), \
    sla_f32_dense_vec: sla_f32_dense_vec_mul, \
    sla_f64_dense_vec: sla_f64_dense_vec_mul, \
    sla_c32_dense_vec: sla_c32_dense_vec_mul, \
    sla_c64_dense_vec: sla_c64_dense_vec_mul, \
    sla_f32_dense_mat: _Generic((b), \
        sla_f32_dense_vec: sla_f32_dense_mat_vec_mul, \
        sla_f32_dense_mat: sla_f32_dense_mat_mul_mat, \
        default: sla_f32_dense_mat_mul), \
    sla_f64_dense_mat: _Generic((b), \
        sla_f64_dense_vec: sla_f64_dense_mat_vec_mul, \
        sla_f64_dense_mat: sla_f64_dense_mat_mul_mat, \
        default: sla_f64_dense_mat_mul), \
    sla_c32_dense_mat: _Generic((b), \
        sla_c32_dense_vec: sla_c32_dense_mat_vec_mul, \
        sla_c32_dense_mat: sla_c32_dense_mat_mul_mat, \
        default: sla_c32_dense_mat_mul), \
    sla_c64_dense_mat: _Generic((b), \
        sla_c64_dense_vec: sla_c64_dense_mat_vec_mul, \
        sla_c64_dense_mat: sla_c64_dense_mat_mul_mat, \
        default: sla_c64_dense_mat_mul), \
    sla_f32_csr_mat: _Generic((b), \
        sla_f32_dense_vec: sla_f32_csr_mat_vec_mul, \
        sla_f32_dense_mat: sla_f32_csr_mat_dense_mat_mul, \
        sla_f32_csr_mat: sla_f32_csr_mat_mul_csr_mat, \
        sla_f32_dia_mat: sla_f32_csr_mat_mul_dia_mat), \
    sla_f64_csr_mat: _Generic((b), \
        sla_f64_dense_vec: sla_f64_csr_mat_vec_mul, \
        sla_f64_dense_mat: sla_f64_csr_mat_dense_mat_mul, \
        sla_f64_csr_mat: sla_f64_csr_mat_mul_csr_mat, \
        sla_f64_dia_mat: sla_f64_csr_mat_mul_dia_mat), \
    sla_c32_csr_mat: _Generic((b), \
        sla_c32_dense_vec: sla_c32_csr_mat_vec_mul, \
        sla_c32_dense_mat: sla_c32_csr_mat_dense_mat_mul, \
        sla_c32_csr_mat: sla_c32_csr_mat_mul_csr_mat, \
        sla_c32_dia_mat: sla_c32_csr_mat_mul_dia_mat), \
    sla_c64_csr_mat: _Generic((b), \
        sla_c64_dense_vec: sla_c64_csr_mat_vec_mul, \
        sla_c64_dense_mat: sla_c64_csr_mat_dense_mat_mul, \
        sla_c64_csr_mat: sla_c64_csr_mat_mul_csr_mat, \
        sla_c64_dia_mat: sla_c64_csr_mat_mul_dia_mat), \
    sla_f32_dia_mat: _Generic((b), \
        sla_f32_dense_vec: sla_f32_dia_mat_vec_mul, \
        sla_f32_dense_mat: sla_f32_dia_mat_dense_mat_mul, \
        sla_f32_dia_mat: sla_f32_dia_mat_mul_dia_mat), \
    sla_f64_dia_mat: _Generic((b), \
        sla_f64_dense_vec: sla_f64_dia_mat_vec_mul, \
        sla_f64_dense_mat: sla_f64_dia_mat_dense_mat_mul, \
        sla_f64_dia_mat: sla_f64_dia_mat_mul_dia_mat), \
    sla_c32_dia_mat: _Generic((b), \
        sla_c32_dense_vec: sla_c32_dia_mat_vec_mul, \
        sla_c32_dense_mat: sla_c32_dia_mat_dense_mat_mul, \
        sla_c32_dia_mat: sla_c32_dia_mat_mul_dia_mat), \
    sla_c64_dia_mat: _Generic((b), \
        sla_c64_dense_vec: sla_c64_dia_mat_vec_mul, \
        sla_c64_dense_mat: sla_c64_dia_mat_dense_mat_mul, \
        sla_c64_dia_mat: sla_c64_dia_mat_mul_dia_mat)
)(a, b)

#endif // C11

#include <stdio.h>

void print_generic() {
    printf("// ============================================================================\n");
    printf("// C11 _Generic Unified API Interface\n");
    printf("// ============================================================================\n\n");

    printf("#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L\n\n");

    printf("#define sla_alloc_dense_vec(type, size) _Generic((type)0, \\\n");
    printf("    sla_f32: sla_f32_dense_vec_alloc, \\\n");
    printf("    sla_f64: sla_f64_dense_vec_alloc, \\\n");
    printf("    sla_c32: sla_c32_dense_vec_alloc, \\\n");
    printf("    sla_c64: sla_c64_dense_vec_alloc)(size)\n\n");

    printf("#define sla_alloc_coo_vec(type, size, nnz) _Generic((type)0, \\\n");
    printf("    sla_f32: sla_f32_coo_vec_alloc, \\\n");
    printf("    sla_f64: sla_f64_coo_vec_alloc, \\\n");
    printf("    sla_c32: sla_c32_coo_vec_alloc, \\\n");
    printf("    sla_c64: sla_c64_coo_vec_alloc)(size, nnz)\n\n");

    printf("#define sla_alloc_dense_mat(type, rows, cols) _Generic((type)0, \\\n");
    printf("    sla_f32: sla_f32_dense_mat_alloc, \\\n");
    printf("    sla_f64: sla_f64_dense_mat_alloc, \\\n");
    printf("    sla_c32: sla_c32_dense_mat_alloc, \\\n");
    printf("    sla_c64: sla_c64_dense_mat_alloc)(rows, cols)\n\n");

    printf("#define sla_alloc_csr_mat(type, rows, cols, nnz) _Generic((type)0, \\\n");
    printf("    sla_f32: sla_f32_csr_mat_alloc, \\\n");
    printf("    sla_f64: sla_f64_csr_mat_alloc, \\\n");
    printf("    sla_c32: sla_c32_csr_mat_alloc, \\\n");
    printf("    sla_c64: sla_c64_csr_mat_alloc)(rows, cols, nnz)\n\n");

    printf("#define sla_alloc_dia_mat(type, rows, cols, num_diags) _Generic((type)0, \\\n");
    printf("    sla_f32: sla_f32_dia_mat_alloc, \\\n");
    printf("    sla_f64: sla_f64_dia_mat_alloc, \\\n");
    printf("    sla_c32: sla_c32_dia_mat_alloc, \\\n");
    printf("    sla_c64: sla_c64_dia_mat_alloc)(rows, cols, num_diags)\n\n");

    printf("#define sla_free(x) _Generic((x), \\\n");
    printf("    sla_f32_dense_vec*: sla_f32_dense_vec_free, \\\n");
    printf("    sla_f64_dense_vec*: sla_f64_dense_vec_free, \\\n");
    printf("    sla_c32_dense_vec*: sla_c32_dense_vec_free, \\\n");
    printf("    sla_c64_dense_vec*: sla_c64_dense_vec_free, \\\n");
    printf("    sla_f32_coo_vec*: sla_f32_coo_vec_free, \\\n");
    printf("    sla_f64_coo_vec*: sla_f64_coo_vec_free, \\\n");
    printf("    sla_c32_coo_vec*: sla_c32_coo_vec_free, \\\n");
    printf("    sla_c64_coo_vec*: sla_c64_coo_vec_free, \\\n");
    printf("    sla_f32_dense_mat*: sla_f32_dense_mat_free, \\\n");
    printf("    sla_f64_dense_mat*: sla_f64_dense_mat_free, \\\n");
    printf("    sla_c32_dense_mat*: sla_c32_dense_mat_free, \\\n");
    printf("    sla_c64_dense_mat*: sla_c64_dense_mat_free, \\\n");
    printf("    sla_f32_csr_mat*: sla_f32_csr_mat_free, \\\n");
    printf("    sla_f64_csr_mat*: sla_f64_csr_mat_free, \\\n");
    printf("    sla_c32_csr_mat*: sla_c32_csr_mat_free, \\\n");
    printf("    sla_c64_csr_mat*: sla_c64_csr_mat_free, \\\n");
    printf("    sla_f32_dia_mat*: sla_f32_dia_mat_free, \\\n");
    printf("    sla_f64_dia_mat*: sla_f64_dia_mat_free, \\\n");
    printf("    sla_c32_dia_mat*: sla_c32_dia_mat_free, \\\n");
    printf("    sla_c64_dia_mat*: sla_c64_dia_mat_free)(x)\n\n");

    printf("#define sla_map_to_device(x) _Generic((x), \\\n");
    printf("    sla_f32_dense_vec*: sla_f32_dense_vec_map_to_device, \\\n");
    printf("    sla_f64_dense_vec*: sla_f64_dense_vec_map_to_device, \\\n");
    printf("    sla_c32_dense_vec*: sla_c32_dense_vec_map_to_device, \\\n");
    printf("    sla_c64_dense_vec*: sla_c64_dense_vec_map_to_device, \\\n");
    printf("    sla_f32_coo_vec*: sla_f32_coo_vec_map_to_device, \\\n");
    printf("    sla_f64_coo_vec*: sla_f64_coo_vec_map_to_device, \\\n");
    printf("    sla_c32_coo_vec*: sla_c32_coo_vec_map_to_device, \\\n");
    printf("    sla_c64_coo_vec*: sla_c64_coo_vec_map_to_device, \\\n");
    printf("    sla_f32_dense_mat*: sla_f32_dense_mat_map_to_device, \\\n");
    printf("    sla_f64_dense_mat*: sla_f64_dense_mat_map_to_device, \\\n");
    printf("    sla_c32_dense_mat*: sla_c32_dense_mat_map_to_device, \\\n");
    printf("    sla_c64_dense_mat*: sla_c64_dense_mat_map_to_device, \\\n");
    printf("    sla_f32_csr_mat*: sla_f32_csr_mat_map_to_device, \\\n");
    printf("    sla_f64_csr_mat*: sla_f64_csr_mat_map_to_device, \\\n");
    printf("    sla_c32_csr_mat*: sla_c32_csr_mat_map_to_device, \\\n");
    printf("    sla_c64_csr_mat*: sla_c64_csr_mat_map_to_device, \\\n");
    printf("    sla_f32_dia_mat*: sla_f32_dia_mat_map_to_device, \\\n");
    printf("    sla_f64_dia_mat*: sla_f64_dia_mat_map_to_device, \\\n");
    printf("    sla_c32_dia_mat*: sla_c32_dia_mat_map_to_device, \\\n");
    printf("    sla_c64_dia_mat*: sla_c64_dia_mat_map_to_device)(x)\n\n");

    printf("#define sla_update_from_device(x) _Generic((x), \\\n");
    printf("    sla_f32_dense_vec*: sla_f32_dense_vec_update_from_device, \\\n");
    printf("    sla_f64_dense_vec*: sla_f64_dense_vec_update_from_device, \\\n");
    printf("    sla_c32_dense_vec*: sla_c32_dense_vec_update_from_device, \\\n");
    printf("    sla_c64_dense_vec*: sla_c64_dense_vec_update_from_device, \\\n");
    printf("    sla_f32_coo_vec*: sla_f32_coo_vec_update_from_device, \\\n");
    printf("    sla_f64_coo_vec*: sla_f64_coo_vec_update_from_device, \\\n");
    printf("    sla_c32_coo_vec*: sla_c32_coo_vec_update_from_device, \\\n");
    printf("    sla_c64_coo_vec*: sla_c64_coo_vec_update_from_device, \\\n");
    printf("    sla_f32_dense_mat*: sla_f32_dense_mat_update_from_device, \\\n");
    printf("    sla_f64_dense_mat*: sla_f64_dense_mat_update_from_device, \\\n");
    printf("    sla_c32_dense_mat*: sla_c32_dense_mat_update_from_device, \\\n");
    printf("    sla_c64_dense_mat*: sla_c64_dense_mat_update_from_device, \\\n");
    printf("    sla_f32_csr_mat*: sla_f32_csr_mat_update_from_device, \\\n");
    printf("    sla_f64_csr_mat*: sla_f64_csr_mat_update_from_device, \\\n");
    printf("    sla_c32_csr_mat*: sla_c32_csr_mat_update_from_device, \\\n");
    printf("    sla_c64_csr_mat*: sla_c64_csr_mat_update_from_device, \\\n");
    printf("    sla_f32_dia_mat*: sla_f32_dia_mat_update_from_device, \\\n");
    printf("    sla_f64_dia_mat*: sla_f64_dia_mat_update_from_device, \\\n");
    printf("    sla_c32_dia_mat*: sla_c32_dia_mat_update_from_device, \\\n");
    printf("    sla_c64_dia_mat*: sla_c64_dia_mat_update_from_device)(x)\n\n");

    printf("#define sla_unmap_from_device(x) _Generic((x), \\\n");
    printf("    sla_f32_dense_vec*: sla_f32_dense_vec_unmap_from_device, \\\n");
    printf("    sla_f64_dense_vec*: sla_f64_dense_vec_unmap_from_device, \\\n");
    printf("    sla_c32_dense_vec*: sla_c32_dense_vec_unmap_from_device, \\\n");
    printf("    sla_c64_dense_vec*: sla_c64_dense_vec_unmap_from_device, \\\n");
    printf("    sla_f32_coo_vec*: sla_f32_coo_vec_unmap_from_device, \\\n");
    printf("    sla_f64_coo_vec*: sla_f64_coo_vec_unmap_from_device, \\\n");
    printf("    sla_c32_coo_vec*: sla_c32_coo_vec_unmap_from_device, \\\n");
    printf("    sla_c64_coo_vec*: sla_c64_coo_vec_unmap_from_device, \\\n");
    printf("    sla_f32_dense_mat*: sla_f32_dense_mat_unmap_from_device, \\\n");
    printf("    sla_f64_dense_mat*: sla_f64_dense_mat_unmap_from_device, \\\n");
    printf("    sla_c32_dense_mat*: sla_c32_dense_mat_unmap_from_device, \\\n");
    printf("    sla_c64_dense_mat*: sla_c64_dense_mat_unmap_from_device, \\\n");
    printf("    sla_f32_csr_mat*: sla_f32_csr_mat_unmap_from_device, \\\n");
    printf("    sla_f64_csr_mat*: sla_f64_csr_mat_unmap_from_device, \\\n");
    printf("    sla_c32_csr_mat*: sla_c32_csr_mat_unmap_from_device, \\\n");
    printf("    sla_c64_csr_mat*: sla_c64_csr_mat_unmap_from_device, \\\n");
    printf("    sla_f32_dia_mat*: sla_f32_dia_mat_unmap_from_device, \\\n");
    printf("    sla_f64_dia_mat*: sla_f64_dia_mat_unmap_from_device, \\\n");
    printf("    sla_c32_dia_mat*: sla_c32_dia_mat_unmap_from_device, \\\n");
    printf("    sla_c64_dia_mat*: sla_c64_dia_mat_unmap_from_device)(x)\n\n");

    printf("#define sla_add(a, b) _Generic((a), \\\n");
    printf("    sla_f32_dense_vec: sla_f32_dense_vec_add, \\\n");
    printf("    sla_f64_dense_vec: sla_f64_dense_vec_add, \\\n");
    printf("    sla_c32_dense_vec: sla_c32_dense_vec_add, \\\n");
    printf("    sla_c64_dense_vec: sla_c64_dense_vec_add, \\\n");
    printf("    sla_f32_dense_mat: sla_f32_dense_mat_add, \\\n");
    printf("    sla_f64_dense_mat: sla_f64_dense_mat_add, \\\n");
    printf("    sla_c32_dense_mat: sla_c32_dense_mat_add, \\\n");
    printf("    sla_c64_dense_mat: sla_c64_dense_mat_add)(a, b)\n\n");

    printf("#define sla_sub(a, b) _Generic((a), \\\n");
    printf("    sla_f32_dense_vec: sla_f32_dense_vec_sub, \\\n");
    printf("    sla_f64_dense_vec: sla_f64_dense_vec_sub, \\\n");
    printf("    sla_c32_dense_vec: sla_c32_dense_vec_sub, \\\n");
    printf("    sla_c64_dense_vec: sla_c64_dense_vec_sub, \\\n");
    printf("    sla_f32_dense_mat: sla_f32_dense_mat_sub, \\\n");
    printf("    sla_f64_dense_mat: sla_f64_dense_mat_sub, \\\n");
    printf("    sla_c32_dense_mat: sla_c32_dense_mat_sub, \\\n");
    printf("    sla_c64_dense_mat: sla_c64_dense_mat_sub)(a, b)\n\n");

    printf("#define sla_div(a, b) _Generic((a), \\\n");
    printf("    sla_f32_dense_vec: sla_f32_dense_vec_div, \\\n");
    printf("    sla_f64_dense_vec: sla_f64_dense_vec_div, \\\n");
    printf("    sla_c32_dense_vec: sla_c32_dense_vec_div, \\\n");
    printf("    sla_c64_dense_vec: sla_c64_dense_vec_div, \\\n");
    printf("    sla_f32_dense_mat: sla_f32_dense_mat_div, \\\n");
    printf("    sla_f64_dense_mat: sla_f64_dense_mat_div, \\\n");
    printf("    sla_c32_dense_mat: sla_c32_dense_mat_div, \\\n");
    printf("    sla_c64_dense_mat: sla_c64_dense_mat_div)(a, b)\n\n");

    printf("#define sla_dot(a, b) _Generic((a), \\\n");
    printf("    sla_f32_dense_vec: sla_f32_dense_vec_dot, \\\n");
    printf("    sla_f64_dense_vec: sla_f64_dense_vec_dot, \\\n");
    printf("    sla_c32_dense_vec: sla_c32_dense_vec_dot, \\\n");
    printf("    sla_c64_dense_vec: sla_c64_dense_vec_dot)(a, b)\n\n");

    printf("#define sla_mul(a, b) _Generic((a), \\\n");
    printf("    sla_f32_dense_vec: sla_f32_dense_vec_mul, \\\n");
    printf("    sla_f64_dense_vec: sla_f64_dense_vec_mul, \\\n");
    printf("    sla_c32_dense_vec: sla_c32_dense_vec_mul, \\\n");
    printf("    sla_c64_dense_vec: sla_c64_dense_vec_mul, \\\n");

    printf("    sla_f32_dense_mat: _Generic((b), \\\n");
    printf("        sla_f32_dense_vec: sla_f32_dense_mat_vec_mul, \\\n");
    printf("        sla_f32_dense_mat: sla_f32_dense_mat_mul_mat, \\\n");
    printf("        default: sla_f32_dense_mat_mul), \\\n");
    printf("    sla_f64_dense_mat: _Generic((b), \\\n");
    printf("        sla_f64_dense_vec: sla_f64_dense_mat_vec_mul, \\\n");
    printf("        sla_f64_dense_mat: sla_f64_dense_mat_mul_mat, \\\n");
    printf("        default: sla_f64_dense_mat_mul), \\\n");
    printf("    sla_c32_dense_mat: _Generic((b), \\\n");
    printf("        sla_c32_dense_vec: sla_c32_dense_mat_vec_mul, \\\n");
    printf("        sla_c32_dense_mat: sla_c32_dense_mat_mul_mat, \\\n");
    printf("        default: sla_c32_dense_mat_mul), \\\n");
    printf("    sla_c64_dense_mat: _Generic((b), \\\n");
    printf("        sla_c64_dense_vec: sla_c64_dense_mat_vec_mul, \\\n");
    printf("        sla_c64_dense_mat: sla_c64_dense_mat_mul_mat, \\\n");
    printf("        default: sla_c64_dense_mat_mul), \\\n");

    printf("    sla_f32_csr_mat: _Generic((b), \\\n");
    printf("        sla_f32_dense_vec: sla_f32_csr_mat_vec_mul, \\\n");
    printf("        sla_f32_dense_mat: sla_f32_csr_mat_dense_mat_mul, \\\n");
    printf("        sla_f32_csr_mat: sla_f32_csr_mat_mul_csr_mat, \\\n");
    printf("        sla_f32_dia_mat: sla_f32_csr_mat_mul_dia_mat), \\\n");
    printf("    sla_f64_csr_mat: _Generic((b), \\\n");
    printf("        sla_f64_dense_vec: sla_f64_csr_mat_vec_mul, \\\n");
    printf("        sla_f64_dense_mat: sla_f64_csr_mat_dense_mat_mul, \\\n");
    printf("        sla_f64_csr_mat: sla_f64_csr_mat_mul_csr_mat, \\\n");
    printf("        sla_f64_dia_mat: sla_f64_csr_mat_mul_dia_mat), \\\n");
    printf("    sla_c32_csr_mat: _Generic((b), \\\n");
    printf("        sla_c32_dense_vec: sla_c32_csr_mat_vec_mul, \\\n");
    printf("        sla_c32_dense_mat: sla_c32_csr_mat_dense_mat_mul, \\\n");
    printf("        sla_c32_csr_mat: sla_c32_csr_mat_mul_csr_mat, \\\n");
    printf("        sla_c32_dia_mat: sla_c32_csr_mat_mul_dia_mat), \\\n");
    printf("    sla_c64_csr_mat: _Generic((b), \\\n");
    printf("        sla_c64_dense_vec: sla_c64_csr_mat_vec_mul, \\\n");
    printf("        sla_c64_dense_mat: sla_c64_csr_mat_dense_mat_mul, \\\n");
    printf("        sla_c64_csr_mat: sla_c64_csr_mat_mul_csr_mat, \\\n");
    printf("        sla_c64_dia_mat: sla_c64_csr_mat_mul_dia_mat), \\\n");

    printf("    sla_f32_dia_mat: _Generic((b), \\\n");
    printf("        sla_f32_dense_vec: sla_f32_dia_mat_vec_mul, \\\n");
    printf("        sla_f32_dense_mat: sla_f32_dia_mat_dense_mat_mul, \\\n");
    printf("        sla_f32_dia_mat: sla_f32_dia_mat_mul_dia_mat), \\\n");
    printf("    sla_f64_dia_mat: _Generic((b), \\\n");
    printf("        sla_f64_dense_vec: sla_f64_dia_mat_vec_mul, \\\n");
    printf("        sla_f64_dense_mat: sla_f64_dia_mat_dense_mat_mul, \\\n");
    printf("        sla_f64_dia_mat: sla_f64_dia_mat_mul_dia_mat), \\\n");
    printf("    sla_c32_dia_mat: _Generic((b), \\\n");
    printf("        sla_c32_dense_vec: sla_c32_dia_mat_vec_mul, \\\n");
    printf("        sla_c32_dense_mat: sla_c32_dia_mat_dense_mat_mul, \\\n");
    printf("        sla_c32_dia_mat: sla_c32_dia_mat_mul_dia_mat), \\\n");
    printf("    sla_c64_dia_mat: _Generic((b), \\\n");
    printf("        sla_c64_dense_vec: sla_c64_dia_mat_vec_mul, \\\n");
    printf("        sla_c64_dense_mat: sla_c64_dia_mat_dense_mat_mul, \\\n");
    printf("        sla_c64_dia_mat: sla_c64_dia_mat_mul_dia_mat)\n");
    printf(")(a, b)\n\n");

    printf("#endif // C11\n\n");

}

int main() {
    print_generic();
    return 0;
}

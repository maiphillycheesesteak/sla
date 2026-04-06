#ifndef SLA_H
#define SLA_H

/**
 * SLA - Single-Header Numerical Linear Algebra Library
 *
 * Supports:
 * - Dense and Sparse (CSR, DIA, COO) vectors and matrices.
 * - float, double, complex float, complex double types.
 * - OpenMP multithreading and GPU target offloading.
 * - C11 _Generic for a unified API.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Platform and Compiler Configurations
// ============================================================================

#ifdef _MSC_VER
    typedef _Fcomplex sla_c32;
    typedef _Dcomplex sla_c64;
    #define SLA_MAKE_C32(r, i) _FCbuild((float)(r), (float)(i))
    #define SLA_MAKE_C64(r, i) _Cbuild((double)(r), (double)(i))
    #define SLA_REAL_C32(c) crealf(c)
    #define SLA_IMAG_C32(c) cimagf(c)
    #define SLA_REAL_C64(c) creal(c)
    #define SLA_IMAG_C64(c) cimag(c)
    #define SLA_CONJ_C32(c) _FCbuild(crealf(c), -cimagf(c))
    #define SLA_CONJ_C64(c) _Cbuild(creal(c), -cimag(c))

    #define SLA_ADD_C32(a, b) _FCbuild(crealf(a) + crealf(b), cimagf(a) + cimagf(b))
    #define SLA_ADD_C64(a, b) _Cbuild(creal(a) + creal(b), cimag(a) + cimag(b))
    #define SLA_SUB_C32(a, b) _FCbuild(crealf(a) - crealf(b), cimagf(a) - cimagf(b))
    #define SLA_SUB_C64(a, b) _Cbuild(creal(a) - creal(b), cimag(a) - cimag(b))
    #define SLA_MUL_C32(a, b) _FCbuild(crealf(a)*crealf(b) - cimagf(a)*cimagf(b), crealf(a)*cimagf(b) + cimagf(a)*crealf(b))
    #define SLA_MUL_C64(a, b) _Cbuild(creal(a)*creal(b) - cimag(a)*cimag(b), creal(a)*cimag(b) + cimag(a)*creal(b))
    #define SLA_DIV_C32(a, b) _FCbuild((crealf(a)*crealf(b) + cimagf(a)*cimagf(b)) / (crealf(b)*crealf(b) + cimagf(b)*cimagf(b)), \
                                       (cimagf(a)*crealf(b) - crealf(a)*cimagf(b)) / (crealf(b)*crealf(b) + cimagf(b)*cimagf(b)))
    #define SLA_DIV_C64(a, b) _Cbuild((creal(a)*creal(b) + cimag(a)*cimag(b)) / (creal(b)*creal(b) + cimag(b)*cimag(b)), \
                                      (cimag(a)*creal(b) - creal(a)*cimag(b)) / (creal(b)*creal(b) + cimag(b)*cimag(b)))
#else
    typedef float _Complex sla_c32;
    typedef double _Complex sla_c64;
    #define SLA_MAKE_C32(r, i) ((float)(r) + (float)(i) * _Complex_I)
    #define SLA_MAKE_C64(r, i) ((double)(r) + (double)(i) * _Complex_I)
    #define SLA_REAL_C32(c) crealf(c)
    #define SLA_IMAG_C32(c) cimagf(c)
    #define SLA_REAL_C64(c) creal(c)
    #define SLA_IMAG_C64(c) cimag(c)
    #define SLA_CONJ_C32(c) conjf(c)
    #define SLA_CONJ_C64(c) conj(c)

    #define SLA_ADD_C32(a, b) ((a) + (b))
    #define SLA_ADD_C64(a, b) ((a) + (b))
    #define SLA_SUB_C32(a, b) ((a) - (b))
    #define SLA_SUB_C64(a, b) ((a) - (b))
    #define SLA_MUL_C32(a, b) ((a) * (b))
    #define SLA_MUL_C64(a, b) ((a) * (b))
    #define SLA_DIV_C32(a, b) ((a) / (b))
    #define SLA_DIV_C64(a, b) ((a) / (b))
#endif

typedef float sla_f32;
typedef double sla_f64;

#define SLA_REAL_F32(x) (x)
#define SLA_IMAG_F32(x) (0.0f)
#define SLA_REAL_F64(x) (x)
#define SLA_IMAG_F64(x) (0.0)

#define SLA_CONJ_F32(x) (x)
#define SLA_CONJ_F64(x) (x)

#define SLA_ADD_F32(a, b) ((a) + (b))
#define SLA_ADD_F64(a, b) ((a) + (b))
#define SLA_SUB_F32(a, b) ((a) - (b))
#define SLA_SUB_F64(a, b) ((a) - (b))
#define SLA_MUL_F32(a, b) ((a) * (b))
#define SLA_MUL_F64(a, b) ((a) * (b))
#define SLA_DIV_F32(a, b) ((a) / (b))
#define SLA_DIV_F64(a, b) ((a) / (b))

#define SLA_ZERO_F32 0.0f
#define SLA_ZERO_F64 0.0
#define SLA_ZERO_C32 SLA_MAKE_C32(0.0f, 0.0f)
#define SLA_ZERO_C64 SLA_MAKE_C64(0.0, 0.0)

// ============================================================================
// Data Structures
// ============================================================================

#define SLA_DECLARE_TYPES(TYPE, PREFIX) \
    /* Dense Vector */ \
    typedef struct { \
        TYPE *data; \
        size_t size; \
    } PREFIX##_dense_vec; \
    \
    /* Sparse Vector (COO) */ \
    typedef struct { \
        TYPE *values; \
        size_t *indices; \
        size_t nnz; \
        size_t size; \
    } PREFIX##_coo_vec; \
    \
    /* Dense Matrix */ \
    typedef struct { \
        TYPE *data; \
        size_t rows; \
        size_t cols; \
    } PREFIX##_dense_mat; \
    \
    /* CSR Matrix */ \
    typedef struct { \
        TYPE *values; \
        size_t *col_indices; \
        size_t *row_ptr; \
        size_t rows; \
        size_t cols; \
        size_t nnz; \
    } PREFIX##_csr_mat; \
    \
    /* DIA Matrix */ \
    typedef struct { \
        TYPE *data; \
        int *offsets; \
        size_t rows; \
        size_t cols; \
        size_t num_diags; \
    } PREFIX##_dia_mat;

SLA_DECLARE_TYPES(sla_f32, sla_f32)
SLA_DECLARE_TYPES(sla_f64, sla_f64)
SLA_DECLARE_TYPES(sla_c32, sla_c32)
SLA_DECLARE_TYPES(sla_c64, sla_c64)

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
// Dot Products & Multiplication Declarations
// ============================================================================

#define SLA_DECLARE_MUL_OPS(TYPE, PREFIX) \
    TYPE PREFIX##_dense_vec_dot(PREFIX##_dense_vec a, PREFIX##_dense_vec b); \
    PREFIX##_dense_vec PREFIX##_dense_mat_vec_mul(PREFIX##_dense_mat a, PREFIX##_dense_vec x); \
    PREFIX##_dense_mat PREFIX##_dense_mat_mul_mat(PREFIX##_dense_mat a, PREFIX##_dense_mat b); \
    PREFIX##_dense_vec PREFIX##_csr_mat_vec_mul(PREFIX##_csr_mat a, PREFIX##_dense_vec x); \
    PREFIX##_dense_vec PREFIX##_csr_mat_coo_vec_mul(PREFIX##_csr_mat a, PREFIX##_coo_vec x); \
    PREFIX##_dense_mat PREFIX##_csr_mat_dense_mat_mul(PREFIX##_csr_mat a, PREFIX##_dense_mat b); \
    PREFIX##_csr_mat PREFIX##_csr_mat_mul_csr_mat(PREFIX##_csr_mat a, PREFIX##_csr_mat b); \
    PREFIX##_dense_vec PREFIX##_dia_mat_vec_mul(PREFIX##_dia_mat a, PREFIX##_dense_vec x); \
    PREFIX##_dense_mat PREFIX##_dia_mat_dense_mat_mul(PREFIX##_dia_mat a, PREFIX##_dense_mat b); \
    PREFIX##_dia_mat PREFIX##_dia_mat_mul_dia_mat(PREFIX##_dia_mat a, PREFIX##_dia_mat b); \
    PREFIX##_csr_mat PREFIX##_csr_mat_mul_dia_mat(PREFIX##_csr_mat a, PREFIX##_dia_mat b);

SLA_DECLARE_MUL_OPS(sla_f32, sla_f32)
SLA_DECLARE_MUL_OPS(sla_f64, sla_f64)
SLA_DECLARE_MUL_OPS(sla_c32, sla_c32)
SLA_DECLARE_MUL_OPS(sla_c64, sla_c64)

// ============================================================================
// C11 _Generic Unified API Interface
// ============================================================================

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L

#define sla_alloc_dense_vec(type, size) _Generic(type, \
    sla_f32*: sla_f32_dense_vec_alloc, \
    sla_f64*: sla_f64_dense_vec_alloc, \
    sla_c32*: sla_c32_dense_vec_alloc, \
    sla_c64*: sla_c64_dense_vec_alloc)(size)

#define sla_alloc_coo_vec(type, size, nnz) _Generic(type, \
    sla_f32*: sla_f32_coo_vec_alloc, \
    sla_f64*: sla_f64_coo_vec_alloc, \
    sla_c32*: sla_c32_coo_vec_alloc, \
    sla_c64*: sla_c64_coo_vec_alloc)(size, nnz)

#define sla_alloc_dense_mat(type, rows, cols) _Generic(type, \
    sla_f32*: sla_f32_dense_mat_alloc, \
    sla_f64*: sla_f64_dense_mat_alloc, \
    sla_c32*: sla_c32_dense_mat_alloc, \
    sla_c64*: sla_c64_dense_mat_alloc)(rows, cols)

#define sla_alloc_csr_mat(type, rows, cols, nnz) _Generic(type, \
    sla_f32*: sla_f32_csr_mat_alloc, \
    sla_f64*: sla_f64_csr_mat_alloc, \
    sla_c32*: sla_c32_csr_mat_alloc, \
    sla_c64*: sla_c64_csr_mat_alloc)(rows, cols, nnz)

#define sla_alloc_dia_mat(type, rows, cols, num_diags) _Generic(type, \
    sla_f32*: sla_f32_dia_mat_alloc, \
    sla_f64*: sla_f64_dia_mat_alloc, \
    sla_c32*: sla_c32_dia_mat_alloc, \
    sla_c64*: sla_c64_dia_mat_alloc)(rows, cols, num_diags)

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
        sla_f32_coo_vec: sla_f32_csr_mat_coo_vec_mul, \
        sla_f32_dense_mat: sla_f32_csr_mat_dense_mat_mul, \
        sla_f32_csr_mat: sla_f32_csr_mat_mul_csr_mat, \
        sla_f32_dia_mat: sla_f32_csr_mat_mul_dia_mat, \
        default: sla_f32_csr_mat_vec_mul), \
    sla_f64_csr_mat: _Generic((b), \
        sla_f64_dense_vec: sla_f64_csr_mat_vec_mul, \
        sla_f64_coo_vec: sla_f64_csr_mat_coo_vec_mul, \
        sla_f64_dense_mat: sla_f64_csr_mat_dense_mat_mul, \
        sla_f64_csr_mat: sla_f64_csr_mat_mul_csr_mat, \
        sla_f64_dia_mat: sla_f64_csr_mat_mul_dia_mat, \
        default: sla_f64_csr_mat_vec_mul), \
    sla_c32_csr_mat: _Generic((b), \
        sla_c32_dense_vec: sla_c32_csr_mat_vec_mul, \
        sla_c32_coo_vec: sla_c32_csr_mat_coo_vec_mul, \
        sla_c32_dense_mat: sla_c32_csr_mat_dense_mat_mul, \
        sla_c32_csr_mat: sla_c32_csr_mat_mul_csr_mat, \
        sla_c32_dia_mat: sla_c32_csr_mat_mul_dia_mat, \
        default: sla_c32_csr_mat_vec_mul), \
    sla_c64_csr_mat: _Generic((b), \
        sla_c64_dense_vec: sla_c64_csr_mat_vec_mul, \
        sla_c64_coo_vec: sla_c64_csr_mat_coo_vec_mul, \
        sla_c64_dense_mat: sla_c64_csr_mat_dense_mat_mul, \
        sla_c64_csr_mat: sla_c64_csr_mat_mul_csr_mat, \
        sla_c64_dia_mat: sla_c64_csr_mat_mul_dia_mat, \
        default: sla_c64_csr_mat_vec_mul), \
    sla_f32_dia_mat: _Generic((b), \
        sla_f32_dense_vec: sla_f32_dia_mat_vec_mul, \
        sla_f32_dense_mat: sla_f32_dia_mat_dense_mat_mul, \
        sla_f32_dia_mat: sla_f32_dia_mat_mul_dia_mat, \
        default: sla_f32_dia_mat_vec_mul), \
    sla_f64_dia_mat: _Generic((b), \
        sla_f64_dense_vec: sla_f64_dia_mat_vec_mul, \
        sla_f64_dense_mat: sla_f64_dia_mat_dense_mat_mul, \
        sla_f64_dia_mat: sla_f64_dia_mat_mul_dia_mat, \
        default: sla_f64_dia_mat_vec_mul), \
    sla_c32_dia_mat: _Generic((b), \
        sla_c32_dense_vec: sla_c32_dia_mat_vec_mul, \
        sla_c32_dense_mat: sla_c32_dia_mat_dense_mat_mul, \
        sla_c32_dia_mat: sla_c32_dia_mat_mul_dia_mat, \
        default: sla_c32_dia_mat_vec_mul), \
    sla_c64_dia_mat: _Generic((b), \
        sla_c64_dense_vec: sla_c64_dia_mat_vec_mul, \
        sla_c64_dense_mat: sla_c64_dia_mat_dense_mat_mul, \
        sla_c64_dia_mat: sla_c64_dia_mat_mul_dia_mat, \
        default: sla_c64_dia_mat_vec_mul) \
)(a, b)

#endif // C11

#endif // SLA_H

#ifdef SLA_IMPLEMENTATION

// ============================================================================
// OpenMP Reductions for Complex Types
// ============================================================================

#ifdef _OPENMP
#ifdef _MSC_VER
#pragma omp declare reduction(+: sla_c32: omp_out = SLA_ADD_C32(omp_out, omp_in)) initializer(omp_priv = SLA_ZERO_C32)
#pragma omp declare reduction(+: sla_c64: omp_out = SLA_ADD_C64(omp_out, omp_in)) initializer(omp_priv = SLA_ZERO_C64)
#endif
#endif

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
        if (rows > 0 && cols > SIZE_MAX / rows) { \
            m.data = NULL; \
            return m; \
        } \
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
        if (num_diags > 0 && max_diag_len > SIZE_MAX / num_diags) { \
            m.data = NULL; \
            m.offsets = NULL; \
            return m; \
        } \
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
        _Pragma("omp target enter data map(to: v[0], v->data[0:v->size])") \
    } \
    void PREFIX##_dense_vec_update_from_device(PREFIX##_dense_vec *v) { \
        _Pragma("omp target update from(v->data[0:v->size])") \
    } \
    void PREFIX##_dense_vec_unmap_from_device(PREFIX##_dense_vec *v) { \
        _Pragma("omp target exit data map(from: v->data[0:v->size], v[0])") \
    } \
    void PREFIX##_coo_vec_map_to_device(PREFIX##_coo_vec *v) { \
        _Pragma("omp target enter data map(to: v[0], v->values[0:v->nnz], v->indices[0:v->nnz])") \
    } \
    void PREFIX##_coo_vec_update_from_device(PREFIX##_coo_vec *v) { \
        _Pragma("omp target update from(v->values[0:v->nnz])") \
    } \
    void PREFIX##_coo_vec_unmap_from_device(PREFIX##_coo_vec *v) { \
        _Pragma("omp target exit data map(from: v->values[0:v->nnz], v->indices[0:v->nnz], v[0])") \
    } \
    void PREFIX##_dense_mat_map_to_device(PREFIX##_dense_mat *m) { \
        _Pragma("omp target enter data map(to: m[0], m->data[0:m->rows*m->cols])") \
    } \
    void PREFIX##_dense_mat_update_from_device(PREFIX##_dense_mat *m) { \
        _Pragma("omp target update from(m->data[0:m->rows*m->cols])") \
    } \
    void PREFIX##_dense_mat_unmap_from_device(PREFIX##_dense_mat *m) { \
        _Pragma("omp target exit data map(from: m->data[0:m->rows*m->cols], m[0])") \
    } \
    void PREFIX##_csr_mat_map_to_device(PREFIX##_csr_mat *m) { \
        _Pragma("omp target enter data map(to: m[0], m->values[0:m->nnz], m->col_indices[0:m->nnz], m->row_ptr[0:m->rows+1])") \
    } \
    void PREFIX##_csr_mat_update_from_device(PREFIX##_csr_mat *m) { \
        _Pragma("omp target update from(m->values[0:m->nnz])") \
    } \
    void PREFIX##_csr_mat_unmap_from_device(PREFIX##_csr_mat *m) { \
        _Pragma("omp target exit data map(from: m->values[0:m->nnz], m->col_indices[0:m->nnz], m->row_ptr[0:m->rows+1], m[0])") \
    } \
    void PREFIX##_dia_mat_map_to_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        _Pragma("omp target enter data map(to: m[0], m->data[0:m->num_diags * max_diag_len], m->offsets[0:m->num_diags])") \
    } \
    void PREFIX##_dia_mat_update_from_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        _Pragma("omp target update from(m->data[0:m->num_diags * max_diag_len])") \
    } \
    void PREFIX##_dia_mat_unmap_from_device(PREFIX##_dia_mat *m) { \
        size_t max_diag_len = (m->rows < m->cols) ? m->rows : m->cols; \
        _Pragma("omp target exit data map(from: m->data[0:m->num_diags * max_diag_len], m->offsets[0:m->num_diags], m[0])") \
    }

SLA_IMPLEMENT_MEM_FUNCTIONS(sla_f32, sla_f32)
SLA_IMPLEMENT_MEM_FUNCTIONS(sla_f64, sla_f64)
SLA_IMPLEMENT_MEM_FUNCTIONS(sla_c32, sla_c32)
SLA_IMPLEMENT_MEM_FUNCTIONS(sla_c64, sla_c64)

// ============================================================================
// Element-wise Operations Implementations
// ============================================================================

#define SLA_IMPLEMENT_ELEM_OPS(TYPE, PREFIX, ADD_MACRO, SUB_MACRO, MUL_MACRO, DIV_MACRO) \
    PREFIX##_dense_vec PREFIX##_dense_vec_add(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(alloc: res.data[0:a.size])") \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_sub(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(alloc: res.data[0:a.size])") \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_mul(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(alloc: res.data[0:a.size])") \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_vec_div(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.size); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(alloc: res.data[0:a.size])") \
        for (size_t i = 0; i < a.size; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_add(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(alloc: res.data[0:n])") \
        for (size_t i = 0; i < n; ++i) res.data[i] = ADD_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_sub(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(alloc: res.data[0:n])") \
        for (size_t i = 0; i < n; ++i) res.data[i] = SUB_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_mul(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(alloc: res.data[0:n])") \
        for (size_t i = 0; i < n; ++i) res.data[i] = MUL_MACRO(a.data[i], b.data[i]); \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_div(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        size_t n = a.rows * a.cols; \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, a.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:n], b.data[0:n]) map(alloc: res.data[0:n])") \
        for (size_t i = 0; i < n; ++i) res.data[i] = DIV_MACRO(a.data[i], b.data[i]); \
        return res; \
    }

SLA_IMPLEMENT_ELEM_OPS(sla_f32, sla_f32, SLA_ADD_F32, SLA_SUB_F32, SLA_MUL_F32, SLA_DIV_F32)
SLA_IMPLEMENT_ELEM_OPS(sla_f64, sla_f64, SLA_ADD_F64, SLA_SUB_F64, SLA_MUL_F64, SLA_DIV_F64)
SLA_IMPLEMENT_ELEM_OPS(sla_c32, sla_c32, SLA_ADD_C32, SLA_SUB_C32, SLA_MUL_C32, SLA_DIV_C32)
SLA_IMPLEMENT_ELEM_OPS(sla_c64, sla_c64, SLA_ADD_C64, SLA_SUB_C64, SLA_MUL_C64, SLA_DIV_C64)

// ============================================================================
// Dot Products & Multiplication Implementations
// ============================================================================

#define SLA_IMPLEMENT_MUL_OPS(TYPE, PREFIX, ADD_MACRO, MUL_MACRO, ZERO, CONJ_MACRO) \
    TYPE PREFIX##_dense_vec_dot(PREFIX##_dense_vec a, PREFIX##_dense_vec b) { \
        TYPE sum = ZERO; \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.size], b.data[0:b.size]) map(tofrom: sum) reduction(+:sum)") \
        for (size_t i = 0; i < a.size; ++i) { \
            sum = ADD_MACRO(sum, MUL_MACRO(CONJ_MACRO(a.data[i]), b.data[i])); \
        } \
        return sum; \
    } \
    PREFIX##_dense_vec PREFIX##_dense_mat_vec_mul(PREFIX##_dense_mat a, PREFIX##_dense_vec x) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.data[0:a.rows*a.cols], x.data[0:x.size]) map(alloc: res.data[0:a.rows])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            TYPE sum = ZERO; \
            for (size_t j = 0; j < a.cols; ++j) { \
                sum = ADD_MACRO(sum, MUL_MACRO(a.data[i * a.cols + j], x.data[j])); \
            } \
            res.data[i] = sum; \
        } \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_csr_mat_coo_vec_mul(PREFIX##_csr_mat a, PREFIX##_coo_vec x) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.row_ptr[0:a.rows+1], a.col_indices[0:a.nnz], a.values[0:a.nnz], x.indices[0:x.nnz], x.values[0:x.nnz]) map(alloc: res.data[0:a.rows])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            TYPE sum = ZERO; \
            for (size_t j = a.row_ptr[i]; j < a.row_ptr[i+1]; ++j) { \
                size_t col = a.col_indices[j]; \
                for (size_t k = 0; k < x.nnz; ++k) { \
                    if (x.indices[k] == col) { \
                        sum = ADD_MACRO(sum, MUL_MACRO(a.values[j], x.values[k])); \
                        break; \
                    } \
                } \
            } \
            res.data[i] = sum; \
        } \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dense_mat_mul_mat(PREFIX##_dense_mat a, PREFIX##_dense_mat b) { \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for collapse(2) map(to: a.data[0:a.rows*a.cols], b.data[0:b.rows*b.cols]) map(alloc: res.data[0:a.rows*b.cols])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            for (size_t j = 0; j < b.cols; ++j) { \
                TYPE sum = ZERO; \
                for (size_t k = 0; k < a.cols; ++k) { \
                    sum = ADD_MACRO(sum, MUL_MACRO(a.data[i * a.cols + k], b.data[k * b.cols + j])); \
                } \
                res.data[i * res.cols + j] = sum; \
            } \
        } \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_csr_mat_vec_mul(PREFIX##_csr_mat a, PREFIX##_dense_vec x) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \
        PREFIX##_dense_vec_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for map(to: a.row_ptr[0:a.rows+1], a.col_indices[0:a.nnz], a.values[0:a.nnz], x.data[0:x.size]) map(alloc: res.data[0:a.rows])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            TYPE sum = ZERO; \
            for (size_t j = a.row_ptr[i]; j < a.row_ptr[i+1]; ++j) { \
                sum = ADD_MACRO(sum, MUL_MACRO(a.values[j], x.data[a.col_indices[j]])); \
            } \
            res.data[i] = sum; \
        } \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_csr_mat_dense_mat_mul(PREFIX##_csr_mat a, PREFIX##_dense_mat b) { \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        _Pragma("omp target teams distribute parallel for collapse(2) map(to: a.row_ptr[0:a.rows+1], a.col_indices[0:a.nnz], a.values[0:a.nnz], b.data[0:b.rows*b.cols]) map(alloc: res.data[0:a.rows*b.cols])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            for (size_t j = 0; j < b.cols; ++j) { \
                TYPE sum = ZERO; \
                for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \
                    sum = ADD_MACRO(sum, MUL_MACRO(a.values[k], b.data[a.col_indices[k] * b.cols + j])); \
                } \
                res.data[i * res.cols + j] = sum; \
            } \
        } \
        return res; \
    } \
    PREFIX##_csr_mat PREFIX##_csr_mat_mul_csr_mat(PREFIX##_csr_mat a, PREFIX##_csr_mat b) { \
        size_t *row_nnz = (size_t*)calloc(a.rows, sizeof(size_t)); \
        int max_threads = 1; \
        _Pragma("omp parallel") \
        { \
            _Pragma("omp single") \
            max_threads = omp_get_num_threads(); \
        } \
        int **thread_markers = (int**)calloc(max_threads, sizeof(int*)); \
        for(int t=0; t<max_threads; ++t) { \
            thread_markers[t] = (int*)calloc(b.cols, sizeof(int)); \
            for(size_t j=0; j<b.cols; ++j) thread_markers[t][j] = -1; \
        } \
        _Pragma("omp parallel for") \
        for (size_t i = 0; i < a.rows; ++i) { \
            int tid = omp_get_thread_num(); \
            int *marker = thread_markers[tid]; \
            size_t nnz_row = 0; \
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \
                size_t a_col = a.col_indices[k]; \
                for (size_t l = b.row_ptr[a_col]; l < b.row_ptr[a_col+1]; ++l) { \
                    size_t b_col = b.col_indices[l]; \
                    if (marker[b_col] != (int)i) { \
                        marker[b_col] = (int)i; \
                        nnz_row++; \
                    } \
                } \
            } \
            row_nnz[i] = nnz_row; \
        } \
        size_t total_nnz = 0; \
        for (size_t i = 0; i < a.rows; ++i) total_nnz += row_nnz[i]; \
        PREFIX##_csr_mat res = PREFIX##_csr_mat_alloc(a.rows, b.cols, total_nnz); \
        res.row_ptr[0] = 0; \
        for (size_t i = 0; i < a.rows; ++i) res.row_ptr[i+1] = res.row_ptr[i] + row_nnz[i]; \
        TYPE **thread_spas = (TYPE**)calloc(max_threads, sizeof(TYPE*)); \
        for(int t=0; t<max_threads; ++t) { \
            thread_spas[t] = (TYPE*)calloc(b.cols, sizeof(TYPE)); \
            for(size_t j=0; j<b.cols; ++j) thread_spas[t][j] = ZERO; \
            for(size_t j=0; j<b.cols; ++j) thread_markers[t][j] = -1; \
        } \
        _Pragma("omp parallel for") \
        for (size_t i = 0; i < a.rows; ++i) { \
            int tid = omp_get_thread_num(); \
            TYPE *spa = thread_spas[tid]; \
            int *marker = thread_markers[tid]; \
            size_t head = res.row_ptr[i]; \
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \
                size_t a_col = a.col_indices[k]; \
                TYPE a_val = a.values[k]; \
                for (size_t l = b.row_ptr[a_col]; l < b.row_ptr[a_col+1]; ++l) { \
                    size_t b_col = b.col_indices[l]; \
                    TYPE b_val = b.values[l]; \
                    spa[b_col] = ADD_MACRO(spa[b_col], MUL_MACRO(a_val, b_val)); \
                    if (marker[b_col] != (int)i) { \
                        marker[b_col] = (int)i; \
                        res.col_indices[head] = b_col; \
                        head++; \
                    } \
                } \
            } \
            for (size_t j = res.row_ptr[i]; j < res.row_ptr[i+1]; ++j) { \
                size_t col = res.col_indices[j]; \
                res.values[j] = spa[col]; \
                spa[col] = ZERO; /* Reset SPA for next row using it */ \
            } \
        } \
        for(int t=0; t<max_threads; ++t) { \
            free(thread_markers[t]); \
            free(thread_spas[t]); \
        } \
        free(thread_markers); \
        free(thread_spas); \
        free(row_nnz); \
        return res; \
    } \
    PREFIX##_dense_vec PREFIX##_dia_mat_vec_mul(PREFIX##_dia_mat a, PREFIX##_dense_vec x) { \
        PREFIX##_dense_vec res = PREFIX##_dense_vec_alloc(a.rows); \
        PREFIX##_dense_vec_map_to_device(&res); \
        size_t max_diag_len = (a.rows < a.cols) ? a.rows : a.cols; \
        _Pragma("omp target teams distribute parallel for map(to: a.offsets[0:a.num_diags], a.data[0:a.num_diags*max_diag_len], x.data[0:x.size]) map(alloc: res.data[0:a.rows])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            TYPE sum = ZERO; \
            for (size_t d = 0; d < a.num_diags; ++d) { \
                int j = (int)i + a.offsets[d]; \
                if (j >= 0 && j < (int)a.cols) { \
                    size_t diag_idx = d * max_diag_len + i; \
                    sum = ADD_MACRO(sum, MUL_MACRO(a.data[diag_idx], x.data[j])); \
                } \
            } \
            res.data[i] = sum; \
        } \
        return res; \
    } \
    PREFIX##_dense_mat PREFIX##_dia_mat_dense_mat_mul(PREFIX##_dia_mat a, PREFIX##_dense_mat b) { \
        PREFIX##_dense_mat res = PREFIX##_dense_mat_alloc(a.rows, b.cols); \
        PREFIX##_dense_mat_map_to_device(&res); \
        size_t max_diag_len = (a.rows < a.cols) ? a.rows : a.cols; \
        _Pragma("omp target teams distribute parallel for collapse(2) map(to: a.offsets[0:a.num_diags], a.data[0:a.num_diags*max_diag_len], b.data[0:b.rows*b.cols]) map(alloc: res.data[0:a.rows*b.cols])") \
        for (size_t i = 0; i < a.rows; ++i) { \
            for (size_t k = 0; k < b.cols; ++k) { \
                TYPE sum = ZERO; \
                for (size_t d = 0; d < a.num_diags; ++d) { \
                    int j = (int)i + a.offsets[d]; \
                    if (j >= 0 && j < (int)a.cols) { \
                        size_t diag_idx = d * max_diag_len + i; \
                        sum = ADD_MACRO(sum, MUL_MACRO(a.data[diag_idx], b.data[j * b.cols + k])); \
                    } \
                } \
                res.data[i * res.cols + k] = sum; \
            } \
        } \
        return res; \
    } \
    PREFIX##_dia_mat PREFIX##_dia_mat_mul_dia_mat(PREFIX##_dia_mat a, PREFIX##_dia_mat b) { \
        int *all_offsets = (int*)calloc(a.num_diags * b.num_diags, sizeof(int)); \
        size_t k = 0; \
        for(size_t i=0; i<a.num_diags; ++i) { \
            for(size_t j=0; j<b.num_diags; ++j) { \
                all_offsets[k++] = a.offsets[i] + b.offsets[j]; \
            } \
        } \
        /* Sort and unique offsets */ \
        for(size_t i=0; i<k; ++i) { \
            for(size_t j=i+1; j<k; ++j) { \
                if(all_offsets[i] > all_offsets[j]) { \
                    int tmp = all_offsets[i]; all_offsets[i] = all_offsets[j]; all_offsets[j] = tmp; \
                } \
            } \
        } \
        size_t unique_diags = 0; \
        if(k > 0) { \
            unique_diags = 1; \
            for(size_t i=1; i<k; ++i) { \
                if(all_offsets[i] != all_offsets[unique_diags-1]) { \
                    all_offsets[unique_diags++] = all_offsets[i]; \
                } \
            } \
        } \
        PREFIX##_dia_mat res = PREFIX##_dia_mat_alloc(a.rows, b.cols, unique_diags); \
        for(size_t i=0; i<unique_diags; ++i) res.offsets[i] = all_offsets[i]; \
        free(all_offsets); \
        size_t max_diag_len_a = (a.rows < a.cols) ? a.rows : a.cols; \
        size_t max_diag_len_b = (b.rows < b.cols) ? b.rows : b.cols; \
        size_t max_diag_len_res = (res.rows < res.cols) ? res.rows : res.cols; \
        _Pragma("omp parallel for") \
        for(size_t d=0; d<unique_diags; ++d) { \
            for(size_t r=0; r<res.rows; ++r) { \
                int c = (int)r + res.offsets[d]; \
                if(c >= 0 && c < (int)res.cols) { \
                    TYPE sum = ZERO; \
                    for(size_t d_a=0; d_a<a.num_diags; ++d_a) { \
                        for(size_t d_b=0; d_b<b.num_diags; ++d_b) { \
                            if(a.offsets[d_a] + b.offsets[d_b] == res.offsets[d]) { \
                                int k_idx = (int)r + a.offsets[d_a]; \
                                if(k_idx >= 0 && k_idx < (int)a.cols) { \
                                    TYPE val_a = a.data[d_a * max_diag_len_a + r]; \
                                    TYPE val_b = b.data[d_b * max_diag_len_b + k_idx]; \
                                    sum = ADD_MACRO(sum, MUL_MACRO(val_a, val_b)); \
                                } \
                            } \
                        } \
                    } \
                    res.data[d * max_diag_len_res + r] = sum; \
                } \
            } \
        } \
        return res; \
    } \
    PREFIX##_csr_mat PREFIX##_csr_mat_mul_dia_mat(PREFIX##_csr_mat a, PREFIX##_dia_mat b) { \
        size_t *row_nnz = (size_t*)calloc(a.rows, sizeof(size_t)); \
        int max_threads = 1; \
        _Pragma("omp parallel") \
        { \
            _Pragma("omp single") \
            max_threads = omp_get_num_threads(); \
        } \
        int **thread_markers = (int**)calloc(max_threads, sizeof(int*)); \
        for(int t=0; t<max_threads; ++t) { \
            thread_markers[t] = (int*)calloc(b.cols, sizeof(int)); \
            for(size_t j=0; j<b.cols; ++j) thread_markers[t][j] = -1; \
        } \
        _Pragma("omp parallel for") \
        for (size_t i = 0; i < a.rows; ++i) { \
            int tid = omp_get_thread_num(); \
            int *marker = thread_markers[tid]; \
            size_t nnz_row = 0; \
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \
                size_t a_col = a.col_indices[k]; \
                for(size_t d=0; d<b.num_diags; ++d) { \
                    int b_col = (int)a_col + b.offsets[d]; \
                    if(b_col >= 0 && b_col < (int)b.cols) { \
                        if(marker[b_col] != (int)i) { \
                            marker[b_col] = (int)i; \
                            nnz_row++; \
                        } \
                    } \
                } \
            } \
            row_nnz[i] = nnz_row; \
        } \
        size_t total_nnz = 0; \
        for(size_t i=0; i<a.rows; ++i) total_nnz += row_nnz[i]; \
        PREFIX##_csr_mat res = PREFIX##_csr_mat_alloc(a.rows, b.cols, total_nnz); \
        res.row_ptr[0] = 0; \
        for(size_t i=0; i<a.rows; ++i) res.row_ptr[i+1] = res.row_ptr[i] + row_nnz[i]; \
        TYPE **thread_spas = (TYPE**)calloc(max_threads, sizeof(TYPE*)); \
        for(int t=0; t<max_threads; ++t) { \
            thread_spas[t] = (TYPE*)calloc(b.cols, sizeof(TYPE)); \
            for(size_t j=0; j<b.cols; ++j) thread_spas[t][j] = ZERO; \
            for(size_t j=0; j<b.cols; ++j) thread_markers[t][j] = -1; \
        } \
        size_t max_diag_len_b = (b.rows < b.cols) ? b.rows : b.cols; \
        _Pragma("omp parallel for") \
        for (size_t i = 0; i < a.rows; ++i) { \
            int tid = omp_get_thread_num(); \
            TYPE *spa = thread_spas[tid]; \
            int *marker = thread_markers[tid]; \
            size_t head = res.row_ptr[i]; \
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i+1]; ++k) { \
                size_t a_col = a.col_indices[k]; \
                TYPE a_val = a.values[k]; \
                for(size_t d=0; d<b.num_diags; ++d) { \
                    int b_col = (int)a_col + b.offsets[d]; \
                    if(b_col >= 0 && b_col < (int)b.cols) { \
                        TYPE b_val = b.data[d * max_diag_len_b + a_col]; \
                        spa[b_col] = ADD_MACRO(spa[b_col], MUL_MACRO(a_val, b_val)); \
                        if(marker[b_col] != (int)i) { \
                            marker[b_col] = (int)i; \
                            res.col_indices[head] = b_col; \
                            head++; \
                        } \
                    } \
                } \
            } \
            for(size_t j=res.row_ptr[i]; j<res.row_ptr[i+1]; ++j) { \
                size_t col = res.col_indices[j]; \
                res.values[j] = spa[col]; \
                spa[col] = ZERO; \
            } \
        } \
        for(int t=0; t<max_threads; ++t) { \
            free(thread_markers[t]); \
            free(thread_spas[t]); \
        } \
        free(thread_markers); \
        free(thread_spas); \
        free(row_nnz); \
        return res; \
    }

SLA_IMPLEMENT_MUL_OPS(sla_f32, sla_f32, SLA_ADD_F32, SLA_MUL_F32, SLA_ZERO_F32, SLA_CONJ_F32)
SLA_IMPLEMENT_MUL_OPS(sla_f64, sla_f64, SLA_ADD_F64, SLA_MUL_F64, SLA_ZERO_F64, SLA_CONJ_F64)
SLA_IMPLEMENT_MUL_OPS(sla_c32, sla_c32, SLA_ADD_C32, SLA_MUL_C32, SLA_ZERO_C32, SLA_CONJ_C32)
SLA_IMPLEMENT_MUL_OPS(sla_c64, sla_c64, SLA_ADD_C64, SLA_MUL_C64, SLA_ZERO_C64, SLA_CONJ_C64)

#endif // SLA_IMPLEMENTATION

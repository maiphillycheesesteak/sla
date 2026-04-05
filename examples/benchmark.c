#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define SLA_IMPLEMENTATION
#include "../sla.h"

// Helper function to get current time in seconds
double get_time() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

void benchmark_dense_mat_mul(size_t n) {
    printf("Benchmarking Dense Matrix Multiplication (%zu x %zu)...\n", n, n);

    sla_f32_dense_mat a = sla_alloc_dense_mat((sla_f32*)NULL, n, n);
    sla_f32_dense_mat b = sla_alloc_dense_mat((sla_f32*)NULL, n, n);

    for (size_t i = 0; i < n * n; i++) {
        a.data[i] = (float)rand() / RAND_MAX;
        b.data[i] = (float)rand() / RAND_MAX;
    }

    sla_map_to_device(&a);
    sla_map_to_device(&b);

    double start = get_time();
    sla_f32_dense_mat c = sla_mul(a, b);
    double end = get_time();

    sla_update_from_device(&c);

    double time_taken = end - start;
    double flops = 2.0 * n * n * n; // 2n^3 FLOPs for matrix multiplication
    double gflops = (flops / time_taken) / 1e9;

    printf("  Time: %.4f s, Performance: %.2f GFLOP/s\n\n", time_taken, gflops);

    sla_free(&a);
    sla_free(&b);
    sla_free(&c);
}

void benchmark_csr_mat_dense_vec_mul(size_t n) {
    printf("Benchmarking CSR Matrix - Dense Vector Multiplication (%zu x %zu)...\n", n, n);

    // Create a tridiagonal-like CSR matrix for testing
    size_t nnz = 3 * n - 2;
    sla_f32_csr_mat a = sla_alloc_csr_mat((sla_f32*)NULL, n, n, nnz);

    size_t idx = 0;
    a.row_ptr[0] = 0;
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {
            a.values[idx] = (float)rand() / RAND_MAX;
            a.col_indices[idx] = i - 1;
            idx++;
        }
        a.values[idx] = (float)rand() / RAND_MAX;
        a.col_indices[idx] = i;
        idx++;
        if (i < n - 1) {
            a.values[idx] = (float)rand() / RAND_MAX;
            a.col_indices[idx] = i + 1;
            idx++;
        }
        a.row_ptr[i + 1] = idx;
    }

    sla_f32_dense_vec x = sla_alloc_dense_vec((sla_f32*)NULL, n);
    for (size_t i = 0; i < n; i++) {
        x.data[i] = (float)rand() / RAND_MAX;
    }

    sla_map_to_device(&a);
    sla_map_to_device(&x);

    double start = get_time();
    sla_f32_dense_vec y = sla_mul(a, x);
    double end = get_time();

    sla_update_from_device(&y);

    double time_taken = end - start;
    double flops = 2.0 * nnz;
    double gflops = (flops / time_taken) / 1e9;

    printf("  Time: %.6f s, Performance: %.6f GFLOP/s\n\n", time_taken, gflops);

    sla_free(&a);
    sla_free(&x);
    sla_free(&y);
}

void benchmark_dense_vec_dot(size_t n) {
    printf("Benchmarking Dense Vector Dot Product (size %zu)...\n", n);

    sla_f32_dense_vec a = sla_alloc_dense_vec((sla_f32*)NULL, n);
    sla_f32_dense_vec b = sla_alloc_dense_vec((sla_f32*)NULL, n);

    for (size_t i = 0; i < n; i++) {
        a.data[i] = (float)rand() / RAND_MAX;
        b.data[i] = (float)rand() / RAND_MAX;
    }

    sla_map_to_device(&a);
    sla_map_to_device(&b);

    double start = get_time();
    sla_f32 dot_product = sla_dot(a, b);
    double end = get_time();

    // Suppress unused variable warning
    (void)dot_product;

    double time_taken = end - start;
    double flops = 2.0 * n;
    double gflops = (flops / time_taken) / 1e9;

    printf("  Time: %.6f s, Performance: %.4f GFLOP/s\n\n", time_taken, gflops);

    sla_free(&a);
    sla_free(&b);
}

void benchmark_csr_mat_coo_vec_mul(size_t n) {
    printf("Benchmarking CSR Matrix - Sparse Vector (COO) Multiplication (%zu x %zu)...\n", n, n);

    // Create a tridiagonal-like CSR matrix for testing
    size_t nnz_mat = 3 * n - 2;
    sla_f32_csr_mat a = sla_alloc_csr_mat((sla_f32*)NULL, n, n, nnz_mat);

    size_t idx = 0;
    a.row_ptr[0] = 0;
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {
            a.values[idx] = (float)rand() / RAND_MAX;
            a.col_indices[idx] = i - 1;
            idx++;
        }
        a.values[idx] = (float)rand() / RAND_MAX;
        a.col_indices[idx] = i;
        idx++;
        if (i < n - 1) {
            a.values[idx] = (float)rand() / RAND_MAX;
            a.col_indices[idx] = i + 1;
            idx++;
        }
        a.row_ptr[i + 1] = idx;
    }

    // Create a very sparse COO vector (e.g. 1% density)
    size_t nnz_vec = n / 100;
    if (nnz_vec == 0) nnz_vec = 1;
    sla_f32_coo_vec x = sla_alloc_coo_vec((sla_f32*)NULL, n, nnz_vec);
    for (size_t i = 0; i < nnz_vec; i++) {
        x.indices[i] = (rand() % n); // Simplistic random indices (might have duplicates but OK for benchmarking)
        x.values[i] = (float)rand() / RAND_MAX;
    }

    sla_map_to_device(&a);
    sla_map_to_device(&x);

    double start = get_time();
    sla_f32_dense_vec y = sla_mul(a, x);
    double end = get_time();

    sla_update_from_device(&y);

    double time_taken = end - start;
    // Number of FLOPs depends on intersection, max is 2 * nnz_vec (actually roughly 2 * nnz_vec * 3 since each row has ~3 elements)
    // Approximate FLOPs as 2 * nnz_vec * 3 (avg row length)
    double flops = 6.0 * nnz_vec;
    double gflops = (flops / time_taken) / 1e9;

    printf("  Time: %.6f s, Performance: %.6f GFLOP/s\n\n", time_taken, gflops);

    sla_free(&a);
    sla_free(&x);
    sla_free(&y);
}

int main() {
    srand(12345);

    size_t N = 100;      // N for dense mat mul
    size_t N_large = 10000; // N for vector operations

    benchmark_dense_mat_mul(N);
    benchmark_csr_mat_dense_vec_mul(N_large);
    benchmark_dense_vec_dot(N_large);
    benchmark_csr_mat_coo_vec_mul(N_large);

    return 0;
}

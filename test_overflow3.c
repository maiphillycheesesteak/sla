#define SLA_IMPLEMENTATION
#include "sla.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
    size_t rows = 4;
    size_t cols = 4;
    size_t num_diags = (SIZE_MAX / sizeof(sla_f32)) / 4 + 2;

    // max_diag_len is 4.
    // num_diags * 4 will overflow size_t?
    // Let's force an overflow of `num_diags * max_diag_len`.
    // size_t wrap around: (num_diags * 4) = small number
    num_diags = (SIZE_MAX / 4) + 2;
    printf("Attempting to allocate with num_diags=%zu\n", num_diags);
    // 4 * ((SIZE_MAX / 4) + 2) = SIZE_MAX - (SIZE_MAX % 4) + 8
    // If SIZE_MAX is 2^64-1 = ...3, then SIZE_MAX % 4 = 3
    // So 4 * ((SIZE_MAX / 4) + 2) = SIZE_MAX - 3 + 8 = SIZE_MAX + 5 = 4 (wrap around to 4)
    // Then calloc(4, sizeof(float)) will succeed!
    // BUT offsets will be calloc(num_diags, sizeof(int))
    // Wait, calloc(num_diags, sizeof(int)) will fail because num_diags is huge!
    // To make calloc(num_diags, sizeof(int)) also wrap around or be small, we'd need a different bug or we can just see if we can trigger the issue.
    // Wait, the vulnerable code is:
    // m.data = (TYPE*)calloc(num_diags * max_diag_len, sizeof(TYPE));
    // m.offsets = (int*)calloc(num_diags, sizeof(int));

    // If we just check for overflow in `num_diags * max_diag_len`, we can prevent undefined behavior.
    return 0;
}

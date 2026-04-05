#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main() {
    size_t num_diags = SIZE_MAX / sizeof(int) + 2;
    printf("num_diags=%zu\n", num_diags);
    int *offsets = calloc(num_diags, sizeof(int));
    if (offsets == NULL) {
        printf("calloc handles overflow correctly for sizes.\n");
    }
    return 0;
}

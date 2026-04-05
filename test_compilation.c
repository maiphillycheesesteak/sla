#define SLA_IMPLEMENTATION
#include "sla.h"

int main() {
    sla_f32_dense_vec a = sla_alloc_dense_vec((sla_f32*)NULL, 10);
    sla_f32_dense_vec b = sla_alloc_dense_vec((sla_f32*)NULL, 10);

    for(int i=0; i<10; i++) {
        a.data[i] = 1.0f;
        b.data[i] = 2.0f;
    }

    sla_map_to_device(&a);
    sla_map_to_device(&b);

    sla_f32_dense_vec c = sla_add(a, b);

    sla_update_from_device(&c);

    printf("Result[0] = %f\n", c.data[0]);

    sla_free(&a);
    sla_free(&b);
    sla_free(&c);

    // Test complex
    sla_c32_dense_vec ca = sla_alloc_dense_vec((sla_c32*)NULL, 10);
    sla_c32_dense_vec cb = sla_alloc_dense_vec((sla_c32*)NULL, 10);
    sla_c32 c_res = sla_dot(ca, cb);
    sla_free(&ca);
    sla_free(&cb);
    return 0;
}

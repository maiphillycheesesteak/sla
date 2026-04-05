#ifndef DUMMY_OMP_H
#define DUMMY_OMP_H
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
#endif

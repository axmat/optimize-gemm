#ifndef GEMM_INIT
#define GEMM_INIT

#include <cstddef>

static void init_array(const size_t m, const size_t n, float *x) {
   for (size_t i = 0; i < m * n; i++) {
      x[i] = 0.;
   }
}

#endif

#ifndef GEMM_BLAS
#define GEMM_BLAS

#include "init.hxx"

extern "C" void sgemm_(const char *transA, const char *transB, const int *m,
                       const int *n, const int *k, const float *alpha,
                       const float *a, const int *lda, const float *b,
                       const int *ldb, const float *beta, float *c,
                       const int *ldc);

static void gemm_blas(const int m, const int n, const int k, const float *a,
                      const float *b, float *c) {
   init_array(m, n, c);
   const char trans_a = 'N';
   const char trans_b = 'N';
   const float alpha = 1.;
   const float beta = 0.;
   sgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
};

#endif

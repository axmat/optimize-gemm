#ifndef GEMM_AUTOVEC
#define GEMM_AUTOVEC

#include "init.hxx"
#include <cstddef>

// Autovectorized gemm
static void gemm_autovec(const size_t m, const size_t n, const size_t k,
                         const float *a, const float *b, float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         float y = 0.;
#pragma clang loop vectorize(enable)
         for (size_t l = 0; l < k; l++) {
            y += a[i * k + l] * tb[j * n + l];
         }
         c[i * n + j] = y;
      }
   }

   delete[] tb;
}

// Autovectorized Blocked gemmm
static void gemm_blocked_autovec(const size_t m, const size_t n, const size_t k,
                                 const size_t block_size, const float *a,
                                 const float *b, float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   for (size_t i = 0; i < m; i += block_size) {
      for (size_t ii = i; ii < i + block_size; ii++) {
         for (size_t j = 0; j < n; j += block_size) {
            for (size_t jj = j; jj < j + block_size; jj++) {
               float y = 0.;
#pragma clang loop vectorize(enable)
               for (size_t l = 0; l < k; l++) {
                  y += a[ii * k + l] * tb[jj * n + l];
               }
               c[ii * n + jj] = y;
            }
         }
      }
   }

   delete[] tb;
}

// Autovectorized tiled gemm
static void gemm_tiled_autovec(const size_t m, const size_t n, const size_t k,
                               const size_t tile_m, size_t tile_n,
                               size_t tile_k, const float *a, const float *b,
                               float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   for (size_t i = 0; i < m; i += tile_m) {
      for (size_t ii = i; ii < i + tile_m; ii++) {
         for (size_t j = 0; j < n; j += tile_n) {
            for (size_t jj = j; jj < j + tile_n; jj++) {
               float x = 0.;
               for (size_t l = 0; l < k; l += tile_k) {
#pragma clang loop vectorize(enable)
                  for (size_t ll = 0; ll < tile_k; ll++) {
                     x += a[ii * k + l + ll] * tb[jj * n + l + ll];
                  }
               }
               c[ii * n + jj] = x;
            }
         }
      }
   }

   delete[] tb;
}

#endif

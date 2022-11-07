#ifndef GEMM_REFRENCE
#define GEMM_REFRENCE

#include "init.hxx"

#include <iostream>
#include <type_traits>
#include <vector>

// Reference Gemm mnk
// a of dim m x k, b of dim k x n and c of dim m x k
// c[m,n] = sum(a[i,k] * b[k,n])
static void gemm_ref(const size_t m, const size_t n, const size_t k,
                     const float *a, const float *b, float *c) {

   init_array(m, n, c);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         for (size_t l = 0; l < k; l++) {
            c[i * n + j] += a[i * k + l] * b[l * n + j];
         }
      }
   }
}

static void gemm_ref_transposed(const size_t m, const size_t n, const size_t k,
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
         for (size_t l = 0; l < k; l++) {
            c[i * n + j] += a[i * k + l] * tb[j * n + l];
         }
      }
   }
   delete[] tb;
}

// Blocked gemm
static void gemm_blocked(const size_t m, const size_t n, const size_t k,
                         const size_t block_size, const float *a,
                         const float *b, float *c) {
   init_array(m, n, c);
   for (size_t i = 0; i < m; i += block_size) {
      for (size_t ii = i; ii < i + block_size; ii++) {
         for (size_t l = 0; l < k; l++) {
            for (size_t j = 0; j < n; j += block_size) {
               for (size_t jj = j; jj < j + block_size; jj++) {
                  c[ii * n + jj] += a[ii * k + l] * b[l * n + jj];
               }
            }
         }
      }
   }
}
// jj and l of size vlen

// Tiled gemm
static void gemm_tiled(const size_t m, const size_t n, const size_t k,
                       const size_t tile_m, size_t tile_n, size_t tile_k,
                       const float *a, const float *b, float *c) {

   init_array(m, n, c);

   for (size_t i = 0; i < m; i += tile_m) {
      for (size_t ii = i; ii < i + tile_m; ii++) {
         for (size_t l = 0; l < k; l += tile_k) {
            for (size_t ll = l; ll < l + tile_k; ll++) {
               for (size_t j = 0; j < n; j += tile_n) {
                  for (size_t jj = j; jj < j + tile_n; jj++) {
                     c[ii * n + jj] += a[ii * k + ll] * b[ll * n + jj];
                  }
               }
            }
         }
      }
   }
}

#endif

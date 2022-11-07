#ifndef GEMM_STD
#define GEMM_STD

#include <experimental/simd>

// std::experimental::simd gemm
template <size_t vlen>
static void gemm_std(const size_t m, const size_t n, const size_t k,
                     const float *a, const float *b, float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   using vector_type = std::experimental::fixed_size_simd<float, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         // Initialize to zero
         vector_type c_vec{};
         for (size_t l = 0; l < k / vlen; l++) {
            //  Load a[ii * k + l * vlen] of size vlen
            vector_type a_vec;
            size_t a_offset = i * k + l * vlen;
            a_vec.copy_from(a + a_offset, alignment{});
            // Load b[jj * n + l * vlen] of size vlen
            size_t tb_offset = j * n + l * vlen;
            vector_type tb_vec;
            tb_vec.copy_from(tb + tb_offset, alignment{});
            // FMA
            c_vec += a_vec * tb_vec;
         }
         // Reduce and store in c
         c[i * n + j] = std::experimental::reduce(c_vec, std::plus<>());
         if (k % vlen > 0) {
            for (size_t l = vlen * (k / vlen); l < k; l++) {
               c[i * n + j] += a[i * k + l] * b[j * n + l];
            }
         }
      }
   }

   delete[] tb;
}

// std::experimental::simd blocked gemm
template <size_t vlen>
static void gemm_blocked_std(const size_t m, const size_t n, const size_t k,
                             const size_t block_size, const float *a,
                             const float *b, float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   using vector_type = std::experimental::fixed_size_simd<float, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < m; i += block_size) {
      for (size_t ii = i; ii < i + block_size; ii++) {
         for (size_t j = 0; j < n; j += block_size) {
            for (size_t jj = j; jj < j + block_size; jj++) {
               vector_type c_vec{};
               for (size_t l = 0; l < k / vlen; l++) {
                  // Load a[ii * k + l * vlen] of size vlen
                  vector_type a_vec;
                  size_t a_offset = ii * k + l * vlen;
                  a_vec.copy_from(a + a_offset, alignment{});
                  // Load b[jj * n + l * vlen] of size vlen
                  size_t tb_offset = jj * n + l * vlen;
                  vector_type tb_vec;
                  tb_vec.copy_from(tb + tb_offset, alignment{});
                  // FMA
                  c_vec += a_vec * tb_vec;
               }
               // Reduce and store in c
               c[ii * n + jj] = std::experimental::reduce(c_vec, std::plus<>());
            }
         }
      }
   }

   delete[] tb;
}

// Vectorized tiled gemm
template <size_t vlen>
static void gemm_tiled_std(const size_t m, const size_t n, const size_t k,
                           const size_t tile_m, size_t tile_n, size_t tile_k,
                           const float *a, const float *b, float *c) {

   init_array(m, n, c);

   float *tb = new float[n * k];
   for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < k; j++) {
         tb[i * k + j] = b[j * k + i];
      }
   }

   using vector_type = std::experimental::fixed_size_simd<float, vlen>;
   using alignment = std::experimental::element_aligned_tag;

   for (size_t i = 0; i < m; i += tile_m) {
      for (size_t ii = i; ii < i + tile_m; ii++) {
         for (size_t j = 0; j < n; j += tile_n) {
            for (size_t jj = j; jj < j + tile_n; jj++) {
               vector_type x{};
               for (size_t l = 0; l < k; l += tile_k) {
                  for (size_t ll = 0; ll < tile_k / vlen; ll++) {
                     // Load a[ii * k + l + ll * vlen] of size vlen
                     vector_type a_vec;
                     size_t a_offset = ii * k + l + ll * vlen;
                     a_vec.copy_from(a + a_offset, alignment{});
                     // Load b[jj * n + l + ll * vlen] of size vlen
                     size_t tb_offset = jj * n + l + ll * vlen;
                     vector_type tb_vec;
                     tb_vec.copy_from(tb + tb_offset, alignment{});
                     // FMA
                     x += a_vec * tb_vec;
                  }
                  // Reduce and store in c
                  c[ii * n + jj] = std::experimental::reduce(x, std::plus<>());
                  if (tile_k % vlen > 0) {
                     for (size_t ll = vlen * (tile_k / vlen); ll < tile_k;
                          ll++) {
                        c[ii * n + jj] +=
                            a[ii * k + l + ll] * b[jj * n + l + ll];
                     }
                  }
               }
            }
         }
      }
   }

   delete[] tb;
}

#endif

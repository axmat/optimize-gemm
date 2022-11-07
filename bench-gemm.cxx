#include <fstream>
#include <iostream>
#include <nanobench.h>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>

#include "gemm.hxx"
#include "gemm_autovec.hxx"
#include "gemm_blas.hxx"
#include "gemm_std.hxx"

bool are_close(const size_t m, const size_t n, const float *a, const float *b,
               const float eps) {
   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         if (std::abs(a[i] - b[i]) > eps) {
            return false;
         }
      }
   }
   return true;
}

void test_output(const size_t m, const size_t n, const float *a, const float *b,
                 const float eps = 1e-3) {
   if (!are_close(m, n, a, b, eps)) {
      std::cerr << "Different output from reference" << std::endl;
   }
}

// Generate tiles sizes for a square matrix of size n x n
// For example for n = 512, the generated tile sizes are [4, 8, 16, 32, 64, 128]
std::vector<size_t> generate_tiles(const size_t n) {
   const size_t max_n = (size_t)log2(n / 4);
   // tile_m = 8...2^(max_m -1)
   std::vector<size_t> tiles(max_n - 2);
   tiles[0] = 8;
   for (size_t i = 1; i < max_n - 2; i++) {
      tiles[i] = 2 * tiles[i - 1];
   }
   return tiles;
}

int main(int argc, char **argv) {

   if (argc != 2) {
      std::cerr << "./main dim_size" << std::endl;
      return 1;
   }
   if (sizeof(float) != 4) {
      std::cerr << "Not supported for sizeof(float)!=4" << std::endl;
      return 1;
   }

   size_t matrix_dim = std::stoi(std::string(argv[1]));
   size_t align = 8 * sizeof(float);
   // Generate the tiles sizes
   auto tiles = generate_tiles(matrix_dim);

   std::cout << "Square matrices of size " << matrix_dim << "x" << matrix_dim
             << std::endl;
   std::cout << "Tile sizes = {";
   for (auto t = tiles.begin(); t != tiles.end(); t++) {
      std::cout << *t << (next(t) == tiles.end() ? "}\n" : ", ");
   }

   const size_t m = matrix_dim;
   const size_t n = matrix_dim;
   const size_t k = matrix_dim;

   float *a = (float *)aligned_alloc(align, m * k * sizeof(float));
   float *b = (float *)aligned_alloc(align, k * n * sizeof(float));
   float *c = (float *)aligned_alloc(align, m * n * sizeof(float));
   float *c_ref = (float *)aligned_alloc(align, m * n * sizeof(float));

   // initialize a and b
   for (size_t i = 0; i < m * k; i++) {
      a[i] = float(i) / float(m * k);
   }
   for (size_t i = 0; i < k * n; i++) {
      b[i] = float(i) / float(k * n);
   }

   ankerl::nanobench::Bench bench;
   bench.title("TITLE");

   // Reference gemm
   bench.run("ref",
                                  [&] { gemm_ref(m, n, k, a, b, c_ref); });

   // Reference gemm with transposed b
   bench.run("tr", [&] { gemm_ref_transposed(m, n, k, a, b, c); });
   test_output(m, n, c, c_ref);

   // Blocked gemm
   for (size_t block_size : tiles) {
      bench.run("blocked_" + std::to_string(block_size),
                [&] { gemm_blocked(m, n, k, block_size, a, b, c); });
      test_output(m, n, c, c_ref);
   }

   // Tiled gemm
   for (size_t tile_size : tiles) {
      bench.run("tiled_" + std::to_string(tile_size) + "x" +
                    std::to_string(tile_size) + "x" + std::to_string(tile_size),
                [&] {
                   gemm_tiled(m, n, k, tile_size, tile_size, tile_size, a, b,
                              c);
                });
      test_output(m, n, c, c_ref);
   }

   // std vectorized gemm
   bench.run("std vlen_" + std::to_string(4),
             [&] { gemm_std<4>(m, n, k, a, b, c); });
   test_output(m, n, c, c_ref);

   bench.run("std vlen_" + std::to_string(8),
             [&] { gemm_std<8>(m, n, k, a, b, c); });
   test_output(m, n, c, c_ref);

   // std vectorized blocked gemm
   for (size_t block_size : tiles) {
      bench.run("std vlen_4 blocked_" + std::to_string(block_size),
                [&] { gemm_blocked_std<4>(m, n, k, block_size, a, b, c); });
      test_output(m, n, c, c_ref);
   }
   for (size_t block_size : tiles) {
      bench.run("std vlen_8 blocked_" + std::to_string(block_size),
                [&] { gemm_blocked_std<8>(m, n, k, block_size, a, b, c); });
      test_output(m, n, c, c_ref);
   }

   // std vectorized tiled gemm
   for (size_t tile_size : tiles) {
      bench.run("std vlen_4 tiled_" + std::to_string(tile_size) + "x" +
                    std::to_string(tile_size) + "x" + std::to_string(tile_size),
                [&] {
                   gemm_tiled_std<4>(m, n, k, tile_size, tile_size, tile_size,
                                     a, b, c);
                });
      test_output(m, n, c, c_ref);
   }

   for (size_t tile_size : tiles) {
      bench.run("std vlen_8 tiled_" + std::to_string(tile_size) + "x" +
                    std::to_string(tile_size) + "x" + std::to_string(tile_size),
                [&] {
                   gemm_tiled_std<8>(m, n, k, tile_size, tile_size, tile_size,
                                     a, b, c);
                });
      test_output(m, n, c, c_ref);
   }

   // Autovectorization
   bench.run("autovec", [&] { gemm_autovec(m, n, k, a, b, c); });
   test_output(m, n, c, c_ref);

   // Autovectorized blocked gemm
   for (size_t block_size : tiles) {
      bench.run("autovec blocked_" + std::to_string(block_size),
                [&] { gemm_blocked_autovec(m, n, k, block_size, a, b, c); });
      test_output(m, n, c, c_ref);
   }

   // Autovectorized tiled gemm
   for (size_t tile_size : tiles) {
      bench.run("autovec tiled_" + std::to_string(tile_size) + "x" +
                    std::to_string(tile_size) + "x" + std::to_string(tile_size),
                [&] {
                   gemm_tiled_autovec(m, n, k, tile_size, tile_size, tile_size,
                                      a, b, c);
                });
      test_output(m, n, c, c_ref);
   }

   // Openblas
   bench.minEpochIterations(10).run("blas",
                                    [&] { gemm_blas(m, n, k, a, b, c); });
   test_output(m, n, c, c_ref);

   // save
   std::ofstream file("benchmark" + std::string(argv[1]) + ".csv");
   ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, file);

   // Free
   free(a);
   free(b);
   free(c);
   free(c_ref);
}

#include "gemm_autovec.hxx"
#include "init.hxx"

// Do not optimize the main function, otherwise the compiler will remove the
// call to gemm_autovec
[[clang::optnone]] int main() {
   size_t m = 256;
   size_t k = 256;
   size_t n = 256;

   float *a = new float[m * k];
   float *b = new float[k * n];
   float *c = new float[m * n];

   // initialize
   for (size_t i = 0; i < m * k; i++) {
      a[i] = float(i) / float(m * k);
   }
   for (size_t i = 0; i < k * n; i++) {
      b[i] = float(i) / float(k * n);
   }

   init_array(m, n, c);

   gemm_autovec(m, n, k, a, b, c);

   // Free
   delete[] a;
   delete[] b;
   delete[] c;
}

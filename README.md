Optimize single-threaded General Matrix Multiplication (GEMM) fo two square matrices

## Transpose the second matrix
![](figures/gemm_reference_transpose.png)

## Blocking
![](figures/gemm_blocked.png)

## Tiling
![](figures/gemm_tiled.png)

## Vectorization with std::experimental::simd
![](figures/gemm_std.png)

## Automatic vectorization on Clang
![](figures/autovec.png)

## OpenBLAS
![](figures/gemm.png)

## Build

```
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++
cmake --build . --
```

## Run the benchmarks
```
export $OMP_NUM_THREADS=1
./bench-gemm [dim_size]
```

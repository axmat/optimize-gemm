cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 20)

project(optimize-gemm LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-fvectorize -O2 -march=native")

include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.1.0
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

find_package(BLAS)

message("BLAS Vendor: ${BLA_VENDOR}")
message("BLAS Libraries: ${BLAS_LIBRARIES}")

add_executable(bench-gemm bench-gemm.cxx)
target_link_libraries(bench-gemm nanobench ${BLAS_LIBRARIES})

# Test auto vectorization
add_custom_target(
	test-autovec
	ALL
	COMMAND ${CMAKE_CXX_COMPILER} -fvectorize -fno-discard-value-names -O2 -S -emit-llvm -o ${CMAKE_CURRENT_BINARY_DIR}/test-autovec.ll ${CMAKE_CURRENT_SOURCE_DIR}/test-autovec.cxx
	COMMENT "Compiling test-autovec.cxx to llvm ir"
	VERBATIM
)


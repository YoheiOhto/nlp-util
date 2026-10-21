cmake_minimum_required(VERSION 3.18)
project(dump_hidden CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Links against the static llama.cpp install inside llamacpp.sif (/opt/llama.cpp,
# v0.7.0 commit 620d2225cc436ab0b001e3200e590e3b286ef9ab).
find_package(llama REQUIRED HINTS /opt/llama.cpp/lib/cmake/llama)
# The installed ggml-config.cmake has GGML_STATIC=OFF although the archives are
# static (libggml-cuda.a), so it does not propagate the CUDA runtime / cuBLAS
# dependencies. Add them explicitly (shared CUDA libs from the same container).
# libggml-cuda.a is built with GGML_CUDA_NCCL=ON, so NCCL is needed as well.
find_package(CUDAToolkit REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
add_executable(dump_hidden dump_hidden.cpp)
target_link_libraries(dump_hidden PRIVATE llama CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver ${NCCL_LIBRARY})
# CMake places these shared libs before the static ggml archives on the link line;
# Ubuntu's default --as-needed would then drop them, so keep them explicitly.
target_link_options(dump_hidden PRIVATE "LINKER:--no-as-needed")

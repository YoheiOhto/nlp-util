cmake_minimum_required(VERSION 3.18)
project(dump_hidden CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Links against the static llama.cpp install inside llamacpp.sif (/opt/llama.cpp).
find_package(llama REQUIRED HINTS /opt/llama.cpp/lib/cmake/llama)
add_executable(dump_hidden dump_hidden.cpp)
target_link_libraries(dump_hidden PRIVATE llama)

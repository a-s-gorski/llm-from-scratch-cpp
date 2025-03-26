cmake_minimum_required(VERSION 3.10)
project(llm_fs)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/lib)
add_subdirectory(src/app)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE llm_fs_lib)
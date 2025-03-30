cmake_minimum_required(VERSION 3.10)
project(llm_fs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED COMPONENTS filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("${TORCH_INCLUDE_DIRS}")
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=llm_fs)

add_subdirectory(src/lib)
add_subdirectory(src/app)

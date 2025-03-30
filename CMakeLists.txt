cmake_minimum_required(VERSION 3.10)
project(llm_fs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix: Use correct Boost component name (no space)
find_package(Boost REQUIRED COMPONENTS filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

# Fix: Correct typo in Python3 component name ("Development.Module" → "Development.Module" is invalid)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

# LibTorch setup
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("${TORCH_INCLUDE_DIRS}")
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=llm_fs)

# Add subdirectories
add_subdirectory(src)
cmake_minimum_required(VERSION 3.10)
project(llm_fs)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include the "include" directory for headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectories
add_subdirectory(src/lib)
add_subdirectory(src/app)

# Create the executable (main program)
add_executable(main src/main.cpp)

# Link the library (llm_fs_lib) to the main executable
target_link_libraries(main PRIVATE llm_fs_lib)

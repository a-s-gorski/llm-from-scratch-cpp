cmake_minimum_required(VERSION 3.14)
project(YourProject VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define project options
include(cmake/options.cmake)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Installation
install(TARGETS your_project_library EXPORT YourProjectTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# Export package
export(EXPORT YourProjectTargets FILE YourProjectTargets.cmake)
add_library(your_project_library STATIC Library.cpp)
target_include_directories(your_project_library PUBLIC ${CMAKE_SOURCE_DIR}/include)

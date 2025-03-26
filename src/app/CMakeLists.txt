add_executable(app App.cpp)

# Link the library
target_link_libraries(app PRIVATE llm_fs_lib)
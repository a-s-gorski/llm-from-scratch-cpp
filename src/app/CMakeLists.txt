# This executable is the application
add_executable(app App.cpp)

# Link the previously created library (llm_fs_lib) to the app
target_link_libraries(app PRIVATE llm_fs_lib)
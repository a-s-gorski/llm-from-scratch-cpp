add_executable(train_tokenizer TrainTokenizer.cpp)
target_link_libraries(train_tokenizer PRIVATE llm_fs_lib ${TORCH_LIBRARIES})

add_executable(train_model TrainModel.cpp)
target_link_libraries(train_model PRIVATE llm_fs_lib ${TORCH_LIBRARIES})

add_executable(run_model RunModel.cpp)
target_link_libraries(run_model PRIVATE llm_fs_lib ${TORCH_LIBRARIES})

# Ensure rpath is set for torch shared libraries
set_target_properties(train_tokenizer train_model run_model PROPERTIES
        CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")

add_executable(train_tokenizer TrainTokenizer.cpp)
add_executable(train_model TrainModel.cpp)
add_executable(run_model RunModel.cpp)

target_link_libraries(train_tokenizer PRIVATE llm_fs_lib)
target_link_libraries(train_model PRIVATE llm_fs_lib)
target_link_libraries(run_model PRIVATE llm_fs_lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
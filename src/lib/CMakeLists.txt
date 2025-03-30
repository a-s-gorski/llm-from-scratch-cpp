add_library(llm_fs_lib STATIC
        dataset/TextDataset.cpp
        tokenizer/BaseTokenizer.cpp
        tokenizer/RegexFastTokenizer.cpp
        tokenizer/Tokenizer.cpp
        tokenizer/trainer/Trainer.cpp
        tokenizer/encoder/Encoder.cpp
        tokenizer/encoder/Encoder.h
)

target_include_directories(llm_fs_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(llm_fs_lib PUBLIC ${TORCH_LIBRARIES})
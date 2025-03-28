add_library(llm_fs_lib STATIC
        dataset/TextDataset.cpp
        tokenizer/RegexFastTokenizer.cpp
        tokenizer/trainer/Trainer.cpp
        tokenizer/encoder/Encoder.cpp
        tokenizer/encoder/Encoder.h
)

# Make include/ available for both the library and other targets
target_include_directories(llm_fs_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
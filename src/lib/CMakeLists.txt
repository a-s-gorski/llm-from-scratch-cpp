add_library(llm_fs_lib STATIC
        dataset/TextDataset.cpp
        tokenizer/BaseTokenizer.cpp
        tokenizer/RegexFastTokenizer.cpp
        tokenizer/Tokenizer.cpp
        tokenizer/trainer/Trainer.cpp
        tokenizer/encoder/Encoder.cpp
        model/dataset/GPTDataset.cpp
)

target_include_directories(llm_fs_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/lib
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)

# Fix: Link Python3 and Torch *PUBLICLY* so executables inherit the links
target_link_libraries(llm_fs_lib PUBLIC
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
        # Add these if missing symbols persist:
        ${Python3_LIBRARY}
        ${Python3_STDLIB}
)
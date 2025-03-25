add_library(llm_fs_lib STATIC
        dataset/TextDataset.cpp
        tokenizer/RegexFastTokenizer.cpp
)

# Add include directories for the library to use the headers in the "include" folder
target_include_directories(llm_fs_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
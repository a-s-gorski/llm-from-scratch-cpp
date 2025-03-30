#include "llm_fs/model/dataset/GPTDataset.h"

namespace llm_fs::model::dataset {
    GPTDataset::GPTDataset(const std::string &file_path, const tokenizer::BaseTokenizer &tokenizer,
                           const int max_length, const int stride): file_path_(file_path), tokenizer_(tokenizer.clone())


    {
    }
}

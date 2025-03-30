#ifndef GPTDATASET_H
#define GPTDATASET_H
#include <torch/torch.h>
#include "llm_fs/tokenizer/BaseTokenizer.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"

namespace llm_fs::model::dataset {
    class GPTDataset : public torch::data::Dataset<GPTDataset>{
        public:
            explicit GPTDataset(const std::string &file_path, const tokenizer::BaseTokenizer &tokenizer, const int max_length, const int stride);

    private:
        std::string file_path_;
        std::unique_ptr<tokenizer::BaseTokenizer> tokenizer_;
    };
}


#endif //GPTDATASET_H

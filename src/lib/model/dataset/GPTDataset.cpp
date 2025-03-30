#include <utility>
#include "llm_fs/model/dataset/GPTDataset.h"
#include "llm_fs/dataset/TextDataset.h"

namespace llm_fs::model::dataset {
    GPTDataset::GPTDataset(std::string file_path,
                          const tokenizer::BaseTokenizer &tokenizer,
                          int max_length,
                          int stride)
        : file_path_(std::move(file_path)),
          max_length_(max_length),
          stride_(stride),
          tokenizer_(tokenizer.clone()) {

        auto dataset = llm_fs::dataset::TextDataset(file_path_);
        auto text = dataset.load_dataset();
        auto token_ids = tokenizer_->encode(text, std::nullopt);

        for (int i = 0; i + max_length < token_ids.size(); i += stride_) {
            std::vector<int> input_chunk(token_ids.begin() + i, token_ids.begin() + i + max_length);
            std::vector<int> target_chunk(token_ids.begin() + i + 1, token_ids.begin() + i + max_length + 1);

            input_ids_.push_back(torch::tensor(input_chunk, torch::dtype(torch::kInt64)));
            target_ids_.push_back(torch::tensor(target_chunk, torch::dtype(torch::kInt64)));
        }
    }

    std::optional<unsigned long> GPTDataset::size() const {
        return input_ids_.size();
    }

    torch::data::Example<> GPTDataset::get(size_t index) {
        return {input_ids_[index], target_ids_[index]};
    }
}
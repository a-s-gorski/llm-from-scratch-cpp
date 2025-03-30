#ifndef GPTDATASET_H
#define GPTDATASET_H
#include <torch/torch.h>
#include "llm_fs/tokenizer/BaseTokenizer.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"

namespace llm_fs::model::dataset {
    class GPTDataset : public torch::data::Dataset<GPTDataset> {
    public:
        explicit GPTDataset(std::string file_path,
                          const tokenizer::BaseTokenizer &tokenizer,
                          int max_length,
                          int stride);

        // Move constructor
        GPTDataset(GPTDataset&& other) noexcept = default;

        // Delete copy constructor
        GPTDataset(const GPTDataset&) = delete;

        // Delete copy assignment
        GPTDataset& operator=(const GPTDataset&) = delete;

        // Move assignment
        GPTDataset& operator=(GPTDataset&&) noexcept = default;

        std::optional<unsigned long> size() const override;
        torch::data::Example<> get(size_t index) override;

        // Add this to make the dataset work with the stateless data loader
        bool is_stateful() const noexcept { return false; }

    private:
        std::string file_path_;
        int max_length_, stride_;
        std::unique_ptr<tokenizer::BaseTokenizer> tokenizer_;
        std::vector<torch::Tensor> input_ids_, target_ids_;
    };
}
#endif // GPTDATASET_H
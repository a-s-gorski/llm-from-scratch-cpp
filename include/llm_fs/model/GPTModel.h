
#ifndef GPTMODEL_H
#define GPTMODEL_H

#include <model/layers/LayerNorm.h>
#include <torch/torch.h>

#include "dataset/GPTDataset.h"
#include "llm_fs/tokenizer/Tokenizer.h"


namespace llm_fs::model {
    namespace layers {
        class LayerNorm;
        class TransformerBlock;
        struct GPTConfig;
    }

    class GPTModelImpl final : public torch::nn::Module {
    public:
        GPTModelImpl(int vocab_size, int context_length, int emb_dim, int n_heads, int n_layers, double drop_rate,
                     bool qkv_bias);

        explicit GPTModelImpl(layers::GPTConfig config);

        torch::Tensor forward(torch::Tensor x);

        void trainModel(
            torch::optim::Optimizer &optimizer,
            const std::string &train_path,
            const std::string &val_path,
            tokenizer::BaseTokenizer &tokenizer,
            torch::Device device,
            int max_length,
            int stride,
            int batch_size,
            int num_epochs,
            int eval_freq,
            int eval_iter,
            bool verbose = true);

        std::string generateResponse(const std::string &message, tokenizer::BaseTokenizer &tokenizer, int max_new_tokens = 50);

    private:
        torch::nn::Embedding tok_emb_{nullptr}, pos_emb_{nullptr};
        torch::nn::Dropout drop_emb_{nullptr};
        torch::nn::Sequential trf_blocks_;
        layers::LayerNorm final_norm_{nullptr};
        torch::nn::Linear out_head_{nullptr};

        static std::vector<uint32_t> tensorToU32Vector(const torch::Tensor& tensor);
        torch::Tensor computeOutputEval(const torch::Tensor &ids, int max_new_tokens, int context_size);

        torch::Tensor calcLossBatch(const torch::Tensor &input_batch, const torch::Tensor &target_batch,
                                    torch::DeviceType device);

        std::tuple<torch::Tensor, torch::Tensor> calcLossLoader(
            torch::data::StatelessDataLoader<
                torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>,
                torch::data::samplers::RandomSampler> &loader,
            torch::Device device,
            c10::optional<size_t> limit_num_batches = c10::nullopt
        );


        std::pair<double, double> evaluateModel(
            torch::data::StatelessDataLoader<
                torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>,
                torch::data::samplers::RandomSampler> &train_loader,
            torch::data::StatelessDataLoader<
                torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>,
                torch::data::samplers::RandomSampler> &val_loader,
            torch::Device device,
            size_t eval_iter
        );

        static void printProgressBar(int batch_idx, int total_batches, int width = 50);

        static void printPrintProgressBarWithEta(
            int batch_idx,
            int total_batches,
            std::chrono::steady_clock::time_point start_time,
            double loss,
            int width = 50
        );
    };


    TORCH_MODULE(GPTModel);
}

#endif //GPTMODEL_H

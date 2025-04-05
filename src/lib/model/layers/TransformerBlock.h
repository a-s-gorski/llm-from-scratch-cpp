#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include <torch/torch.h>
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "LayerNorm.h"

namespace llm_fs::model::layers {

    struct GPTConfig {
        int vocab_size=256;
        int context_length=256;
        int emb_dim=768;
        int n_heads=12;
        int n_layers=12;
        double drop_rate = 0.1;
        bool qkv_bias = false;
    };

    class TransformerBlockImpl final : public torch::nn::Module {
    public:
        explicit TransformerBlockImpl(const GPTConfig &config);
        TransformerBlockImpl(int64_t emb_dim, int64_t context_length, int num_heads, double drop_rate, bool qkv_bias);
        torch::Tensor forward(torch::Tensor x);

    private:
        CustomMultiheadAttention att{nullptr};
        FeedForward ff{nullptr};
        LayerNorm norm1{nullptr}, norm2{nullptr};
        torch::nn::Dropout drop_shortcut{nullptr};
    };

    TORCH_MODULE(TransformerBlock);

}

#endif // TRANSFORMERBLOCK_H

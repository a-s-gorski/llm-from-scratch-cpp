#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include <torch/torch.h>
#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "LayerNorm.h"

namespace llm_fs::model::layers {

    class TransformerBlockImpl : public torch::nn::Module {
    public:
        TransformerBlockImpl(int64_t emb_dim, int64_t context_length, int num_heads, double drop_rate, bool qkv_bias);

        torch::Tensor forward(torch::Tensor x);

    private:
        CustomMultiheadAttention att;
        FeedForward ff;
        LayerNorm norm1, norm2;
        torch::nn::Dropout drop_shortcut;
        // torch::nn::ModuleHolder<CustomMultiheadAttention> att;
        // torch::nn::ModuleHolder<FeedForward> ff;
        // torch::nn::ModuleHolder<LayerNorm> norm1, norm2;
        // torch::nn::Dropout drop_shortcut;
    };

    TORCH_MODULE(TransformerBlock);

}

#endif // TRANSFORMERBLOCK_H

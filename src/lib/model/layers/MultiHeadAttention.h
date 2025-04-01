#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#include <torch/torch.h>

namespace llm_fs::model::layers {
    class CustomMultiheadAttentionImpl : public torch::nn::Module {
    public:
        CustomMultiheadAttentionImpl(int d_in, int d_out, int context_length, double dropout, int num_heads, bool qkv_bias = false);
        torch::Tensor forward(torch::Tensor x);

    private:
        int d_out;
        int num_heads;
        int head_dim;

        torch::nn::Linear W_query{nullptr}, W_key{nullptr}, W_value{nullptr}, out_proj{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
        torch::Tensor mask;
    };

    // Only declare the module once
    TORCH_MODULE(CustomMultiheadAttention);
}

#endif //MULTIHEADATTENTION_H


#ifndef GPTMODEL_H
#define GPTMODEL_H

#include <model/layers/LayerNorm.h>
#include <torch/torch.h>



namespace llm_fs::model {

    namespace layers {
        class LayerNorm;
        class TransformerBlock;
        struct GPTConfig;
    }

    class GPTModelImpl final : public torch::nn::Module{
    public:
        GPTModelImpl(int vocab_size, int context_length, int emb_dim, int n_heads, int n_layers, double drop_rate, bool qkv_bias);
        explicit GPTModelImpl(layers::GPTConfig config);
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Embedding tok_emb_{nullptr}, pos_emb_{nullptr};

        torch::nn::Dropout drop_emb_{nullptr};
        torch::nn::Sequential trf_blocks_;
        layers::LayerNorm final_norm_{nullptr};
        torch::nn::Linear out_head_{nullptr};



    };

    TORCH_MODULE(GPTModel);

}

#endif //GPTMODEL_H

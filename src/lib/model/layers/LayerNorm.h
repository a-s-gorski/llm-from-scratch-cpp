#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <torch/torch.h>

namespace llm_fs::model::layers {

    class LayerNormImpl : public torch::nn::Module {
    public:
        explicit LayerNormImpl(int64_t emb_dim, double eps = 1e-5);

        torch::Tensor forward(torch::Tensor x);

    private:
        double eps;
        torch::Tensor scale, shift;
    };

    TORCH_MODULE(LayerNorm);

}

#endif // LAYERNORM_H

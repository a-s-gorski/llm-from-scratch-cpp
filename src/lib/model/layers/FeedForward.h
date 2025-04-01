#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include <torch/torch.h>
#include "GELU.h"

namespace llm_fs::model::layers {
    class FeedForwardImpl : public torch::nn::Module {
    public:
        explicit FeedForwardImpl(int64_t emb_dim);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Sequential layers;
    };

    TORCH_MODULE(FeedForward);
}

#endif //FEEDFORWARD_H

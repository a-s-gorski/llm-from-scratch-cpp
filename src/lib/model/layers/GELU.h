#ifndef GELU_H
#define GELU_H

#include <torch/torch.h>
#include <cmath>

namespace llm_fs::model::layers {
    class GELUImpl : public torch::nn::Module {
    public:
        torch::Tensor forward(torch::Tensor x);
    };

    TORCH_MODULE(GELU);
}

#endif //GELU_H

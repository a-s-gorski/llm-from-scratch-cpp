#include "LayerNorm.h"

namespace llm_fs::model::layers {

    LayerNormImpl::LayerNormImpl(int64_t emb_dim, double eps)
        : eps(eps) {
        // Initialize learnable scale and shift parameters
        scale = register_parameter("scale", torch::ones({emb_dim}));
        shift = register_parameter("shift", torch::zeros({emb_dim}));
    }

    torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
        auto mean = x.mean({-1}, /*keepdim=*/true);
        auto variance = x.var({-1}, /*unbiased=*/false, /*keepdim=*/true);

        // Normalize x
        x = (x - mean) / torch::sqrt(variance + eps);

        // Apply scale and shift
        return x * scale + shift;
    }

}

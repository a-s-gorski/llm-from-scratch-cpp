#include "GELU.h"

namespace llm_fs::model::layers {

    torch::Tensor GELUImpl::forward(torch::Tensor x) {
        constexpr float sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        return 0.5 * x * (1 + torch::tanh(sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3))));
    }

} // namespace llm_fs::model::layers

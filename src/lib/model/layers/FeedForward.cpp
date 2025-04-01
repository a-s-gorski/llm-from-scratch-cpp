#include "FeedForward.h"

namespace llm_fs::model::layers {

    FeedForwardImpl::FeedForwardImpl(int64_t emb_dim) {
        layers = register_module("layers", torch::nn::Sequential(
            torch::nn::Linear(emb_dim, 4 * emb_dim),
            GELU(),
            torch::nn::Linear(4 * emb_dim, emb_dim)
        ));
    }

    torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
        return layers->forward(x);
    }

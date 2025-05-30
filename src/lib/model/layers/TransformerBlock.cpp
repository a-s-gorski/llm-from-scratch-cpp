#include "TransformerBlock.h"

namespace llm_fs::model::layers {

    TransformerBlockImpl::TransformerBlockImpl(const GPTConfig &config)
        : TransformerBlockImpl(config.emb_dim, config.context_length, config.n_heads, config.drop_rate, config.qkv_bias) {}

    TransformerBlockImpl::TransformerBlockImpl(int64_t emb_dim, int64_t context_length, int num_heads, double drop_rate, bool qkv_bias)
        : att(emb_dim, emb_dim, context_length, drop_rate, num_heads, qkv_bias), ff(emb_dim), norm1(emb_dim), norm2(emb_dim), drop_shortcut(drop_rate)
    {
        register_module("att", att);
        register_module("ff", ff);
        register_module("norm1", norm1);
        register_module("norm2", norm2);
        register_module("drop_shortcut", drop_shortcut);
    }

    torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
        auto shortcut = x;
        x = norm1->forward(x);
        x = att->forward(x);
        x = drop_shortcut(x);
        x = x + shortcut;

        shortcut = x;
        x = norm2->forward(x);
        x = ff->forward(x);
        x = drop_shortcut(x);
        x = x + shortcut;

        return x;
    }

}

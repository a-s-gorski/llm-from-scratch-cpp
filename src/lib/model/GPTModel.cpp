#include "llm_fs/model/GPTModel.h"
#include "layers/TransformerBlock.h"

using llm_fs::model::layers::GPTConfig;
using llm_fs::model::layers::TransformerBlock;


namespace llm_fs::model {


    GPTModelImpl::GPTModelImpl(GPTConfig config) {
        tok_emb_ = torch::nn::Embedding(config.vocab_size, config.emb_dim);
        pos_emb_ = torch::nn::Embedding(config.context_length, config.emb_dim);
        drop_emb_ = torch::nn::Dropout(config.drop_rate);
        trf_blocks_ = torch::nn::Sequential();
        for (int i = 0; i < config.n_layers; i++) {
            trf_blocks_->push_back(TransformerBlock(config));
        }
        final_norm_ = layers::LayerNorm(config.emb_dim);
        out_head_ = torch::nn::Linear(torch::nn::LinearOptions(config.emb_dim, config.vocab_size).bias(false));


        register_module("tok_emb_", tok_emb_);
        register_module("pos_emb_", pos_emb_);
        register_module("drop_emb_", drop_emb_);
        register_module("trf_blocks_", trf_blocks_);
        register_module("final_norm_", final_norm_);
        // register_module("out_head_", out_head_);

    }


    GPTModelImpl::GPTModelImpl(const int vocab_size, const int context_length, const int emb_dim, const int n_heads, const int n_layers, const double drop_rate, const bool qkv_bias)
    : GPTModelImpl(GPTConfig{vocab_size, context_length, emb_dim, n_heads, n_layers, drop_rate, qkv_bias})
    {

    }

    torch::Tensor GPTModelImpl::forward(torch::Tensor x) {
        const auto seq_len = x.size(1);

        const auto tok_embeds = tok_emb_(x);
        auto pos_indices = torch::arange(seq_len, x.device()).unsqueeze(0); // Shape: (1, seq_len)
        const auto pos_embeds = pos_emb_(pos_indices);
        auto x_out = tok_embeds + pos_embeds;
        x_out = drop_emb_(x_out);
        x_out = trf_blocks_->forward(x_out);
        x_out = final_norm_(x_out);
        auto logits = out_head_(x_out);
        return logits;


    }







}
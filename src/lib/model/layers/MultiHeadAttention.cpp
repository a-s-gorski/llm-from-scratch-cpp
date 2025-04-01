#include "MultiHeadAttention.h"
#include <cmath> // for std::sqrt
#include <limits> // for std::numeric_limits

namespace llm_fs::model::layers {

    CustomMultiheadAttentionImpl::CustomMultiheadAttentionImpl(int d_in, int d_out, int context_length, double dropout, int num_heads, bool qkv_bias)
        : d_out(d_out), num_heads(num_heads), head_dim(d_out / num_heads),
          W_query(register_module("W_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
          W_key(register_module("W_key", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
          W_value(register_module("W_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
          out_proj(register_module("out_proj", torch::nn::Linear(torch::nn::LinearOptions(d_out, d_out).bias(true)))),
          dropout_layer(register_module("dropout_layer", torch::nn::Dropout(dropout))) {

        mask = torch::triu(torch::ones({context_length, context_length}, torch::kFloat32), /*diagonal=*/1)
                   .to(torch::kBool);  // Convert to boolean mask
    }

    torch::Tensor CustomMultiheadAttentionImpl::forward(torch::Tensor x) {
        int64_t b = x.size(0);
        int64_t num_tokens = x.size(1);

        auto keys = W_key(x).view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);
        auto queries = W_query(x).view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);
        auto values = W_value(x).view({b, num_tokens, num_heads, head_dim}).transpose(1, 2);

        auto attn_scores = torch::matmul(queries, keys.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim));

        // Apply mask correctly
        auto mask_bool = mask.slice(0, 0, num_tokens).slice(1, 0, num_tokens).to(attn_scores.device());
        attn_scores.masked_fill_(mask_bool, -std::numeric_limits<float>::infinity());

        auto attn_weights = torch::softmax(attn_scores, -1);
        attn_weights = dropout_layer(attn_weights);

        auto context_vec = torch::matmul(attn_weights, values)
                               .transpose(1, 2)
                               .contiguous()
                               .view({b, num_tokens, d_out});

        return out_proj(context_vec);
    }

}

#include "llm_fs/model/dataset/GPTDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/model/GPTModel.h"

#include <torch/torch.h>
#include <iostream>

using namespace llm_fs::model::dataset;
using llm_fs::tokenizer::RegexFastTokenizer;
using llm_fs::dataset::TextDataset;

int main() {
    torch::manual_seed(123);

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);
    std::cout << "Using device: " << (device_type == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    int vocab_size = 50257, context_length = 256, emb_dim = 768, n_heads = 12, n_layers = 12;
    float drop_rate = 0.1;
    bool qkv_bias = false;

    auto model = llm_fs::model::GPTModel(vocab_size, context_length, emb_dim, n_heads, n_layers, drop_rate, qkv_bias);
    model->to(device);

    RegexFastTokenizer tokenizer(RegexFastTokenizer::getPatternGPT2());
    tokenizer.load("../../tokenizers/regex_tokenizer");

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4));

    model->trainModel(
        optimizer,
        "../../data/openwebtext-10k-train-small.txt",
        "../../data/openwebtext-10k-test-small.txt",
        tokenizer, device,
        256, 128, 16, 5, 50, 10
    );


    torch::save(model, "../../models/model.pt");

    return 0;
}

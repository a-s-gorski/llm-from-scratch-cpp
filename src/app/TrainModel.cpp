#include "llm_fs/model/dataset/GPTDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include "llm_fs/dataset/TextDataset.h"

#include <torch/torch.h>

#include "llm_fs/model/GPTModel.h"


using namespace llm_fs::model::dataset;
using llm_fs::tokenizer::RegexFastTokenizer;
using llm_fs::dataset::TextDataset;

int main() {

    torch::manual_seed(123);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    torch::Device device(device_type);


    int vocab_size = 50257;
    int context_length = 256;
    int emb_dim = 768;
    int n_heads = 12;
    int n_layers = 12;
    float drop_rate = 0.1;
    bool qkv_bias = false;

    auto model = llm_fs::model::GPTModel(
        vocab_size, context_length, emb_dim, n_heads, n_layers, drop_rate, qkv_bias);

    // Tokenizer
    std::string file_prefix = "../../tokenizers/regex_tokenizer";
    auto tokenizer = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT2());
    auto dataset = TextDataset("../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();
    tokenizer.load(file_prefix.append(".model"));

    // Dataset
    std::cout << "Initializing dataset..." << std::endl;
    std::string train_dataset_path = "../../data/openwebtext-10k-train.txt";
    std::string test_dataset_path = "../../data/openwebtext-10k-test.txt";

    int max_length = 256;
    int stride = 128;

    auto train_shared_dataset = torch::data::datasets::make_shared_dataset<GPTDataset>(
        train_dataset_path, tokenizer, max_length, stride);

    auto test_shared_dataset = torch::data::datasets::make_shared_dataset<GPTDataset>(
    test_dataset_path, tokenizer, max_length, stride);

    auto train_loader = torch::data::make_data_loader(
        train_shared_dataset,
        torch::data::DataLoaderOptions().batch_size(2).drop_last(false)
    );

    auto test_loader = torch::data::make_data_loader(
        test_shared_dataset, torch::data::DataLoaderOptions().batch_size(2).drop_last(false));

    auto input_tensor = torch::randint(/*low=*/0, /*high=*/vocab_size, {2, context_length}, torch::kLong).to(device);
    model->to(device);
    std::cout << "input_tensor datatype" << input_tensor.dtype() << std::endl;
    // Run forward pass
    auto output = model->forward(input_tensor);
    std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;
    std::cout << "Output tensor shape: " << output.sizes() << std::endl;


    for (auto& batch : *train_loader) {

        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> targets;
        std::cout << batch.size() << std::endl;

        for (const auto& ex : batch) {
            inputs.push_back(ex.data);
            targets.push_back(ex.target);
        }

        auto input_tensor = torch::stack(inputs);
        auto target_tensor = torch::stack(targets);

        std::cout << "second_input" << input_tensor.dtype() << std::endl;

        input_tensor = input_tensor.to(device).to(torch::kLong);
        target_tensor = target_tensor.to(device).to(torch::kLong);

        std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;
        std::cout << "Target tensor shape: " << target_tensor.sizes() << std::endl;
        std::cout << "Max target value: " << input_tensor.max().item<int64_t>() << std::endl;
        std::cout << "Min target value: " << input_tensor.min().item<int64_t>() << std::endl;

        target_tensor = target_tensor.clamp(0, vocab_size - 1);



        auto output = model->forward(input_tensor);

        std::cout << "Output shape: " << output.sizes() << std::endl;

        break;
    }










    return 0;
}

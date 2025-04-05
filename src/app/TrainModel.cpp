


#include "llm_fs/model/dataset/GPTDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include "llm_fs/dataset/TextDataset.h"

#include <torch/torch.h>

#include "llm_fs/model/GPTModel.h"


// using namespace llm_fs::model::dataset;
// using llm_fs::tokenizer::RegexFastTokenizer;
// using llm_fs::dataset::TextDataset;

int main() {

    torch::manual_seed(123);

    int vocab_size = 50257;
    int context_length = 256;
    int emb_dim = 768;
    int n_heads = 12;
    int n_layers = 12;
    float drop_rate = 0.1;
    bool qkv_bias = false;

    auto model = llm_fs::model::GPTModel(
        vocab_size, context_length, emb_dim, n_heads, n_layers, drop_rate, qkv_bias);

    torch::Tensor inputs = torch::tensor({{16833, 3626, 6100},
                                              {40, 1107, 588}});

    auto outputs = model->forward(inputs);


    std::cout << "inputs_size: ";
    for (const auto& s : inputs.sizes()) {
        std::cout << s << " ";
    }
    std::cout << std::endl;


    std::cout << "outputs_size: ";
    for (const auto& s : outputs.sizes()) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    // torch::DeviceType device_type;
    // if (torch::cuda::is_available()) {
    //     std::cout << "CUDA available! Training on GPU." << std::endl;
    //     device_type = torch::kCUDA;
    // } else {
    //     std::cout << "Training on CPU." << std::endl;
    //     device_type = torch::kCPU;
    // }

    // torch::Device device(device_type);
    //
    // // Tokenizer
    // std::string file_prefix = "../../tokenizers/regex_tokenizer";
    // auto tokenizer = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT2());
    // auto dataset = TextDataset("../../data/openwebtext-10k.txt");
    // const auto text_dataset = dataset.load_dataset();
    // tokenizer.load(file_prefix.append(".model"));
    //
    // // Dataset
    // std::cout << "Initializing dataset..." << std::endl;
    // std::string dataset_path = "../../data/openwebtext-10k.txt";
    // int max_length = 256;
    // int stride = 128;
    //
    // auto shared_dataset = torch::data::datasets::make_shared_dataset<GPTDataset>(
    //     dataset_path, tokenizer, max_length, stride);
    //
    // auto dataloader = torch::data::make_data_loader(
    //     shared_dataset,
    //     torch::data::DataLoaderOptions().batch_size(3).drop_last(false)
    // );
    //
    // for (auto& batch : *dataloader) {
    //     std::cout << batch.size() << std::endl;
    //     std::cout << batch[0].data.size(0) << std::endl;
    //     std::cout << batch[0].target.size(0) << std::endl;
    //     std::cout << batch[1].data.size(0) << std::endl;
    //     std::cout << batch[1].target.size(0) << std::endl;
    //     break;
    // }

    return 0;
}

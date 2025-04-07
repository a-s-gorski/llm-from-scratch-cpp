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
        register_module("out_head_", out_head_);

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


    torch::Tensor GPTModelImpl::calcLossBatch(const torch::Tensor &input_batch, const torch::Tensor &target_batch, torch::DeviceType device) {
        const auto input = input_batch.to(device);
        const auto target = target_batch.to(device);
        auto logits = this->forward(input);
        auto loss = torch::nn::functional::cross_entropy(logits.flatten(0, 1), target.flatten(0,1));
        return loss;
    }

    std::tuple<torch::Tensor, torch::Tensor> GPTModelImpl::calcLossLoader(torch::data::StatelessDataLoader<torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>, torch::data::samplers::RandomSampler> &loader, torch::Device device, c10::optional<size_t> limit_num_batches) {

        double total_loss = 0.0;
        size_t batch_count = 0;

        for (auto& batch : loader) {
            if (limit_num_batches.has_value() && batch_count >= *limit_num_batches) break;

            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> targets;

            for (const auto& ex : batch) {
                inputs.push_back(ex.data);
                targets.push_back(ex.target);
            }

            auto input_tensor = torch::stack(inputs).to(device).to(torch::kLong);
            auto target_tensor = torch::stack(targets).to(device).to(torch::kLong);

            torch::Tensor loss = this->calcLossBatch(input_tensor, target_tensor, device.type());
            total_loss += loss.item<double>();
            ++batch_count;
        }

        if (batch_count == 0) {
            return {
                torch::full({}, std::numeric_limits<float>::quiet_NaN()),
                torch::tensor(0)
            };
        }

        return {
            torch::tensor(total_loss / batch_count),
            torch::tensor(static_cast<int64_t>(batch_count))
        };
    }

    void GPTModelImpl::trainModel(torch::optim::Optimizer &optimizer, const std::string &train_path, const std::string &val_path, tokenizer::BaseTokenizer &tokenizer, torch::Device device, int max_length, int stride, int batch_size, int num_epochs, int eval_freq, int eval_iter, bool verbose) {
        auto train_shared_dataset = torch::data::datasets::make_shared_dataset<dataset::GPTDataset>(
        train_path, tokenizer, max_length, stride);

    auto val_shared_dataset = torch::data::datasets::make_shared_dataset<dataset::GPTDataset>(
        val_path, tokenizer, max_length, stride);

    auto train_loader = torch::data::make_data_loader(
        train_shared_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(false)
    );

    auto val_loader = torch::data::make_data_loader(
        val_shared_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(false)
    );

    std::vector<double> train_losses, val_losses;
    std::vector<int64_t> track_tokens_seen;
    int64_t tokens_seen = 0;
    int global_step = -1;

    int total_batches = 0;
    if (train_shared_dataset->size().has_value()) {
        total_batches = train_shared_dataset->size().value() / batch_size;
        if (train_shared_dataset->size().value() % batch_size != 0) {
            total_batches += 1;
        }
    } else {
        verbose = false;
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        if (verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        }
        int batch_idx = 0;
        auto start_time = std::chrono::steady_clock::now();

        for (auto &batch: *train_loader) {
            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> targets;

            for (const auto &ex: batch) {
                inputs.push_back(ex.data);
                targets.push_back(ex.target);
            }

            auto input_tensor = torch::stack(inputs).to(device).to(torch::kLong);
            auto target_tensor = torch::stack(targets).to(device).to(torch::kLong);

            optimizer.zero_grad();
            auto loss = this->calcLossBatch(input_tensor, target_tensor, device.type());
            loss.backward();
            optimizer.step();

            tokens_seen += input_tensor.numel();
            ++global_step;
            if (verbose) {
                this->printPrintProgressBarWithEta(batch_idx + 1, total_batches, start_time, loss.item<double>());
            }
            ++batch_idx;

            if (global_step >= eval_iter && global_step % eval_freq == 0 && epoch > 0) {
                auto [train_loss, val_loss] = this->evaluateModel(*train_loader, *val_loader, device, eval_iter);
                train_losses.push_back(train_loss);
                val_losses.push_back(val_loss);
                track_tokens_seen.push_back(tokens_seen);
                if (verbose) {
                    std::cout << "\nEp " << (epoch + 1)
                            << " (Step " << global_step << "): "
                            << "Train loss " << std::fixed << std::setprecision(3) << train_loss
                            << ", Val loss " << val_loss << std::endl;
                }
            }
        }
        if (verbose) {
            std::cout << std::endl;
        }
    }
    }


    std::pair<double, double> GPTModelImpl::evaluateModel(
    torch::data::StatelessDataLoader<
            torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>,
            torch::data::samplers::RandomSampler> &train_loader,
        torch::data::StatelessDataLoader<
            torch::data::datasets::SharedBatchDataset<dataset::GPTDataset>,
            torch::data::samplers::RandomSampler> &val_loader,
        torch::Device device,
        size_t eval_iter
        ) {
        this->eval();
        torch::NoGradGuard no_grad;

        auto [train_loss_tensor, _1] = this->calcLossLoader(train_loader, device, eval_iter);
        auto [val_loss_tensor, _2] = this->calcLossLoader(val_loader, device, eval_iter);

        this->train();

        return {
            train_loss_tensor.item<double>(),
            val_loss_tensor.item<double>()
        };


    }

    void GPTModelImpl::printProgressBar(int batch_idx, int total_batches, int width) {
        float progress = static_cast<float>(batch_idx) / total_batches;
        int bar_width = width - 2;
        int pos = static_cast<int>(bar_width * progress);

        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%";
        std::cout.flush();
    }

    void GPTModelImpl::printPrintProgressBarWithEta(int batch_idx, int total_batches, std::chrono::steady_clock::time_point start_time, double loss, int width) {
        float progress = static_cast<float>(batch_idx) / total_batches;
        int bar_width = width - 2;
        int pos = static_cast<int>(bar_width * progress);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        int eta = static_cast<int>((elapsed / (progress + 1e-8)) * (1.0 - progress));

        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%"
                  << " (" << batch_idx << "/" << total_batches << ")"
                  << " ETA: " << eta << "s"
                  << " Loss: " << std::fixed << std::setprecision(3) << loss;
        std::cout.flush();
    }



}
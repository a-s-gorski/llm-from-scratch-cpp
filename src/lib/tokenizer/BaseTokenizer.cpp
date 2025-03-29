#include "llm_fs/tokenizer/BaseTokenizer.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>

namespace llm_fs::tokenizer {
    std::map<uint32_t, std::string> BaseTokenizer::_build_vocab() {
        std::map<uint32_t, std::string> vocab;
        for (int i = 0; i < 256; ++i) {
            vocab[i] = std::string(1, static_cast<char>(i));
        }
        for (const auto &[pair, idx]: merges) {
            vocab[idx] = vocab[pair.first] + vocab[pair.second];
        }
        for (const auto &[special, idx]: special_tokens) {
            vocab[idx] = special;
        }
        return vocab;
    }

    void BaseTokenizer::save(const std::string &file_prefix) {
        std::ofstream model_file(file_prefix + ".model");
        if (!model_file.is_open()) {
            throw std::runtime_error("Failed to open model file for writing");
        }

        model_file << "quickbpe v1\n";
        model_file << pattern << "\n";
        model_file << special_tokens.size() << "\n";
        for (const auto &[special, idx]: special_tokens) {
            model_file << special << " " << idx << "\n";
        }
        for (const auto &[pair, idx]: merges) {
            model_file << pair.first << " " << pair.second << "\n";
        }
        model_file.close();

        std::ofstream vocab_file(file_prefix + ".vocab");
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Failed to open vocab file for writing");
        }

        std::map<int, std::pair<int, int> > inverted_merges;
        for (const auto &[pair, idx]: merges) {
            inverted_merges[idx] = pair;
        }

        auto built_vocab = _build_vocab();
        for (const auto &[idx, token]: built_vocab) {
            auto it = inverted_merges.find(idx);
            if (it != inverted_merges.end()) {
                vocab_file << "[" << built_vocab[it->second.first] << "][" << built_vocab[it->second.second] << "] -> ["
                        << token << "] " << idx << "\n";
            } else {
                vocab_file << "[" << token << "] " << idx << "\n";
            }
        }
        vocab_file.close();
    }

    void BaseTokenizer::load(const std::string &model_file) {
        if (model_file.size() < 6 || model_file.substr(model_file.size() - 6) != ".model") {
            throw std::runtime_error("Model file must end with .model");
        }

        std::ifstream file(model_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open model file");
        }

        std::string version;
        std::getline(file, version);
        if (version != "quickbpe v1") {
            throw std::runtime_error("Unsupported model version");
        }

        std::string pattern_str;
        std::getline(file, pattern_str);
        pattern = boost::regex(pattern_str);

        int num_special;
        file >> num_special;
        file.ignore();

        special_tokens.clear();
        merges.clear();
        int idx = 256;

        for (int i = 0; i < num_special; ++i) {
            std::string special;
            int special_idx;
            file >> special >> special_idx;
            file.ignore();
            special_tokens[special] = special_idx;
        }

        int idx1, idx2;
        while (file >> idx1 >> idx2) {
            file.ignore();
            merges[{idx1, idx2}] = idx++;
        }

        file.close();
        vocab = _build_vocab();
    }
}

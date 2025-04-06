#include "llm_fs/tokenizer/BaseTokenizer.h"
#include <fstream>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <iostream> // for debug logging
#include <boost/regex.hpp>  // Make sure to include boost/regex for the pattern

namespace llm_fs::tokenizer {

    // Helper function to sanitize tokens (remove non-printable characters)
    std::string sanitize_token(const std::string& token) {
        std::string sanitized = token;
        sanitized.erase(std::remove_if(sanitized.begin(), sanitized.end(), [](char c) {
            return !std::isprint(static_cast<unsigned char>(c));  // Remove non-printable characters
        }), sanitized.end());
        return sanitized;
    }

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

    // void BaseTokenizer::save(const std::string &file_prefix) {
    //     std::ofstream model_file(file_prefix + ".model");
    //     if (!model_file.is_open()) {
    //         throw std::runtime_error("Failed to open model file for writing");
    //     }
    //
    //     model_file << "quickbpe v1\n";
    //     model_file << pattern.str() << "\n";  // Save the pattern as a string
    //     model_file << special_tokens.size() << "\n";
    //     for (const auto &[special, idx]: special_tokens) {
    //         model_file << sanitize_token(special) << " " << idx << "\n";  // Sanitize special tokens
    //     }
    //     for (const auto &[pair, idx]: merges) {
    //         model_file << pair.first << " " << pair.second << "\n";
    //     }
    //     model_file.close();
    //
    //     std::ofstream vocab_file(file_prefix + ".vocab");
    //     if (!vocab_file.is_open()) {
    //         throw std::runtime_error("Failed to open vocab file for writing");
    //     }
    //
    //     std::map<int, std::pair<int, int>> inverted_merges;
    //     for (const auto &[pair, idx]: merges) {
    //         inverted_merges[idx] = pair;
    //     }
    //
    //     auto built_vocab = _build_vocab();
    //     for (const auto &[idx, token]: built_vocab) {
    //         auto it = inverted_merges.find(idx);
    //         if (it != inverted_merges.end()) {
    //             vocab_file << "[" << built_vocab[it->second.first] << "][" << built_vocab[it->second.second] << "] -> ["
    //                        << token << "] " << idx << "\n";
    //         } else {
    //             vocab_file << "[" << sanitize_token(token) << "] " << idx << "\n";  // Sanitize vocab tokens
    //         }
    //     }
    //     vocab_file.close();
    // }
    //
    // void BaseTokenizer::load(const std::string &model_file) {
    //     if (model_file.size() < 6 || model_file.substr(model_file.size() - 6) != ".model") {
    //         throw std::runtime_error("Model file must end with .model");
    //     }
    //
    //     std::ifstream file(model_file);
    //     if (!file.is_open()) {
    //         throw std::runtime_error("Failed to open model file");
    //     }
    //
    //     std::string line;
    //
    //     // Read version
    //     std::getline(file, line);
    //     if (line != "quickbpe v1") {
    //         throw std::runtime_error("Unsupported model version: " + line);
    //     }
    //
    //     // Read regex pattern
    //     std::getline(file, line);
    //     pattern = boost::regex(line);  // Load regex pattern from file
    //
    //     // Read number of special tokens
    //     std::getline(file, line);
    //     int num_special;
    //     try {
    //         num_special = std::stoi(line);
    //     } catch (...) {
    //         throw std::runtime_error("Failed to parse number of special tokens");
    //     }
    //
    //     special_tokens.clear();
    //     merges.clear();
    //     int idx = 256;
    //
    //     // Read special tokens
    //     for (int i = 0; i < num_special; ++i) {
    //         std::getline(file, line);
    //         std::istringstream iss(line);
    //         std::string special;
    //         int special_idx;
    //         if (!(iss >> special >> special_idx)) {
    //             throw std::runtime_error("Failed to parse special token line: " + line);
    //         }
    //         special_tokens[special] = special_idx;  // Populate special_tokens
    //     }
    //
    //     // Debugging: print loaded special tokens
    //     std::cout << "Loaded special tokens:\n";
    //     for (const auto& token : special_tokens) {
    //         std::cout << "Special token: " << token.first << " -> " << token.second << std::endl;
    //     }
    //
    //     // Read merges
    //     while (std::getline(file, line)) {
    //         if (line.empty()) continue;
    //         std::istringstream iss(line);
    //         int idx1, idx2;
    //         if (!(iss >> idx1 >> idx2)) {
    //             std::cerr << "Warning: Skipping malformed merge line: " << line << "\n";
    //             continue;
    //         }
    //         merges[{idx1, idx2}] = idx++;  // Populate merges
    //     }
    //
    //     // Debugging: print loaded merges
    //     std::cout << "Loaded merges:\n";
    //     for (const auto& merge : merges) {
    //         std::cout << "Merge: " << merge.first.first << " " << merge.first.second << " -> " << merge.second << std::endl;
    //     }
    //
    //     file.close();
    //
    //     // Now build the vocab with the loaded merges and special tokens
    //     vocab = _build_vocab();
    //
    //     std::cout << "[Tokenizer::load] Loaded " << special_tokens.size() << " special tokens, "
    //               << merges.size() << " merges, vocab size: " << vocab.size() << "\n";
    // }
}

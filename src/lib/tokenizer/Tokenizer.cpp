#include "llm_fs/tokenizer/Tokenizer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>

namespace llm_fs::tokenizer {

    using json = nlohmann::json;

    std::unique_ptr<BaseTokenizer> Tokenizer::clone() const {
        return std::make_unique<Tokenizer>(*this);
    }

    std::string Tokenizer::decode(std::vector<uint32_t> tokens) {
        std::string text_bytes;
        for (const auto &idx : tokens) {
            if (vocab.find(idx) != vocab.end()) {
                const auto &bytes = vocab.at(idx);
                text_bytes.append(bytes.begin(), bytes.end());
            } else {
                text_bytes += "[UNK]";
            }
        }
        return text_bytes;
    }

    std::vector<uint32_t> Tokenizer::encode(std::string text, const std::optional<std::vector<uint8_t>> &ids) {
        std::vector<int> byte_ids;
        if (ids.has_value()) {
            byte_ids.assign(ids->begin(), ids->end());
        } else {
            for (char c : text) {
                byte_ids.push_back(static_cast<uint8_t>(c));
            }
        }

        while (byte_ids.size() >= 2) {
            auto stats = get_stats(byte_ids);
            if (stats.empty()) break;

            auto best_pair = stats.begin()->first;
            int min_merge_index = merges.count(best_pair) ? merges[best_pair] : INT_MAX;

            for (const auto &[pair, count] : stats) {
                int current_merge_index = merges.count(pair) ? merges[pair] : INT_MAX;
                if (current_merge_index < min_merge_index) {
                    min_merge_index = current_merge_index;
                    best_pair = pair;
                }
            }

            if (min_merge_index == INT_MAX) {
                break;
            }

            byte_ids = merge(byte_ids, best_pair, min_merge_index);
        }

        return std::vector<uint32_t>(byte_ids.begin(), byte_ids.end());
    }

    void Tokenizer::train(std::string text, unsigned int vocab_size) {
        assert(vocab_size >= 256);
        const unsigned int num_merges = vocab_size - 256;
        std::vector<int> ids(text.begin(), text.end());

        for (int idx = 0; idx < 256; ++idx) {
            vocab[idx] = {static_cast<uint8_t>(idx)};
        }

        for (unsigned int i = 0; i < num_merges; ++i) {
            auto stats = get_stats(ids);
            if (stats.empty()) break;

            auto max_pair = stats.begin()->first;
            int max_count = stats.begin()->second;
            for (const auto &stat : stats) {
                if (stat.second > max_count) {
                    max_pair = stat.first;
                    max_count = stat.second;
                }
            }

            int idx = 256 + i;
            ids = merge(ids, max_pair, idx);
            merges[max_pair] = idx;

            auto merged_bytes = vocab[max_pair.first];
            auto second_part = vocab[max_pair.second];
            merged_bytes.insert(merged_bytes.end(), second_part.begin(), second_part.end());
            vocab[idx] = merged_bytes;
        }
    }

    std::map<std::pair<int, int>, int> Tokenizer::get_stats(const std::vector<int> &ids,
        std::map<std::pair<int, int>, int> *counts) {
        std::map<std::pair<int, int>, int> local_counts;
        std::map<std::pair<int, int>, int> &result = counts ? *counts : local_counts;

        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            std::pair pair = {ids[i], ids[i + 1]};
            result[pair]++;
        }
        return result;
    }

    std::vector<int> Tokenizer::merge(const std::vector<int> &ids, const std::pair<int, int> &pair, int idx) {
        std::vector<int> newids;
        size_t i = 0;

        while (i < ids.size()) {
            if (i < ids.size() - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
                newids.push_back(idx);
                i += 2;
            } else {
                newids.push_back(ids[i]);
                ++i;
            }
        }

        return newids;
    }

    void Tokenizer::save(const std::string &file_prefix) {
        // Save vocab (Base64-encoded)
        json j_vocab;
        for (const auto &[token_id, bytes] : vocab) {
            j_vocab[std::to_string(token_id)] = base64_encode(bytes);
        }

        std::ofstream vocab_file(file_prefix + ".vocab.json");
        vocab_file << j_vocab.dump(2);
        vocab_file.close();

        // Save merges
        json j_merges;
        for (const auto &[pair, id] : merges) {
            j_merges.push_back({
                {"first", pair.first},
                {"second", pair.second},
                {"id", id}
            });
        }

        std::ofstream merges_file(file_prefix + ".merges.json");
        merges_file << j_merges.dump(2);
        merges_file.close();
    }

    void Tokenizer::load(const std::string &file_prefix) {
        // Load vocab
        std::ifstream vocab_file(file_prefix + ".vocab.json");
        if (!vocab_file.is_open()) throw std::runtime_error("Failed to open vocab file");

        json j_vocab;
        vocab_file >> j_vocab;

        vocab.clear();
        for (auto it = j_vocab.begin(); it != j_vocab.end(); ++it) {
            int id = std::stoi(it.key());
            vocab[id] = base64_decode(it.value().get<std::string>());
        }

        // Load merges
        std::ifstream merges_file(file_prefix + ".merges.json");
        if (!merges_file.is_open()) throw std::runtime_error("Failed to open merges file");

        json j_merges;
        merges_file >> j_merges;

        merges.clear();
        for (const auto &entry : j_merges) {
            int first = entry.at("first");
            int second = entry.at("second");
            int id = entry.at("id");
            merges[{first, second}] = id;
        }
    }

    uint32_t Tokenizer::vocabSize() const {
        return vocab.size();
    }

    // Base64 encode function
    std::string Tokenizer::base64_encode(const std::vector<uint8_t>& data) {
        static constexpr char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string encoded;
        int val = 0;
        int valb = -6;
        for (uint8_t c : data) {
            val = (val << 8) + c;
            valb += 8;
            while (valb >= 0) {
                encoded.push_back(base64_table[(val >> valb) & 0x3F]);
                valb -= 6;
            }
        }
        if (valb > -6) encoded.push_back(base64_table[((val << 8) >> (valb + 8)) & 0x3F]);
        while (encoded.size() % 4) encoded.push_back('=');
        return encoded;
    }

    // Base64 decode function
    std::vector<uint8_t> Tokenizer::base64_decode(const std::string& encoded) {
        static constexpr int base64_reverse_table[256] = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, -1,
            -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1
        };

        std::vector<uint8_t> decoded;
        int val = 0;
        int valb = -8;
        for (uint8_t c : encoded) {
            if (base64_reverse_table[c] == -1) break;
            val = (val << 6) + base64_reverse_table[c];
            valb += 6;
            if (valb >= 0) {
                decoded.push_back((val >> valb) & 0xFF);
                valb -= 8;
            }
        }
        return decoded;
    }

}


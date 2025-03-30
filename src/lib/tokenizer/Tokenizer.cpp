#include "llm_fs/tokenizer/Tokenizer.h"

namespace llm_fs::tokenizer {

    std::unique_ptr<BaseTokenizer> Tokenizer::clone() const {
        return  std::make_unique<Tokenizer>(*this);
    }

    std::string Tokenizer::decode(std::vector<uint32_t> tokens) {
        // given ids (list of integers), return C++ string
        std::string text_bytes;
        for (const auto &idx: tokens) {
            if (vocab.find(idx) != vocab.end()) {
                text_bytes += vocab[idx];
            } else {
                text_bytes += "�";
            }
        }
        return text_bytes;
    }

    std::vector<uint32_t> Tokenizer::encode(std::string text, const std::optional<std::vector<uint8_t> > &ids) {
        // given a string text, return the token ids
        std::vector<int> byte_ids;
        if (ids.has_value()) {
            // Use provided byte IDs if available
            byte_ids.assign(ids->begin(), ids->end());
        } else {
            // Convert text to raw bytes
            for (char c: text) {
                byte_ids.push_back(static_cast<uint8_t>(c));
            }
        }

        while (byte_ids.size() >= 2) {
            // find the pair with the lowest merge index
            auto stats = get_stats(byte_ids);
            if (stats.empty()) break;

            // Find the pair with the lowest merge index
            auto best_pair = stats.begin()->first;
            int min_merge_index = merges.count(best_pair) ? merges[best_pair] : INT_MAX;

            for (const auto &[pair, count]: stats) {
                int current_merge_index = merges.count(pair) ? merges[pair] : INT_MAX;
                if (current_merge_index < min_merge_index) {
                    min_merge_index = current_merge_index;
                    best_pair = pair;
                }
            }

            // if no merges available, break
            if (min_merge_index == INT_MAX) {
                break;
            }

            // merge the best pair
            byte_ids = merge(byte_ids, best_pair, min_merge_index);
        }

        // Convert to uint32_t before returning
        return std::vector<uint32_t>(byte_ids.begin(), byte_ids.end());
    }

    void Tokenizer::train(std::string text, unsigned int vocab_size) {
        assert(vocab_size >= 256);
        const unsigned int num_merges = vocab_size - 256;
        std::vector<int> ids(text.begin(), text.end());

        for (int idx = 0; idx < 256; ++idx) {
            vocab[idx] = {static_cast<unsigned char>(idx)};
        }

        for (unsigned int i = 0; i < num_merges; ++i) {
            auto stats = get_stats(ids);
            if (stats.empty()) break;

            auto max_pair = stats.begin()->first;
            int max_count = stats.begin()->second;
            for (const auto &stat: stats) {
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
            std::pair<int, int> pair = {ids[i], ids[i + 1]};
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
}

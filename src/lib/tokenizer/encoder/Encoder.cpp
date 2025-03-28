#include "Encoder.h"
#include <algorithm>

namespace llm_fs::tokenizer::encoder {

    Encoder::TokenizeResult Encoder::encode(const std::vector<uint8_t>& ids, const std::vector<int>& splits,
                                            const std::vector<int64_t>& token_pairs, int init_tokens, int num_threads) {
        int vocab_size = token_pairs.size() + init_tokens;

        std::vector<std::vector<uint32_t>> splitted;
        splitted.reserve(splits.size() - 1);

        for (size_t i = 0; i < splits.size() - 1; i++) {
            int curr = splits[i], next = splits[i + 1];
            splitted.emplace_back(ids.begin() + curr, ids.begin() + next);
        }

        std::unordered_map<int64_t, uint32_t> pair_to_token;
        for (size_t i = 0; i < token_pairs.size(); i++) {
            pair_to_token[token_pairs[i]] = static_cast<uint32_t>(init_tokens + i);
        }

        auto tokenizeChunks = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                tokenizeChunk(splitted[i], pair_to_token, vocab_size);
            }
        };

        std::vector<std::thread> threads;
        size_t chunkSize = splitted.size() / num_threads;

        for (size_t i = 0; i < num_threads - 1; i++) {
            threads.emplace_back(tokenizeChunks, i * chunkSize, (i + 1) * chunkSize);
        }
        for (size_t i = (num_threads - 1) * chunkSize; i < splitted.size(); i++) {
            tokenizeChunk(splitted[i], pair_to_token, vocab_size);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        std::vector<uint32_t> result;
        for (const auto& chunk : splitted) {
            result.insert(result.end(), chunk.begin(), chunk.end());
        }

        return {result};
    }

    void Encoder::tokenizeChunk(std::vector<uint32_t>& ids, std::unordered_map<int64_t, uint32_t>& pair_to_tok, int vocab_size) {
        std::vector<TokenStat> stats;
        stats.reserve(ids.size());

        for (size_t i = 0; i < ids.size() - 1; i++) {
            if (int64_t key = static_cast<int64_t>(ids[i]) * vocab_size + ids[i + 1]; pair_to_tok.count(key)) {
                stats.push_back({key, pair_to_tok[key]});
            }
        }

        while (!stats.empty()) {
            auto min_it = std::min_element(stats.begin(), stats.end(), [](const TokenStat& a, const TokenStat& b) {
                return a.tok_id < b.tok_id;
            });
            uint32_t min_tok_id = min_it->tok_id;
            const int64_t min_pair_id = min_it->pair_id;

            stats.erase(std::remove_if(stats.begin(), stats.end(), [min_tok_id](const TokenStat& s) {
                return s.tok_id == min_tok_id;
            }), stats.end());

            const uint32_t tok_1 = min_pair_id / vocab_size;
            const uint32_t tok_2 = min_pair_id % vocab_size;

            size_t curr_append = 0;
            uint32_t prev = -1;
            for (size_t i = 0; i < ids.size(); i++) {
                if (i < ids.size() - 1 && ids[i] == tok_1 && ids[i + 1] == tok_2) {
                    ids[curr_append] = min_tok_id;
                    i++;
                } else {
                    ids[curr_append] = ids[i];
                }
                if (prev != static_cast<uint32_t>(-1)) {
                    if (int64_t pair_id = static_cast<int64_t>(prev) * vocab_size + ids[curr_append]; pair_to_tok.count(pair_id)) {
                        stats.push_back({pair_id, pair_to_tok[pair_id]});
                    }
                }
                prev = ids[curr_append];
                curr_append++;
            }
            ids.resize(curr_append);
        }
    }

}

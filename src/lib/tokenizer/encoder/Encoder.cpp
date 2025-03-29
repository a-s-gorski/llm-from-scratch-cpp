#include "Encoder.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <thread>
#include <limits>
#include <cstdint>

namespace llm_fs::tokenizer::encoder {
    void Encoder::_tokenizeChunk(std::vector<uint32_t> &ids,
                                 const std::unordered_map<int64_t, uint32_t> &pair_to_tok,
                                 int vocab_size) {
        if (vocab_size <= 0 || ids.size() < 2) {
            return;
        }

        std::vector<TokenStat> stats;
        stats.reserve(ids.size() / 2); // Pre-allocate reasonable space

        while (true) {
            stats.clear();
            for (size_t i = 0; i < ids.size() - 1; ++i) {
                const uint32_t t1 = ids[i];
                const uint32_t t2 = ids[i + 1];

                // Safe multiplication check
                if (t1 > static_cast<uint32_t>(std::numeric_limits<int64_t>::max() / vocab_size)) {
                    continue;
                }

                const int64_t key = static_cast<int64_t>(t1) * vocab_size + t2;
                if (auto it = pair_to_tok.find(key); it != pair_to_tok.end()) {
                    stats.push_back({key, it->second});
                }
            }

            if (stats.empty()) break;

            // Find the merge with the smallest token ID
            auto min_it = std::min_element(stats.begin(), stats.end(),
                                           [](const TokenStat &a, const TokenStat &b) {
                                               return a.tok_id < b.tok_id;
                                           });

            const uint32_t tok1 = static_cast<uint32_t>(min_it->pair_id / vocab_size);
            const uint32_t tok2 = static_cast<uint32_t>(min_it->pair_id % vocab_size);
            const uint32_t merged_tok = min_it->tok_id;

            // Perform merge in-place where possible
            size_t write_pos = 0;
            for (size_t read_pos = 0; read_pos < ids.size();) {
                if (read_pos < ids.size() - 1 &&
                    ids[read_pos] == tok1 &&
                    ids[read_pos + 1] == tok2) {
                    ids[write_pos++] = merged_tok;
                    read_pos += 2;
                } else {
                    if (write_pos != read_pos) {
                        ids[write_pos] = ids[read_pos];
                    }
                    write_pos++;
                    read_pos++;
                }
            }
            ids.resize(write_pos);
        }
    }

    std::vector<uint32_t> Encoder::tokenize(const std::vector<uint8_t> &input_ids,
                                            const std::vector<int> &splits,
                                            const std::vector<int64_t> &token_pairs,
                                            int init_tokens,
                                            int num_threads) {
        // Validate inputs
        if (input_ids.empty() || splits.size() <= 1 || token_pairs.empty() || num_threads <= 0 || init_tokens < 0) {
            return {};
        }

        // Prepare vocabulary mapping
        const int token_pairs_count = static_cast<int>(token_pairs.size());
        const int vocab_size = token_pairs_count + init_tokens;
        std::unordered_map<int64_t, uint32_t> pair_to_token;
        for (int i = 0; i < token_pairs_count; i++) {
            pair_to_token[token_pairs[i]] = init_tokens + i;
        }

        // Split input into chunks based on split points
        std::vector<std::vector<uint32_t> > chunks(splits.size() - 1);
        for (size_t i = 0; i < splits.size() - 1; i++) {
            const int start = splits[i];
            const int end = splits[i + 1];
            if (start < 0 || end > static_cast<int>(input_ids.size()) || start > end) {
                return {}; // Invalid split points
            }

            chunks[i].reserve(end - start);
            for (int j = start; j < end; j++) {
                chunks[i].push_back(static_cast<uint32_t>(input_ids[j]));
            }
        }

        // Adjust number of threads if necessary
        num_threads = std::min(num_threads, static_cast<int>(chunks.size()));
        if (num_threads <= 1) {
            // Single-threaded path
            for (auto &chunk: chunks) {
                _tokenizeChunk(chunk, pair_to_token, vocab_size);
            }
        } else {
            // Multithreaded path
            std::vector<std::thread> threads;
            threads.reserve(num_threads - 1);
            const int chunks_per_thread = static_cast<int>(chunks.size()) / num_threads;

            for (int i = 0; i < num_threads - 1; i++) {
                const int start = i * chunks_per_thread;
                const int end = start + chunks_per_thread;
                threads.emplace_back([&, start, end]() {
                    for (int j = start; j < end; j++) {
                        _tokenizeChunk(chunks[j], pair_to_token, vocab_size);
                    }
                });
            }

            // Main thread handles remaining chunks
            const int remaining_start = (num_threads - 1) * chunks_per_thread;
            for (size_t j = remaining_start; j < chunks.size(); j++) {
                _tokenizeChunk(chunks[j], pair_to_token, vocab_size);
            }

            // Wait for all threads to finish
            for (auto &thread: threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        }

        // Concatenate all chunks into final result
        std::vector<uint32_t> result;
        size_t total_size = 0;
        for (const auto &chunk: chunks) {
            total_size += chunk.size();
        }
        result.reserve(total_size);

        for (const auto &chunk: chunks) {
            result.insert(result.end(), chunk.begin(), chunk.end());
        }

        return result;
    }
} // namespace llm_fs::tokenizer::encoder

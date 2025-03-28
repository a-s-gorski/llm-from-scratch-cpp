#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <unordered_map>
#include <thread>
#include <cstdint>  // for fixed-width integer types

namespace llm_fs::tokenizer::encoder {

    struct TokenStat {
        int64_t pair_id;
        uint32_t tok_id;
    };

    class Encoder {
    public:
        Encoder() = default;
        std::vector<uint32_t> tokenize(const std::vector<uint8_t>& input_ids,
                                      const std::vector<int>& splits,
                                      const std::vector<int64_t>& token_pairs,
                                      int init_tokens,
                                      int num_threads = 4);

    private:
        static void _tokenizeChunk(std::vector<uint32_t>& ids,
                                 const std::unordered_map<int64_t, uint32_t>& pair_to_tok,
                                 int vocab_size);
    };

} // namespace llm_fs::tokenizer::encoder

#endif // ENCODER_H
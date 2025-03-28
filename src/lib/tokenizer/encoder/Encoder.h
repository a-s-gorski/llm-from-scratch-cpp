#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <unordered_map>
#include <thread>

namespace llm_fs::tokenizer::encoder {

    class Encoder {
    public:
        struct TokenStat {
            int64_t pair_id;
            uint32_t tok_id;
        };

        struct TokenizeResult {
            std::vector<uint32_t> ids;
        };

        TokenizeResult encode(const std::vector<uint8_t>& ids, const std::vector<int>& splits,
                              const std::vector<int64_t>& token_pairs, int init_tokens, int num_threads=4);

    private:
        static void tokenizeChunk(std::vector<uint32_t>& ids, std::unordered_map<int64_t, uint32_t>& pair_to_tok, int vocab_size);
    };

}

#endif // ENCODER_H

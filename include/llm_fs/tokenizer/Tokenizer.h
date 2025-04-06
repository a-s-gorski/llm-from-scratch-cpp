#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "BaseTokenizer.h"
#include <map>
#include <vector>
#include <string>
#include <optional>

namespace llm_fs::tokenizer {
    class Tokenizer final : public BaseTokenizer {
    public:
        Tokenizer() = default;
        std::unique_ptr<BaseTokenizer> clone() const override;

        void train(std::string text, unsigned int vocab_size) override;
        std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<uint8_t>>& ids) override;
        std::string decode(std::vector<uint32_t> tokens) override;
        void save(const std::string &file_prefix) override;
        void load(const std::string &file_prefix) override;
        uint32_t vocabSize() const;

    private:
        std::map<std::pair<int, int>, int> merges;  // Holds merge operations
        std::map<int, std::vector<uint8_t>> vocab;  // Maps token id to byte vector

        static std::map<std::pair<int, int>, int> get_stats(const std::vector<int> &ids,
                                                            std::map<std::pair<int, int>, int> *counts = nullptr);
        static std::vector<int> merge(const std::vector<int> &ids, const std::pair<int, int> &pair, int idx);

        // Base64 encoding and decoding methods
        static std::string base64_encode(const std::vector<uint8_t>& data);
        static std::vector<uint8_t> base64_decode(const std::string& encoded);
    };
}

#endif // TOKENIZER_H

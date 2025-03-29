#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "BaseTokenizer.h"

namespace llm_fs::tokenizer {
    class Tokenizer final : public BaseTokenizer {
    public:
        Tokenizer() = default;

        void train(std::string text, unsigned int vocab_size) override;

        std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<u_int8_t> > &ids) override;

        std::string decode(std::vector<uint32_t> tokens) override;

    private:
        std::map<std::pair<int, int>, int> merges;
        std::map<int, std::string> vocab;

        static std::map<std::pair<int, int>, int> get_stats(const std::vector<int> &ids,
                                                            std::map<std::pair<int, int>, int> *counts = nullptr);

        static std::vector<int> merge(const std::vector<int> &ids, const std::pair<int, int> &pair, int idx);
    };
}


#endif //TOKENIZER_H

#ifndef REGEXFASTTOKENIZER_H
#define REGEXFASTTOKENIZER_H

#include <boost/regex.hpp>
#include "BaseTokenizer.h"

namespace llm_fs::tokenizer {
    namespace trainer {
        struct Merge;
    }

    namespace encoder {
        struct Result;
    }

    class Trainer;

    class Encoder;

    class RegexFastTokenizer final : public BaseTokenizer {
    public:
        enum class PatternType {
            GPT2,
            GPT4
        };

        RegexFastTokenizer()
            : BaseTokenizer(),
              pattern(pattern_gpt4, boost::regex_constants::icase | boost::regex_constants::optimize) {}

        explicit RegexFastTokenizer(const std::string& pattern)
            : BaseTokenizer(),
              pattern(pattern, boost::regex_constants::icase | boost::regex_constants::optimize) {}

        explicit RegexFastTokenizer(PatternType pattern_type);

        void train(std::string text, unsigned int vocab_size) override;
        std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<u_int8_t>> &ids) override;
        std::vector<uint32_t> encode_efficient(std::string text, const std::optional<std::vector<u_int8_t>> &ids) override;

        std::string decode(std::vector<uint32_t> tokens) override;

        static const std::string& getPatternGPT2();
        static const std::string& getPatternGPT4();

    private:
        boost::regex pattern;
        std::map<std::pair<int, int>, int> merges;
        std::map<int, std::string> vocab;

        static inline const std::string pattern_gpt2 =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
        static inline const std::string pattern_gpt4 =
            R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";

        std::vector<int32_t> presplit(const std::string& text) const;
        static std::vector<uint8_t> generateIds(const std::string& text);
        std::vector<std::string> generateTextChunks(const std::string& text) const;
        static std::vector<int32_t> generateIdsList(const std::string& text);
        static std::tuple<std::map<std::pair<int, int>, int>, std::map<int, std::string>>
            calculate_merges_and_vocab(const std::vector<trainer::Merge>& results, int init_tokens);

        static std::vector<int32_t> generateSplitIndices(const std::vector<std::string>& text_chunks, int id_processed_size);
    };
}

#endif // REGEXFASTTOKENIZER_H
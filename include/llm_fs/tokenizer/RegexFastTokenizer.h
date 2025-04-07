#ifndef REGEXFASTTOKENIZER_H
#define REGEXFASTTOKENIZER_H

#include <boost/regex.hpp>
#include "BaseTokenizer.h"
#include <map>
#include <vector>
#include <optional>

namespace llm_fs::tokenizer {
    namespace trainer {
        struct Merge;
    }

    class RegexFastTokenizer final : public BaseTokenizer {
    public:
        enum class PatternType {
            GPT2,
            GPT4
        };

        RegexFastTokenizer();
        std::unique_ptr<BaseTokenizer> clone() const override;
        explicit RegexFastTokenizer(const std::string &pattern);
        explicit RegexFastTokenizer(PatternType pattern_type);
        void train(std::string text, unsigned int vocab_size) override;
        std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<uint8_t> > &ids) override;
        std::string decode(std::vector<uint32_t> tokens) override;
        static const std::string &getPatternGPT2();
        static const std::string &getPatternGPT4();
        uint32_t vocabSize() const;
        void save(const std::string& file_prefix) override;
        void load(const std::string& file_prefix) override;


    private:
        boost::regex pattern;
        std::map<std::pair<int, int>, int> merges;
        std::map<uint32_t, std::string> vocab;
        std::map<std::string, uint32_t> vocab_inverse;

        static const std::string pattern_gpt2;
        static const std::string pattern_gpt4;

        static std::string toLowerCase(const std::string& str);
        static std::string toCapitalizedCase(const std::string &str);

    };
}

#endif // REGEXFASTTOKENIZER_H

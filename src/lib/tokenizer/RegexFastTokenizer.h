#ifndef REGEXFASTTOKENIZER_H
#define REGEXFASTTOKENIZER_H

#include <boost/regex.hpp>
#include "BaseTokenizer.h"

class RegexFastTokenizer : public BaseTokenizer {
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

    void train(std::string text, unsigned int vocab_size, bool verbose = false) override;

private:
    boost::regex pattern;
    static const std::string pattern_gpt2;
    static const std::string pattern_gpt4;
    std::vector<int32_t> presplit(const std::string& text);
    std::vector<uint8_t> generate_ids(const std::string& text);
    std::vector<std::string> generate_text_chunks(const std::string& text);
    std::vector<int32_t> generate_ids_list(const std::string& text);
};

const std::string RegexFastTokenizer::pattern_gpt2 =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

const std::string RegexFastTokenizer::pattern_gpt4 =
    R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";

#endif // REGEXFASTTOKENIZER_H
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <iostream>

namespace llm_fs::tokenizer {
    const std::string RegexFastTokenizer::pattern_gpt2 =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

    const std::string RegexFastTokenizer::pattern_gpt4 =
        R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";

    const std::string& RegexFastTokenizer::getPatternGPT2() {
        return pattern_gpt2;
    }

    const std::string& RegexFastTokenizer::getPatternGPT4() {
        return pattern_gpt4;
    }

    RegexFastTokenizer::RegexFastTokenizer()
        : BaseTokenizer(), pattern(pattern_gpt4, boost::regex_constants::icase | boost::regex_constants::optimize) {}

    RegexFastTokenizer::RegexFastTokenizer(const std::string& pattern)
        : BaseTokenizer(), pattern(pattern, boost::regex_constants::icase | boost::regex_constants::optimize) {}

    RegexFastTokenizer::RegexFastTokenizer(PatternType pattern_type)
        : BaseTokenizer(), pattern(pattern_type == PatternType::GPT2 ? pattern_gpt2 : pattern_gpt4,
                                  boost::regex_constants::icase | boost::regex_constants::optimize) {}

    void RegexFastTokenizer::train(std::string text, unsigned int vocab_size) {
        boost::sregex_iterator it(text.begin(), text.end(), pattern);
        boost::sregex_iterator end;

        uint32_t token_id = 0;
        for (; it != end; ++it) {
            std::string token = it->str();
            if (vocab_inverse.find(token) == vocab_inverse.end()) {
                vocab[token_id] = token;
                vocab_inverse[token] = token_id;
                token_id++;
                if (vocab.size() >= vocab_size) break;
            }
        }
    }

    std::vector<uint32_t> RegexFastTokenizer::encode(std::string text, const std::optional<std::vector<uint8_t>>& ids) {
        std::vector<uint32_t> encoded_tokens;
        boost::sregex_iterator it(text.begin(), text.end(), pattern);
        boost::sregex_iterator end;

        for (; it != end; ++it) {
            std::string token = it->str();
            if (vocab_inverse.find(token) != vocab_inverse.end()) {
                encoded_tokens.push_back(vocab_inverse[token]);
            } else {
                uint32_t new_id = vocab.size();
                vocab[new_id] = token;
                vocab_inverse[token] = new_id;
                encoded_tokens.push_back(new_id);
            }
        }

        std::cout << "Encoded: ";
        for (auto t : encoded_tokens) std::cout << t << " ";
        std::cout << std::endl;
        return encoded_tokens;
    }

    std::string RegexFastTokenizer::decode(std::vector<uint32_t> tokens) {
        std::string output;
        for (uint32_t token_id : tokens) {
            if (vocab.find(token_id) != vocab.end()) {
                output += vocab[token_id];
            } else {
                output += "[UNK]";
            }
        }

        std::cout << "Decoded: " << output << std::endl;
        return output;
    }
}
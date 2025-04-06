#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <fstream>
#include <nlohmann/json.hpp>

#include <iostream>

namespace llm_fs::tokenizer {
    std::unique_ptr<BaseTokenizer> RegexFastTokenizer::clone() const {
        return  std::make_unique<RegexFastTokenizer>(*this);
    }

    const std::string RegexFastTokenizer::pattern_gpt2 =
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";

    const std::string RegexFastTokenizer::pattern_gpt4 =
            R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";

    const std::string &RegexFastTokenizer::getPatternGPT2() {
        return pattern_gpt2;
    }

    const std::string &RegexFastTokenizer::getPatternGPT4() {
        return pattern_gpt4;
    }

    RegexFastTokenizer::RegexFastTokenizer()
        : BaseTokenizer(), pattern(pattern_gpt4, boost::regex_constants::icase | boost::regex_constants::optimize) {
    }

    RegexFastTokenizer::RegexFastTokenizer(const std::string &pattern)
        : BaseTokenizer(), pattern(pattern, boost::regex_constants::icase | boost::regex_constants::optimize) {
    }

    RegexFastTokenizer::RegexFastTokenizer(PatternType pattern_type)
        : BaseTokenizer(), pattern(pattern_type == PatternType::GPT2 ? pattern_gpt2 : pattern_gpt4,
                                   boost::regex_constants::icase | boost::regex_constants::optimize) {
    }

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
        std::cout << "vocab size after training: "<< vocab.size() << std::endl;
        vocab_size = vocab.size();
    }

    std::vector<uint32_t>
    RegexFastTokenizer::encode(std::string text, const std::optional<std::vector<uint8_t> > &ids) {
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

        return encoded_tokens;
    }

    std::string RegexFastTokenizer::decode(std::vector<uint32_t> tokens) {
        std::string output;
        for (uint32_t token_id: tokens) {
            if (vocab.find(token_id) != vocab.end()) {
                output += vocab[token_id];
            } else {
                output += "[UNK]";
            }
        }

        return output;
    }

    uint32_t RegexFastTokenizer::vocabSize() const {
        std::cout << "vocab size: " << vocab.size() << std::endl;
        std::cout << "reverse vocab size: " << vocab_inverse.size() << std::endl;
        return vocab_size;
    }

    void RegexFastTokenizer::save(const std::string& file_prefix) {
    // ---- Save vocab ----
    nlohmann::json vocab_json;
    for (const auto& [id, token] : vocab) {
        vocab_json[std::to_string(id)] = token;
    }

    std::ofstream vocab_out(file_prefix + "_vocab.json");
    if (!vocab_out) {
        throw std::runtime_error("Failed to open vocab file for saving: " + file_prefix + "_vocab.json");
    }
    vocab_out << vocab_json.dump(2);
    vocab_out.close();

    // ---- Save merges ----
    nlohmann::json merges_json;
    for (const auto& [pair, id] : merges) {
        std::string key = std::to_string(pair.first) + "," + std::to_string(pair.second);
        merges_json[key] = id;
    }

    std::ofstream merges_out(file_prefix + "_merges.json");
    if (!merges_out) {
        throw std::runtime_error("Failed to open merges file for saving: " + file_prefix + "_merges.json");
    }
    merges_out << merges_json.dump(2);
    merges_out.close();
}

void RegexFastTokenizer::load(const std::string& file_prefix) {
    // ---- Load vocab ----
    std::ifstream vocab_in(file_prefix + "_vocab.json");
    if (!vocab_in) {
        throw std::runtime_error("Failed to open vocab file for loading: " + file_prefix + "_vocab.json");
    }

    nlohmann::json vocab_json;
    vocab_in >> vocab_json;
    vocab_in.close();

    vocab.clear();
    vocab_inverse.clear();

    for (auto& [id_str, token] : vocab_json.items()) {
        uint32_t id = std::stoul(id_str);
        vocab[id] = token;
        vocab_inverse[token] = id;
    }

    vocab_size = vocab.size();

    // ---- Load merges ----
    std::ifstream merges_in(file_prefix + "_merges.json");
    if (!merges_in) {
        throw std::runtime_error("Failed to open merges file for loading: " + file_prefix + "_merges.json");
    }

    nlohmann::json merges_json;
    merges_in >> merges_json;
    merges_in.close();

    merges.clear();
    for (auto& [key, id] : merges_json.items()) {
        auto comma_pos = key.find(',');
        if (comma_pos == std::string::npos) continue;

        int first = std::stoi(key.substr(0, comma_pos));
        int second = std::stoi(key.substr(comma_pos + 1));
        merges[{first, second}] = id;
    }
}
}

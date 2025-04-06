#ifndef BASETOKENIZER_H
#define BASETOKENIZER_H

#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <boost/regex.hpp>

namespace llm_fs::tokenizer {
    class BaseTokenizer {
    public:
        virtual ~BaseTokenizer() = default;
        virtual std::unique_ptr<BaseTokenizer> clone() const = 0;
        virtual void train(std::string text, unsigned int vocab_size) = 0;
        virtual std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<uint8_t>>& ids) = 0;
        virtual std::string decode(std::vector<uint32_t> tokens) = 0;
        virtual void save(const std::string& file_prefix) = 0;
        virtual void load(const std::string& model_file) = 0;

    protected:
        std::map<std::pair<int, int>, int> merges;
        boost::regex pattern;
        std::unordered_map<std::string, int> special_tokens = {};
        std::map<uint32_t, std::string> vocab;
        unsigned int vocab_size = 0;
        unsigned int init_tokens = 256;
        std::map<uint32_t, std::string> _build_vocab();
    };
}

#endif //BASETOKENIZER_H

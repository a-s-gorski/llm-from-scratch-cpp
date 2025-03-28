#ifndef BASETOKENIZER_H
#define BASETOKENIZER_H

#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include <boost/regex.hpp>

namespace llm_fs::tokenizer {
    class BaseTokenizer {
    public:
        virtual ~BaseTokenizer() = default;
        virtual void train(std::string text, unsigned int vocab_size)=0;
        virtual std::vector<uint32_t> encode(std::string text, const std::optional<std::vector<u_int8_t>> &ids )=0;
        virtual std::vector<uint32_t> encode_efficient(std::string text, const std::optional<std::vector<u_int8_t>> &ids )=0;
        virtual std::string decode(std::vector<uint32_t> tokens)=0;
    protected:
        std::unordered_map<int, int> merges = {};
        boost::regex pattern;
        std::unordered_map<std::string, int> special_tokens = {};
        std::unordered_map<int, std::vector<unsigned char>> vocab = {};
        unsigned int vocab_size = 0;
        unsigned int init_tokens = 256;
    };
}





#endif //BASETOKENIZER_H

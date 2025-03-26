#ifndef BASETOKENIZER_H
#define BASETOKENIZER_H

#include <unordered_map>
#include <string>
#include <vector>
#include <boost/regex.hpp>

namespace llm_fs::tokenizer {
    class BaseTokenizer {
    public:
        virtual ~BaseTokenizer() = default;
        virtual void train(std::string text, unsigned int vocab_size)=0;
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

#include "RegexFastTokenizer.h"

RegexFastTokenizer::RegexFastTokenizer(PatternType pattern_type) : BaseTokenizer() {
    switch(pattern_type) {
        case PatternType::GPT2:
            pattern = boost::regex(pattern_gpt2,
                boost::regex_constants::icase | boost::regex_constants::optimize);
        break;
        case PatternType::GPT4:
            default:
                pattern = boost::regex(pattern_gpt4,
                    boost::regex_constants::icase | boost::regex_constants::optimize);
        break;
    }
}


void RegexFastTokenizer::train(std::string text, unsigned int vocab_size, bool verbose){
    this->vocab_size = vocab_size;
    auto id_list = presplit(text);
    

}

std::vector<int32_t> RegexFastTokenizer::presplit(const std::string& text){
    const std::vector<std::string>& text_chunks = generate_text_chunks(text);
    const std::vector<uint8_t>& ids = generate_ids(text);
    std::vector<int32_t> id_list = generate_ids_list(text);
    size_t i = 0, j = 0;

    for (const std::string& chunk : text_chunks) {
        // Get UTF-8 byte length (same as Python's len(chunk.encode("utf-8")))
        size_t chunk_length = chunk.size();

        // Ensure we don't exceed bounds
        if (j + chunk_length > ids.size() || i + chunk_length > id_list.size()) {
            throw std::out_of_range("Index out of bounds while processing chunks");
        }

        // Copy the chunk of ids
        std::copy(ids.begin() + j,
                 ids.begin() + j + chunk_length,
                 id_list.begin() + i);

        // Update positions (+1 for padding as in original Python)
        i += chunk_length + 1;
        j += chunk_length;
    }

    return id_list;


return id_list;


}

std::vector<uint8_t> RegexFastTokenizer::generate_ids(const std::string& text){
    return std::vector<uint8_t>(text.begin(), text.end());
}

std::vector<std::string> RegexFastTokenizer::generate_text_chunks(const std::string& text){
    std::vector<std::string> tokens;
    boost::sregex_iterator it(text.begin(), text.end(), pattern);
    boost::sregex_iterator end;

    for (; it != end; ++it) {
        tokens.push_back(it->str());
    }

    return tokens;
}

std::vector<int32_t> RegexFastTokenizer::generate_ids_list(const std::string& text){
    return std::vector<int>(text.length(), 0);
}
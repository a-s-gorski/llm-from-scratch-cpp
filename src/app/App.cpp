#include <boost/regex.hpp>
#include <iostream>
#include <vector>

std::vector<std::string> tokenize_unicode(const std::string &text) {

    boost::regex pattern(
        R"('(?:[sdmt]|ll|ve|re)| ?[a-zA-ZÀ-ÿ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)",
        boost::regex_constants::icase | boost::regex_constants::optimize
    );

    boost::regex pattern_gpt2(
        R"((?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\w]?\w+|\d{1,3}| ?[^\s\w][\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)",
        boost::regex_constants::icase | boost::regex_constants::optimize
    );

    std::vector<std::string> tokens;
    boost::sregex_iterator it(text.begin(), text.end(), pattern);
    boost::sregex_iterator end;

    for (; it != end; ++it) {
        tokens.push_back(it->str());
    }

    return tokens;
}

std::vector<uint8_t> string_to_uint8_array(const std::string& str) {
    // Directly reinterpret each char as uint8_t (no copy of data)
    return std::vector<uint8_t>(str.begin(), str.end());
}

int main() {
    const std::string text = "This is a sample text";
    auto ids = string_to_uint8_array(text);
    for (float f : ids) std::cout << f << " ";


}
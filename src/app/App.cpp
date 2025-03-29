#include <iostream>
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <filesystem>

#include "llm_fs/tokenizer/Tokenizer.h"

using llm_fs::dataset::TextDataset;
using llm_fs::tokenizer::RegexFastTokenizer;


int main() {
    auto dataset = TextDataset("../../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();

    std::cout << text_dataset.size() << text_dataset.substr(0, 30) << std::endl;

    // auto tokenizer = llm_fs::tokenizer::Tokenizer();

    auto tokenizer = llm_fs::tokenizer::RegexFastTokenizer(RegexFastTokenizer::getPatternGPT4());

    //small
    // tokenizer.train(text_dataset.substr(0, 1000), 256 + 10000);

    //regular
    tokenizer.train(text_dataset, 256+10000);

    std::string input = "Hello my name is adam!!";

    std::cout << input << std::endl;

    auto encoded = tokenizer.encode(input, std::nullopt);

    auto decoded = tokenizer.decode(encoded);

    std::cout << decoded << std::endl;


    // std::string query = "Hello World";
    // auto tokens = tokenizer.encode(query, std::nullopt);
    //
    //
    // std::cout << "tokens: " << tokens.size() << std::endl;
    // for (const auto& token : tokens) {
    //     std::cout << token << " ";
    // }
    // std::cout << std::endl;
    //
    // const std::string decoded = tokenizer.decode(tokens);
    //
    // std::cout << decoded << std::endl;



}
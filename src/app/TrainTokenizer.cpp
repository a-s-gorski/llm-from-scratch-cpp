#include <iostream>
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <filesystem>

#include "llm_fs/tokenizer/Tokenizer.h"

using llm_fs::dataset::TextDataset;
using llm_fs::tokenizer::RegexFastTokenizer;
using llm_fs::tokenizer::Tokenizer;


int main() {
    auto dataset = TextDataset("../../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();

    std::string file_prefix = "../../../tokenizers/regular_tokenizer";
    auto tokenizer = Tokenizer();


    // std::string file_prefix = "../../../tokenizers/regex_tokenizer";
    // auto tokenizer = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT4());

    //small
    // tokenizer.train(text_dataset.substr(0, 1000), 256 + 10000);

    //regular
    tokenizer.train(text_dataset.substr(0, 10000), 256 + 10000);

    const std::string input = "Hello my name is adam!!";

    std::cout << input << std::endl;

    const auto encoded = tokenizer.encode(input, std::nullopt);

    const auto decoded = tokenizer.decode(encoded);

    std::cout << decoded << std::endl;

    // test after saving

    tokenizer.save(file_prefix);


    auto tokenizer2 = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT2());

    tokenizer2.load(file_prefix.append(".model"));

    const std::string input2 = "Hello my name is adam!!";

    std::cout << input2 << std::endl;

    auto encoded2 = tokenizer.encode(input, std::nullopt);

    const auto decoded2 = tokenizer.decode(encoded);

    std::cout << decoded2 << std::endl;
}

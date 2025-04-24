#include <iostream>
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <filesystem>
#include <c10/util/typeid.h>

#include "llm_fs/tokenizer/Tokenizer.h"

using llm_fs::dataset::TextDataset;
using llm_fs::tokenizer::RegexFastTokenizer;
using llm_fs::tokenizer::Tokenizer;

int main() {
    auto dataset = TextDataset("../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();

    std::string file_prefix = "../../tokenizers/regex_tokenizer";
    auto tokenizer = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT2());

    // std::string file_prefix = "../../tokenizers/tokenizer";
    // auto tokenizer = Tokenizer();

    tokenizer.train(text_dataset, 256 + 20000);

    const std::string input = " my name is Adam";
    std::cout << input << std::endl;

    const auto encoded = tokenizer.encode(input, std::nullopt);
    const auto decoded = tokenizer.decode(encoded);

    std::cout << "Decoded after encoding: " << decoded << std::endl;

    tokenizer.save(file_prefix);

    auto tokenizer2 = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT2());
    tokenizer2.load(file_prefix);

    // auto tokenizer2 = Tokenizer();
    // tokenizer2.load(file_prefix);

    const std::string input2 = "my name is Adam";
    std::cout << input2 << std::endl;

    auto encoded2 = tokenizer2.encode(input2, std::nullopt);
    const auto decoded2 = tokenizer2.decode(encoded2);

    std::cout << "Decoded after loading tokenizer: " << decoded2 << std::endl;

    std::cout << "Vocabulary size: " << tokenizer2.vocabSize() << std::endl;


    return 0;
}

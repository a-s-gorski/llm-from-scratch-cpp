#include <iostream>
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <filesystem>

using llm_fs::dataset::TextDataset;
using llm_fs::tokenizer::RegexFastTokenizer;


int main() {
    auto dataset = TextDataset("../../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();

    std::cout << text_dataset.size() << text_dataset.substr(0, 30) << std::endl;

    auto tokenizer = RegexFastTokenizer(RegexFastTokenizer::getPatternGPT4());

    //small
    tokenizer.train(text_dataset.substr(0, 1000), 256 + 10000);




}
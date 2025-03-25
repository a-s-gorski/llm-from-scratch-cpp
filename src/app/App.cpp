#include <iostream>
#include "llm_fs/dataset/TextDataset.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include <filesystem>


int main() {
    auto dataset = TextDataset("../../../data/openwebtext-10k.txt");
    const auto text_dataset = dataset.load_dataset();

    std::cout << text_dataset.size() << text_dataset.substr(0, 30) << std::endl;



}
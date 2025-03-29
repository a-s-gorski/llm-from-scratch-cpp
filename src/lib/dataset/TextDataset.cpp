#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include  "llm_fs/dataset/TextDataset.h"

namespace llm_fs::dataset {
    std::string TextDataset::load_dataset() {
        std::ifstream file(dataset_path);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + dataset_path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
}


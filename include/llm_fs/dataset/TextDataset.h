#ifndef TEXTDATASET_H
#define TEXTDATASET_H

#include <utility>

#include "BaseDataset.h"

namespace llm_fs::dataset {
    class TextDataset : BaseDataset {
    public:
        explicit TextDataset(std::string dataset_path) : BaseDataset(), dataset_path(std::move(dataset_path)) {}
        std::string load_dataset() override;
    private:
        const std::string dataset_path;
    };

}




#endif //TEXTDATASET_H

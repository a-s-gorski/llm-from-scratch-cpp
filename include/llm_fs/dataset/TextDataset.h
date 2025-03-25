#ifndef TEXTDATASET_H
#define TEXTDATASET_H

#include "BaseDataset.h"


class TextDataset : BaseDataset {
   public:
     TextDataset(std::string dataset_path) : BaseDataset(), dataset_path(dataset_path) {}
     std::string load_dataset();
   private:
     const std::string dataset_path;
};



#endif //TEXTDATASET_H

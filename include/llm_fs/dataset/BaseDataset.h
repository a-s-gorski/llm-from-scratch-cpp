#ifndef BASEDATASET_H
#define BASEDATASET_H

#include <string>


    namespace llm_fs::dataset {
        class BaseDataset {
        public:
            virtual ~BaseDataset() = default;

            virtual std::string load_dataset() = 0;
        };
    }





#endif //BASEDATASET_H

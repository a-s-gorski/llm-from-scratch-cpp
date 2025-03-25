#ifndef BASEDATASET_H
#define BASEDATASET_H

#include <string>


class BaseDataset {
  public:
    virtual std::string load_dataset() = 0;
};


#endif //BASEDATASET_H

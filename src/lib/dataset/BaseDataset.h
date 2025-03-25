//
// Created by agorski on 3/25/25.
//
#include <string>

#ifndef BASEDATASET_H
#define BASEDATASET_H

class BaseDataset {
  public:
    virtual std::string load_dataset() = 0;
}


#endif //BASEDATASET_H

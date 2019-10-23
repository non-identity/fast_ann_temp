#ifndef FAST_ANN_DATASET_H_
#define FAST_ANN_DATASET_H_

#include <string>
#include <vector>

#include "fast_ann/logger.h"

namespace fast_ann {

typedef size_t DatasetSizeType;
typedef DatasetSizeType DatasetIndexType;
typedef unsigned int DimensionType;

template <typename T>
class DataReader;

template <typename T>
class Dataset {
   public:
    inline DatasetSizeType size() { return data_.size(); }

    inline T* get(DatasetIndexType index) { return data_[index]; }

    inline DimensionType get_dimension() { return dimension_; }

    inline void LogData(DatasetIndexType index) {
        T* ptr = data_[index];
        std::string data_str;
        for (int i = 0; i < dimension_; i++) {
            data_str += std::to_string(ptr[i]);
            data_str += " ";
        }
        LOG_INFO("Data at index " << index << " : " << data_str);
    }

   private:
    Dataset(DimensionType dimension) : dimension_(dimension) {}

    Dataset(DimensionType dimension, DatasetSizeType ds_size)
        : dimension_(dimension) {
        data_.reserve(ds_size);
    };

    friend DataReader<T>;
    DimensionType dimension_;
    std::vector<T*> data_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_DATASET_H_

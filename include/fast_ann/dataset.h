#ifndef FAST_ANN_DATASET_H_
#define FAST_ANN_DATASET_H_

#include <string>
#include <vector>

#include "fast_ann/logger.h"

namespace fast_ann {

typedef size_t DatasetIndexType;
typedef size_t DimensionType;

template <typename T>
class DataReader;

template <typename T>
class Dataset {
   public:
    typedef std::pair<DatasetIndexType, T*> DataType;

    inline DatasetIndexType size() { return data_.size(); }

    inline DataType item_at(DatasetIndexType index) { return data_[index]; }

    inline DimensionType dimension() { return dimension_; }

    inline void LogData(DatasetIndexType index) {
        T* ptr = data_[index].second;
        std::string data_str;
        for (int i = 0; i < dimension_; i++) {
            data_str += std::to_string(ptr[i]);
            data_str += " ";
        }
        LOG_INFO("Data at index " << index << " : " << data_str);
    }

   private:
    Dataset(DimensionType dimension) : dimension_(dimension) {}

    Dataset(DimensionType dimension, DatasetIndexType ds_size)
        : dimension_(dimension) {
        data_.reserve(ds_size);
    };

    friend DataReader<T>;
    DimensionType dimension_;
    std::vector<DataType> data_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_DATASET_H_

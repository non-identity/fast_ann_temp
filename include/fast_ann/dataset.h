#ifndef FAST_ANN_DATASET_H_
#define FAST_ANN_DATASET_H_

#include <string>
#include <vector>

#include "fast_ann/distance.h"
#include "fast_ann/logger.h"

namespace fast_ann {

typedef int DatasetIndexType;
typedef int DimensionType;

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

    inline void SwapData(DatasetIndexType a, DatasetIndexType b) {
        std::swap(data_[a], data_[b]);
    }

    inline void PartitionByDistance(DatasetIndexType lower,
                                    DatasetIndexType pos,
                                    DatasetIndexType upper,
                                    Distance<T, T>* dist_func) {
        std::nth_element(
            data_.begin() + lower + 1, data_.begin() + pos,
            data_.begin() + upper,
            [lower, this, dist_func](const DataType& lhs, const DataType& rhs) {
                return dist_func->operator()(data_[lower].second, lhs.second,
                                             dimension_) <
                       dist_func->operator()(data_[lower].second, rhs.second,
                                             dimension_);
            });
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

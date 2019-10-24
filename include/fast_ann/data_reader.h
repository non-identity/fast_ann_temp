#ifndef FAST_ANN_DATA_READER_H_
#define FAST_ANN_DATA_READER_H_

#include "fast_ann/dataset.h"

namespace fast_ann {

template <typename T>
class DataReader {
   public:
    virtual Dataset<T> read(const std::string file_name) = 0;
    virtual ~DataReader() {}

   protected:
    Dataset<T> CreateDataset(DimensionType dimension) {
        return Dataset<T>(dimension);
    }

    Dataset<T> CreateDataset(DimensionType dimension,
                             DatasetIndexType ds_size) {
        return Dataset<T>(dimension, ds_size);
    }

    void PushData(Dataset<T>& dataset, DatasetIndexType id, T* data_ptr) {
        dataset.data_.push_back({id, data_ptr});
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_DATA_READER_H_

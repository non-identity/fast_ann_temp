#ifndef FAST_ANN_DATA_READERS_XVECS_READER_H_
#define FAST_ANN_DATA_READERS_XVECS_READER_H_

#include <fstream>

#include "fast_ann/data_reader.h"
#include "fast_ann/logger.h"

namespace fast_ann {

template <typename T>
class XvecsReader : public DataReader<T> {
   public:
    Dataset<T> read(const std::string file_name) {
        std::ifstream file_stream(file_name, std::ios::binary);
        DimensionType dim;
        file_stream.read((char*)&dim, sizeof(dim));
        LOG_DEBUG("Dataset dimension is " << dim);
        Dataset<T> dataset = DataReader<T>::CreateDataset(dim);
        file_stream.seekg(0, file_stream.end);
        unsigned long long num_bytes_in_file = file_stream.tellg();
        DatasetIndexType dataset_size =
            num_bytes_in_file / (sizeof(dim) + dim * sizeof(T));
        LOG_DEBUG("Dataset size is " << dataset_size << "\n");
        T* data_ptr = new T[dim * dataset_size];
        file_stream.seekg(sizeof(dim), file_stream.beg);
        for (DatasetIndexType i = 0; i < dataset_size; i++) {
            file_stream.read((char*)data_ptr, dim * sizeof(T));
            DataReader<T>::PushData(dataset, i, data_ptr);
            data_ptr += dim;
            file_stream.seekg(sizeof(dim), file_stream.cur);
        }
        file_stream.close();
        return dataset;
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_DATA_READERS_XVECS_READER_H_

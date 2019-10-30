#ifndef FAST_ANN_DATASET_H_
#define FAST_ANN_DATASET_H_

#include <mpi.h>
#include <string>
#include <vector>

#include "fast_ann/logger.h"
#include "hnswlib/hnswlib.h"

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
                                    hnswlib::DISTFUNC<T> fstdistfunc_,
                                    void* dist_func_param_) {
        std::nth_element(data_.begin() + lower + 1, data_.begin() + pos,
                         data_.begin() + upper,
                         [lower, this, fstdistfunc_, dist_func_param_](
                             const DataType& lhs, const DataType& rhs) {
                             return fstdistfunc_(data_[lower].second,
                                                 lhs.second, dist_func_param_) <
                                    fstdistfunc_(data_[lower].second,
                                                 rhs.second, dist_func_param_);
                         });
    }

    inline Dataset<T>* GetSubset(DatasetIndexType lower,
                                 DatasetIndexType upper) {
        auto start = data_.begin() + lower;
        auto end = data_.begin() + upper;
        Dataset<T>* result_ptr = new Dataset<T>(dimension_, upper - lower);
        result_ptr->data_.insert(result_ptr->data_.end(), start, end);
        return result_ptr;
    }

    void sendData(int rank, MPI_Request &req) {
        int size = data_.size();
        MPI_Isend(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &req);
        MPI_Isend(
            data_.data(),
            (sizeof(DataType) * data_.size()),
            MPI_BYTE, rank, 0, MPI_COMM_WORLD, &req);
    }

    static Dataset<T>* recvData(int rank, int dim) {
        MPI_Status status;
        int size;
        MPI_Recv(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
        DataType* data = new DataType[size];
        MPI_Recv(data,
                 (sizeof(DataType) * size),
                 MPI_BYTE, rank, 0, MPI_COMM_WORLD, &status);
        Dataset<T>* new_dataset_ = new Dataset<T>(dim);
        std::vector<DataType> vec_data(data, data + size);
        new_dataset_ -> data_ = vec_data;
        new_dataset_ -> LogData(0);
        return new_dataset_;
    }

   private:
    Dataset(DimensionType dimension) : dimension_(dimension) {}

    Dataset(DimensionType dimension, DatasetIndexType ds_size)
        : dimension_(dimension) {
        data_.resize(ds_size);
    };

    friend DataReader<T>;
    DimensionType dimension_;
    std::vector<DataType> data_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_DATASET_H_

#ifndef FAST_ANN_DISTANCES_L2_NORM_H_
#define FAST_ANN_DISTANCES_L2_NORM_H_

namespace fast_ann {

typedef unsigned int DimensionType;

template <typename T>
T L2SquaredNaive(const T* data_ptr_l, const T* data_ptr_r,
                 DimensionType dimension) {
    T result = 0;
    for (DimensionType i = 0; i < dimension; i++) {
        T diff = data_ptr_l[i] - data_ptr_r[i];
        result += (diff * diff);
    }
    return result;
}

}  // namespace fast_ann

#endif  // FAST_ANN_DISTANCES_L2_NORM_H_

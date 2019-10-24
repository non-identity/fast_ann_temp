#ifndef FAST_ANN_DISTANCES_L2_NORM_H_
#define FAST_ANN_DISTANCES_L2_NORM_H_

#include "fast_ann/distance.h"

namespace fast_ann {

template <typename T, typename R>
class L2SquaredNaive : public Distance<T, R> {
   public:
    R operator()(const T* ptr_l, const T* ptr_r, DimensionType dimension) {
        R result = 0;
        for(DimensionType i = 0; i < dimension; i++) {
            T diff = ptr_l[i] - ptr_r[i];
            result += (diff * diff);
        }
        return result;
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_DISTANCES_L2_NORM_H_

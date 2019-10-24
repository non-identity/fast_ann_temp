#ifndef FAST_ANN_DISTANCE_H_
#define FAST_ANN_DISTANCE_H_

namespace fast_ann {

typedef size_t DimensionType;

template <typename T, typename R>
class Distance {
   public:
    virtual R operator()(const T* ptr_l, const T* ptr_r,
                         DimensionType dimension) = 0;
};

}  // namespace fast_ann

#endif  // FAST_ANN_DISTANCE_H_

#ifndef FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_

#include "fast_ann/search_algorithm.h"

namespace fast_ann {

template <typename T, typename R, typename DIST>
class BruteForceSearch : public SearchAlgorithm<T, R, DIST> {
   public:
    using SearchAlgorithm<T, R, DIST>::dataset_;
    using SearchAlgorithm<T, R, DIST>::k_;
    using SearchAlgorithm<T, R, DIST>::dist_func_;
    using ResultType = typename SearchAlgorithm<T, R, DIST>::ResultType;

    BruteForceSearch(Dataset<T> dataset, int k)
        : SearchAlgorithm<T, R, DIST>(dataset, k) {}

    ResultType Search(const T* query_ptr) {
        ResultType result;
        DatasetIndexType size = dataset_.size();
        DimensionType dimension = dataset_.dimension();
        DatasetIndexType first_index = std::min((int)k_, (int)size);
        for (DatasetIndexType i = 0; i < first_index; i++) {
            result.push({dist_func_->operator()(
                             query_ptr, dataset_.item_at(i).second, dimension),
                         dataset_.item_at(i).first});
        }
        T cur_max_dist = result.top().first;
        for (DatasetIndexType i = first_index; i < size; i++) {
            T cur_dist = dist_func_->operator()(
                query_ptr, dataset_.item_at(i).second, dimension);
            if (cur_dist < cur_max_dist) {
                result.push({cur_dist, dataset_.item_at(i).first});
                result.pop();
                cur_max_dist = result.top().first;
            }
        }
        return result;
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_

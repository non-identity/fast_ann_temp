#ifndef FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_

#include "fast_ann/search_algorithm.h"

namespace fast_ann {

template <typename T, class DIST>
class BruteForceSearch : public SearchAlgorithm<T, DIST> {
   public:
    BruteForceSearch(Dataset<T> dataset, int k) : SearchAlgorithm(dataset, k) {}

    ResultType Search(const T* query_ptr) {
        ResultType result;
        DatasetSizeType size = dataset_.size();
        DimensionType dimension = dataset_.get_dimension();
        DatasetIndexType first_index = std::min(k, size);
        for (DatasetIndexType i = 0; i < first_index; i++) {
            result.push({DIST(query_ptr, dataset_.get(i), dimension), i});
        }
        T cur_max_dist = result.top().first;
        for (DatasetIndexType i = first_index; i < size; i++) {
            T cur_dist = DIST(query_ptr, dataset_.get(i), dimension);
            if (cur_dist < cur_max_dist) {
                result.push({cur_dist, i});
                result.pop();
                cur_max_dist = result.top().first;
            }
        }
        return result;
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_BRUTE_FORCE_SEARCH_H_

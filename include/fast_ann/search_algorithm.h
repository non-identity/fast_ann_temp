#ifndef FAST_ANN_SEARCH_ALGORITHM_H_
#define FAST_ANN_SEARCH_ALGORITHM_H_

#include <queue>

#include "fast_ann/dataset.h"

namespace fast_ann {

template <typename T, typename DIST>
class SearchAlgorithm {
   public:
    typedef std::priority_queue<std::pair<T, DatasetIndexType> > ResultType;

    SearchAlgorithm(Dataset<T>& dataset, int k) : dataset_(dataset), k_(k) {}

    virtual ResultType Search(const T* query_ptr) = 0;

   protected:
    Dataset<T> dataset_;
    int k_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHM_H_

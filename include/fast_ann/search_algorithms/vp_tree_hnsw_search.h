#ifndef FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_

#include <mpi.h>

#include "fast_ann/search_algorithm.h"

namespace fast_ann {

template <typename T, typename R, typename DIST>
class VPTreeHNSWSearch : public SearchAlgorithm<T, R, DIST> {
   public:
    using SearchAlgorithm<T, R, DIST>::dataset_;
    using SearchAlgorithm<T, R, DIST>::k_;
    using SearchAlgorithm<T, R, DIST>::dist_func_;
    using ResultType = typename SearchAlgorithm<T, R, DIST>::ResultType;

    BruteForceSearch(Dataset<T> dataset, int k)
        : SearchAlgorithm<T, R, DIST>(dataset, k) {
        std::random_device rd;
        rng_.seed(rd());
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
        levels_ = std::log(num_procs) / std::log(2);
        if (rank == 0) {
            LOG_DEBUG("Num levels : " << levels_);
        }
        nodes_.reserve(2 * num_procs_ - 1);
        ConstructVPTree(0, dataset_.size(), 0);
    }

    ResultType Search(const T* query_ptr) {
        ResultType result;
        return result;
    }

   private:
    struct VPTreeNode {
        VPTreeNode(int data_pos_t)
            : data_pos(data_pos_t), left(-1), right(-1) {}

        DatasetIndexType data_pos;
        R threshold;
        DatasetIndexType left;
        DatasetIndexType right;
    };

    DatasetIndexType MakeVPTreeNode(DatasetIndexType data_pos) {
        nodes_.push_back(VPTreeNode(data_pos));
        return ((DatasetIndexType)nodes_.size()) - 1;
    }

    void SelectVPTreeRoot(DatasetIndexType lower, DatasetIndexType upper) {
        std::uniform_int_distribution<DatasetIndexType> uni(lower, upper - 1);
        DatasetIndexType root = uni(rng_);
        dataset_.SwapData(lower, root);
        return;
    }

    void PartitionByDistance(DatasetIndexType lower, DatasetIndexType pos,
                             DatasetIndexType upper) {
        dataset_.PartitionByDistance(lower, pos, upper, dist_func_);
    }

    DatasetIndexType ConstructVPTree(DatasetIndexType lower,
                                     DatasetIndexType upper, int level, int id) {
        if (lower >= upper) {
            return -1;
        } else if (lower + 1 == upper) {
            return MakeVPTreeNode(lower);
        } else {
            SelectVPTreeRoot(lower, upper);
            DatasetIndexType median = (upper + lower) / 2;
            PartitionByDistance(lower, median, upper);
            auto node_pos = MakeVPTreeNode(lower);
            nodes_[node_pos].threshold = dist_func_->operator()(
                dataset_.item_at(lower).second, dataset_.item_at(median).second,
                dataset_.dimension());
            nodes_[node_pos].left = ConstructVPTree(lower + 1, median);
            nodes_[node_pos].right = ConstructVPTree(median, upper);
            return node_pos;
        }
    }

    std::vector<VPTreeNode> nodes_;
    std::mt19937 rng_;
    R tau_;
    int num_procs_;
    int rank_;
    int levels_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_

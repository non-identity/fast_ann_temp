#ifndef FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_

#include <random>

#include "fast_ann/search_algorithm.h"

namespace fast_ann {

template <typename T, typename R, typename DIST>
class VPTreeSearch : public SearchAlgorithm<T, R, DIST> {
   public:
    using SearchAlgorithm<T, R, DIST>::dataset_;
    using SearchAlgorithm<T, R, DIST>::k_;
    using SearchAlgorithm<T, R, DIST>::dist_func_;
    using ResultType = typename SearchAlgorithm<T, R, DIST>::ResultType;

    VPTreeSearch(Dataset<T> dataset, int k)
        : SearchAlgorithm<T, R, DIST>(dataset, k) {
        std::random_device rd;
        rng_.seed(rd());
        nodes_.reserve(dataset_.size());
        ConstructVPTree(0, dataset_.size());
    }

    ResultType Search(const T* query_ptr) {
        ResultType result;
        tau_ = std::numeric_limits<R>::max();
        SearchNode(query_ptr, nodes_[0], result);
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
                                     DatasetIndexType upper) {
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

    void SearchNode(const T* query_ptr, const VPTreeNode& node,
                    ResultType& result) {
        R dist = dist_func_->operator()(dataset_.item_at(node.data_pos).second,
                                        query_ptr, dataset_.dimension());
        if (dist < tau_) {
            if (result.size() == k_) {
                result.pop();
            }
            result.push({dist, dataset_.item_at(node.data_pos).first});
            if (result.size() == k_) {
                tau_ = result.top().first;
            }
        }
        if (dist < node.threshold) {
            if (node.left != -1 && dist - tau_ <= node.threshold)
                SearchNode(query_ptr, nodes_[node.left], result);

            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result);
        } else {
            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result);

            if (node.left != -1 && dist - tau_ <= node.threshold)
                SearchNode(query_ptr, nodes_[node.left], result);
        }
    }

    std::vector<VPTreeNode> nodes_;
    std::mt19937 rng_;
    R tau_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_

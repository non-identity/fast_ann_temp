#ifndef FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_

#include <random>

#include "hnswlib/hnswlib.h"
#include "fast_ann/dataset.h"

namespace fast_ann {

template<typename dist_t>
class VPTreeSearch {
   public:
    typedef std::priority_queue<std::pair<dist_t, DatasetIndexType> > ResultType;

    VPTreeSearch(hnswlib::SpaceInterface <dist_t> *s, Dataset<dist_t> dataset) : dataset_(dataset) {
        std::random_device rd;
        rng_.seed(rd());
        nodes_.reserve(dataset_.size());
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        ConstructVPTree(0, dataset_.size());
    }

    ResultType searchKnn(const dist_t* query_ptr, size_t k) {
        ResultType result;
        tau_ = std::numeric_limits<dist_t>::max();
        SearchNode(query_ptr, nodes_[0], result, k);
        return result;
    }

   private:
    struct VPTreeNode {
        VPTreeNode(size_t data_pos_t)
            : data_pos(data_pos_t), left(-1), right(-1) {}

        size_t data_pos;
        dist_t threshold;
        size_t left;
        size_t right;
    };

    DatasetIndexType MakeVPTreeNode(size_t data_pos) {
        nodes_.push_back(VPTreeNode(data_pos));
        return ((size_t)nodes_.size()) - 1;
    }

    void SelectVPTreeRoot(size_t lower, size_t upper) {
        std::uniform_int_distribution<DatasetIndexType> uni(lower, upper - 1);
        size_t root = uni(rng_);
        dataset_.SwapData(lower, root);
        return;
    }

    void PartitionByDistance(DatasetIndexType lower, DatasetIndexType pos,
                             DatasetIndexType upper) {
        dataset_.PartitionByDistance(lower, pos, upper, fstdistfunc_, dist_func_param_);
    }

    DatasetIndexType ConstructVPTree(size_t lower, size_t upper) {
        if (lower >= upper) {
            return -1;
        } else if (lower + 1 == upper) {
            return MakeVPTreeNode(lower);
        } else {
            SelectVPTreeRoot(lower, upper);
            size_t median = (upper + lower) / 2;
            PartitionByDistance(lower, median, upper);
            auto node_pos = MakeVPTreeNode(lower);
            nodes_[node_pos].threshold = fstdistfunc_(
                dataset_.item_at(lower).second, dataset_.item_at(median).second,
                dist_func_param_);
            nodes_[node_pos].left = ConstructVPTree(lower + 1, median);
            nodes_[node_pos].right = ConstructVPTree(median, upper);
            return node_pos;
        }
    }

    void SearchNode(const dist_t* query_ptr, const VPTreeNode& node,
                    ResultType& result, size_t k) {
        dist_t dist = fstdistfunc_(dataset_.item_at(node.data_pos).second,
                                        query_ptr, dist_func_param_);
        if (dist < tau_) {
            if (result.size() == k) {
                result.pop();
            }
            result.push({dist, dataset_.item_at(node.data_pos).first});
            if (result.size() == k) {
                tau_ = result.top().first;
            }
        }
        if (dist < node.threshold) {
            if (node.left != -1 && dist - tau_ <= node.threshold)
                SearchNode(query_ptr, nodes_[node.left], result, k);

            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result, k);
        } else {
            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result, k);

            if (node.left != -1 && dist - tau_ <= node.threshold)
                SearchNode(query_ptr, nodes_[node.left], result, k);
        }
    }

    std::vector<VPTreeNode> nodes_;
    std::mt19937 rng_;
    dist_t tau_;
    Dataset<dist_t> dataset_;
    hnswlib::DISTFUNC <dist_t> fstdistfunc_;
    void *dist_func_param_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_SEARCH_H_

#ifndef FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_

#include <mpi.h>

#include "fast_ann/dataset.h"
#include "hnswlib/hnswlib.h"

namespace fast_ann {

template <typename dist_t>
class VPTreeHNSWSearch {
   public:
    typedef std::priority_queue<std::pair<dist_t, DatasetIndexType> >
        ResultType;

    VPTreeHNSWSearch(hnswlib::SpaceInterface<dist_t>* s,
                     Dataset<dist_t> dataset)
        : dataset_(dataset) {
        std::random_device rd;
        rng_.seed(rd());
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
        levels_ = std::log2(num_procs_ - 1);
        if (rank_ == 0) {
            LOG_DEBUG("Num levels : " << levels_);
        }
        dim_ = dataset.dimension();
        nodes_.reserve(2 * num_procs_ - 1);
        ConstructVPTree(0, dataset_.size(), 0, 0);
        if (rank_ == num_procs_ - 1) {
            MPI_Waitall(mpi_reqs_.size(), mpi_reqs_.data(),
                        new MPI_Status[mpi_reqs_.size()]);
            mpi_reqs_.clear();
        }
    }

    ResultType searchKnn(const dist_t* query_ptr, size_t k) {
        ResultType result;
        tau_ = std::numeric_limits<dist_t>::max();
        SearchNode(query_ptr, nodes_[0], result, 0, 0, k);
        int num_sent = mpi_reqs_.size();
        if (!mpi_reqs_.empty()) {
            MPI_Waitall(mpi_reqs_.size(), mpi_reqs_.data(),
                        new MPI_Status[mpi_reqs_.size()]);
        }
        mpi_reqs_.clear();
        if (rank_ == num_procs_ - 1) {
            float dummy = 0;
            for (int i = 0; i < rank_; i++) {
                MPI_Send(&dummy, 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
            }
            std::pair<dist_t, size_t>* data = new std::pair<dist_t, size_t>[k];
            MPI_Status status;
            while (num_sent--) {
                MPI_Recv(data, sizeof(std::pair<dist_t, size_t>) * k, MPI_BYTE,
                         MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                for (int i = 0; i < k; i++) {
                    result.push(*data);
                    data += 1;
                }
                while (result.size() > k) {
                    result.pop();
                }
            }
        }
        return result;
    }

   private:
    struct VPTreeNode {
        VPTreeNode(int data_pos_t)
            : data_pos(data_pos_t), left(-1), right(-1) {}

        DatasetIndexType data_pos;
        dist_t threshold;
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
        dataset_.PartitionByDistance(lower, pos, upper, fstdistfunc_,
                                     dist_func_param_);
    }

    DatasetIndexType ConstructVPTree(DatasetIndexType lower,
                                     DatasetIndexType upper, int level,
                                     int id) {
        if (rank_ != num_procs_ - 1) {
            Dataset<dist_t>* local_dataset_ =
                Dataset<dist_t>::recvData(num_procs_ - 1, dim_);
            local_algorithm_ = new hnswlib::HierarchicalNSW<dist_t>(
                new hnswlib::L2Space(local_dataset_->dimension()),
                local_dataset_->size());
            for (int i = 0; i < local_dataset_->size(); i++) {
                local_algorithm_->addPoint(local_dataset_->item_at(i).second,
                                           i);
            }
            return 0;
        }
        if (lower >= upper) {
            return -1;
        } else if (lower + 1 == upper) {
            return MakeVPTreeNode(lower);
        } else {
            if (level == levels_) {
                Dataset<dist_t>* local_dataset_ =
                    dataset_.GetSubset(lower, upper);
                MPI_Request req;
                local_dataset_->sendData(id, req);
                mpi_reqs_.push_back(req);
                return -2 - id;
                /**
                if (rank_ == id) {
                    Dataset<dist_t>* local_dataset_ = dataset_.GetSubset(lower,
                upper); local_algorithm_ = new
                hnswlib::HierarchicalNSW<dist_t>(new
                hnswlib::L2Space(local_dataset_->dimension()),
                local_dataset_->size()); for (int i=0; i<local_dataset_->size();
                i++) {
                        local_algorithm_->addPoint(local_dataset_->item_at(i).second,
                i);
                    }
                    return -2 - id;
                }
                return -1;
                **/
            }
            SelectVPTreeRoot(lower, upper);
            DatasetIndexType median = (upper + lower) / 2;
            PartitionByDistance(lower, median, upper);
            auto node_pos = MakeVPTreeNode(lower);
            nodes_[node_pos].threshold =
                fstdistfunc_(dataset_.item_at(lower).second,
                             dataset_.item_at(median).second, dist_func_param_);
            nodes_[node_pos].left =
                ConstructVPTree(lower + 1, median, level + 1, 2 * id);
            nodes_[node_pos].right =
                ConstructVPTree(median, upper, level + 1, 2 * id + 1);
            return node_pos;
        }
    }

    void SearchNode(const dist_t* query_ptr, const VPTreeNode& node,
                    ResultType& result, int level, int id, size_t k) {
        /**
        if (level == levels_ && id == rank_) {
            auto lrs = local_algorithm_->searchKnn(query_ptr, k);
            while (!lrs.empty()) {
                result.push(lrs.top());
                lrs.pop();
            }
            while (result.size() > k) {
                result.pop();
            }
            return;
        }
        **/
        if (rank_ != num_procs_ - 1) {
            while (true) {
                MPI_Status status;
                int msg_size;
                MPI_Probe(num_procs_ - 1, 1, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_FLOAT, &msg_size);
                if (msg_size < dim_) {
                    return;
                }
                dist_t* query_ptr = new dist_t[1];
                MPI_Recv(query_ptr, dim_, MPI_FLOAT, num_procs_ - 1, 1,
                         MPI_COMM_WORLD, &status);
                auto lrs = local_algorithm_->searchKnn(query_ptr, k);
                std::pair<dist_t, size_t>* data =
                    new std::pair<dist_t, size_t>[k];
                std::pair<dist_t, size_t>* cur_pos = data;
                while (!lrs.empty()) {
                    *cur_pos = lrs.top();
                    lrs.pop();
                    cur_pos += 1;
                }
                MPI_Request req;
                MPI_ISend(data, sizeof(std::pair<dist_t, size_t>) * k, MPI_BYTE,
                          num_procs_ - 1, MPI_COMM_WORLD, &req);
                mpi_reqs_.push_back(req);
            }
        }
        if (level == levels_) {
            MPI_Request req;
            MPI_ISend(query_ptr, dim_, MPI_FLOAT, id, 1, MPI_COMM_WORLD, &req);
            mpi_reqs_.push_back(req);
        }
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
                SearchNode(query_ptr, nodes_[node.left], result, level + 1,
                           2 * id, k);

            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result, level + 1,
                           2 * id + 1, k);
        } else {
            if (node.right != -1 && dist + tau_ >= node.threshold)
                SearchNode(query_ptr, nodes_[node.right], result, level + 1,
                           2 * id + 1, k);

            if (node.left != -1 && dist - tau_ <= node.threshold)
                SearchNode(query_ptr, nodes_[node.left], result, level + 1,
                           2 * id, k);
        }
    }

    std::vector<VPTreeNode> nodes_;
    std::mt19937 rng_;
    dist_t tau_;
    int num_procs_;
    int rank_;
    int levels_;
    int dim_;
    Dataset<dist_t> dataset_;
    hnswlib::HierarchicalNSW<dist_t>* local_algorithm_;
    hnswlib::DISTFUNC<dist_t> fstdistfunc_;
    void* dist_func_param_;
    std::vector<MPI_Request> mpi_reqs_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_VP_TREE_HNSW_SEARCH_H_

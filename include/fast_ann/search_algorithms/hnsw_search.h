#ifndef FAST_ANN_SEARCH_ALGORITHMS_HNSW_SEARCH_H_
#define FAST_ANN_SEARCH_ALGORITHMS_HNSW_SEARCH_H_

#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fast_ann/search_algorithm.h"

namespace fast_ann {

template <typename T, typename R, typename DIST>
class HNSWSearch : public SearchAlgorithm<T, R, DIST> {
   public:
    using SearchAlgorithm<T, R, DIST>::dataset_;
    using SearchAlgorithm<T, R, DIST>::k_;
    using SearchAlgorithm<T, R, DIST>::dist_func_;
    using ResultType = typename SearchAlgorithm<T, R, DIST>::ResultType;

    HNSWSearch(Dataset<T> dataset, int k, DatasetIndexType m = 16,
               DatasetIndexType ef_construction = 200,
               DatasetIndexType ef = 100)
        : SearchAlgorithm<T, R, DIST>(dataset, k) {
        std::random_device rd;
        rng_.seed(rd());
        m_ = m;
        max_m_ = m;
        max_m_0_ = 2 * max_m_;
        ef_construction_ = std::max(ef_construction, m);
        ef_ = ef;
        max_level_ = -1;
        ep_ = -1;
        ml_ = 1 / std::log(1.0 * m_);
        DatasetIndexType ds_size = dataset_.size();
        for (DatasetIndexType i = 0; i < ds_size; i++) {
            InsertData(dataset_.item_at(i).first);
        }
    }

    ResultType Search(const T* query_ptr) {
        std::unordered_set<DatasetIndexType> epts;
        epts.insert(ep_);
        for (int l = max_level_; l > 0; l--) {
            ResultType rs = SearchLayer(query_ptr, epts, 1, l);
            epts.clear();
            epts.insert(rs.top().second);
        }
        ResultType result = SearchLayer(query_ptr, epts, ef_, 0);
        while (result.size() > k_) {
            result.pop();
        }
        return result;
    }

   private:
    class GraphLayer {
       public:
        void AddNode(DatasetIndexType data_pos) {
            nodes.push_back(data_pos);
            edge_list[data_pos] = std::vector<DatasetIndexType>();
        }

        void AddLink(DatasetIndexType v1, DatasetIndexType v2) {
            edge_list[v1].push_back(v2);
            edge_list[v2].push_back(v1);
        }

        std::vector<DatasetIndexType> nodes;
        std::unordered_map<DatasetIndexType, std::vector<DatasetIndexType> >
            edge_list;
    };

    void InsertData(DatasetIndexType data_pos) {
        T* data_ptr = dataset_.item_at(data_pos).second;
        int cur_level = GetRandomLevel();
        std::unordered_set<DatasetIndexType> epts;
        epts.insert(ep_);
        for (int l = max_level_; l > cur_level; l--) {
            ResultType rs = SearchLayer(data_ptr, epts, 1, l);
            epts.clear();
            epts.insert(rs.top().second);
        }
        for (int l = cur_level; l > max_level_; l--) {
            graphs_[l] = new GraphLayer();
            graphs_[l]->AddNode(data_pos);
        }
        for (int l = std::min(max_level_, cur_level); l >= 0; l--) {
            GraphLayer* graph = graphs_[l];
            ResultType rs = SearchLayer(data_ptr, epts, ef_construction_, l);
            epts.clear();
            while (rs.size() > m_) {
                epts.insert(rs.top().second);
                rs.pop();
            }
            std::unordered_set<DatasetIndexType> nbrs;
            while (rs.size() > 0) {
                epts.insert(rs.top().second);
                nbrs.insert(rs.top().second);
                graph->AddLink(data_pos, rs.top().second);
                rs.pop();
            }
            DatasetIndexType prune_limit = (l == 0) ? max_m_0_ : max_m_;
            for (const auto& nbr : nbrs) {
                PruneVertex(graph, nbr, prune_limit);
            }
        }
        if (cur_level > max_level_) {
            max_level_ = cur_level;
            ep_ = data_pos;
        }
    }

    int GetRandomLevel() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double level = -std::log(distribution(rng_)) * ml_;
        return (int)level;
    }

    void PruneVertex(GraphLayer* graph, DatasetIndexType vertex,
                     DatasetIndexType prune_limit) {
        if (graph->edge_list[vertex].size() > prune_limit) {
            ResultType rs;
            for (const auto& nbr : graph->edge_list[vertex]) {
                rs.push({-1 * dist_func_->operator()(
                                  dataset_.item_at(vertex).second,
                                  dataset_.item_at(nbr).second,
                                  dataset_.dimension()),
                         nbr});
            }
            graph->edge_list[vertex].clear();
            graph->edge_list[vertex].reserve(prune_limit);
            for (DatasetIndexType i = 0; i < prune_limit; i++) {
                graph->edge_list[vertex].push_back(rs.top().second);
                rs.pop();
            }
        }
    }

    ResultType SearchLayer(const T* query_ptr,
                           std::unordered_set<DatasetIndexType>& enter_points,
                           int local_k, int layer) {
        ResultType candidates, result;
        for (const auto& enter_point : enter_points) {
            R dist =
                dist_func_->operator()(dataset_.item_at(enter_point).second,
                                       query_ptr, dataset_.dimension());
            candidates.push({-dist, enter_point});
            result.push({dist, enter_point});
        }
        GraphLayer* graph = graphs_[layer];
        std::unordered_set<DatasetIndexType>& visited = enter_points;
        while (!candidates.empty()) {
            R cnd_dist = candidates.top().first;
            R frt_dist = result.top().first;
            if (cnd_dist > frt_dist) {
                break;
            }
            DatasetIndexType cnd = candidates.top().second;
            candidates.pop();
            for (auto neighbor : graph->edge_list[cnd]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                }
                R cur_dist =
                    dist_func_->operator()(dataset_.item_at(neighbor).second,
                                           query_ptr, dataset_.dimension());
                if (cur_dist < frt_dist || result.size() < local_k) {
                    candidates.push({-cur_dist, neighbor});
                    result.push({cur_dist, neighbor});
                    if (result.size() > local_k) {
                        result.pop();
                    }
                }
            }
        }
        return result;
    }

    DatasetIndexType m_;
    DatasetIndexType max_m_;
    DatasetIndexType max_m_0_;
    DatasetIndexType ef_construction_;
    DatasetIndexType ef_;
    DatasetIndexType ep_;
    int max_level_;
    double ml_;
    std::mt19937 rng_;
    std::unordered_map<int, GraphLayer*> graphs_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_SEARCH_ALGORITHMS_HNSW_SEARCH_H_

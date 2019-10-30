#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "fast_ann/data_readers/xvecs_reader.h"
#include "fast_ann/log_sinks/console_sink.h"
#include "fast_ann/log_sinks/file_sink.h"
#include "fast_ann/logger.h"
#include "hnswlib/hnswlib.h"
#include "fast_ann/search_algorithms/vp_tree_search.h"

int main(int argc, char **argv) {
    std::string base_vectors_file_name, query_vectors_file_name,
        ground_truth_file_name, log_file_name;
    int cmd_flag;
    while ((cmd_flag = getopt(argc, argv, "b:q:g:l:")) != -1) {
        switch (cmd_flag) {
            case 'b':
                base_vectors_file_name.assign(optarg);
                break;
            case 'q':
                query_vectors_file_name.assign(optarg);
                break;
            case 'g':
                ground_truth_file_name.assign(optarg);
                break;
            case 'l':
                log_file_name.assign(optarg);
                break;
            default:
                std::cerr << "main() : Invalid command line argument"
                          << std::endl;
                exit(1);
        }
    }
    if (base_vectors_file_name.empty() || query_vectors_file_name.empty() ||
        ground_truth_file_name.empty()) {
        std::cerr << "main() : Base vector file, query vector file and ground"
                     "truth file must be specified (use -b -q -g flags)\n";
        exit(1);
    }
    if (log_file_name.empty()) {
        fast_ann::SetLogSink(new fast_ann::ConsoleSink());
    } else {
        std::ofstream log_stream(log_file_name);
        if (!log_stream) {
            std::cerr << "main() : Error opening log file\n";
            exit(1);
        }
        fast_ann::SetLogSink(new fast_ann::FileSink(log_stream));
    }
    fast_ann::SetLogLevel(fast_ann::LogLevel::DEBUG);

    fast_ann::XvecsReader<float> float_reader;
    fast_ann::Dataset<float> base_dataset =
        float_reader.read(base_vectors_file_name);

    hnswlib::L2Space l2space(base_dataset.dimension());
    fast_ann::VPTreeSearch<float> search_algo(&l2space, base_dataset);

    int k = 100;

    fast_ann::Dataset<float> query_dataset =
        float_reader.read(query_vectors_file_name);
    fast_ann::DatasetIndexType num_queries = query_dataset.size();
    std::vector<std::vector<fast_ann::DatasetIndexType>> results(num_queries);
    for (fast_ann::DatasetIndexType i = 0; i < num_queries; i++) {
        auto result = search_algo.searchKnn(query_dataset.item_at(i).second, k);
        results[i].reserve(k);
        while (!result.empty()) {
            results[i].push_back(result.top().second);
            result.pop();
        }
    }

    fast_ann::XvecsReader<int> gt_reader;
    fast_ann::Dataset<int> gt_dataset = gt_reader.read(ground_truth_file_name);
    float sum_recall = 0;
    for (fast_ann::DatasetIndexType i = 0; i < num_queries; i++) {
        auto algo_result = results[i];
        std::vector<fast_ann::DatasetIndexType> gt_result;
        gt_result.reserve(k);
        auto ptr = gt_dataset.item_at(i).second;
        for (int j = 0; j < k; j++) {
            gt_result.push_back(ptr[j]);
        }
        std::sort(algo_result.begin(), algo_result.end());
        std::sort(gt_result.begin(), gt_result.end());
        std::vector<fast_ann::DatasetIndexType> common_el(k);
        auto it = std::set_intersection(algo_result.begin(), algo_result.end(),
                                        gt_result.begin(), gt_result.end(),
                                        common_el.begin());
        sum_recall += (float)(it - common_el.begin()) / k;
    }
    std::cout << "Average recall is : " << sum_recall / num_queries << "\n";

    return 0;
}

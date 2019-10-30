#include <mpi.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "fast_ann/data_readers/xvecs_reader.h"
#include "fast_ann/log_sinks/console_sink.h"
#include "fast_ann/log_sinks/file_sink.h"
#include "fast_ann/logger.h"
#include "hnswlib/hnswlib.h"
#include "fast_ann/search_algorithms/vp_tree_hnsw_search.h"

class Timer {
   public:
    Timer() { start_time = std::chrono::steady_clock::now(); };
    float GetElapsedTime() {
        std::chrono::steady_clock::time_point end_time =
            std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count());
    };
    void reset() { start_time = std::chrono::steady_clock::now(); };

   private:
    std::chrono::steady_clock::time_point start_time;
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &count);

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
    fast_ann::SetLogLevel(fast_ann::LogLevel::INFO);

    fast_ann::XvecsReader<float> float_reader;
    fast_ann::Dataset<float> base_dataset =
        float_reader.read(base_vectors_file_name);

    int k = 100;
    hnswlib::L2Space l2space(base_dataset.dimension());
    fast_ann::VPTreeHNSWSearch<float> search_algo(&l2space, base_dataset);

    fast_ann::Dataset<float> query_dataset =
        float_reader.read(query_vectors_file_name);
    fast_ann::DatasetIndexType num_queries = query_dataset.size();
    MPI_Barrier(MPI_COMM_WORLD);
    Timer timer;
    for (fast_ann::DatasetIndexType i = 0; i < num_queries; i++) {
        auto result = search_algo.searchKnn(query_dataset.item_at(i).second, k);
    }
    std::cout << "Elapsed Time: " << timer.GetElapsedTime() << "\n";

    MPI_Finalize();
    return 0;
}

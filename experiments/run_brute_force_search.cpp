#include "fast_ann/data_readers/xvecs_reader.h"
#include "fast_ann/distances/l2_norm.h"
#include "fast_ann/log_sinks/console_sink.h"
#include "fast_ann/logger.h"
#include "fast_ann/search_algorithms/brute_force_search.h"

int main() {
    // fast_ann::SetLogSink(new fast_ann::ConsoleSink());
    // fast_ann::SetLogLevel(fast_ann::LogLevel::DEBUG);
    std::string sift_file_name("datasets/siftsmall/siftsmall_base.fvecs");
    fast_ann::XvecsReader<float> reader;
    fast_ann::Dataset<float> dataset = reader.read(sift_file_name);
    fast_ann::BruteForceSearch<float, float,
                               fast_ann::L2SquaredNaive<float, float> >
        algorithm(dataset, 100);
    return 0;
}

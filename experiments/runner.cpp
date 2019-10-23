#include "fast_ann/data_readers/xvecs_reader.h"
#include "fast_ann/log_sinks/console_sink.h"
#include "fast_ann/logger.h"

int main() {
    fast_ann::SetLogSink(new fast_ann::ConsoleSink());
    fast_ann::SetLogLevel(fast_ann::LogLevel::DEBUG);
    std::string sift_file_name("datasets/siftsmall/siftsmall_base.fvecs");
    fast_ann::XvecsReader<float> reader;
    fast_ann::Dataset<float> dataset = reader.read(sift_file_name);
    dataset.LogData(1);
    return 0;
}

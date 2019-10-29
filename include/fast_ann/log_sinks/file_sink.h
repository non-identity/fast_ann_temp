#ifndef FAST_ANN_LOG_SINKS_FILE_SINK_H_
#define FAST_ANN_LOG_SINKS_FILE_SINK_H_

#include <fstream>
#include <mutex>

#include "fast_ann/log_sink.h"

namespace fast_ann {

class FileSink : public LogSink {
   public:
    FileSink(std::ofstream &stream) : stream_(stream) {}

    void write(const std::string& message) {
        lock_.lock();
        stream_ << message;
        stream_.flush();
        lock_.unlock();
    }

   private:
    std::ofstream& stream_;
    std::mutex lock_;
};

}  // namespace fast_ann

#endif  // FAST_ANN_LOG_SINKS_FILE_SINK_H_

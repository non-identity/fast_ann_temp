#ifndef FAST_ANN_LOG_SINKS_NULL_SINK_H_
#define FAST_ANN_LOG_SINKS_NULL_SINK_H_

#include <string>

#include "fast_ann/log_sink.h"

namespace fast_ann {

class NullSink : public LogSink {
   public:
    void write(const std::string& message) { return; }
};

}  // namespace fast_ann

#endif  // FAST_ANN_LOG_SINKS_NULL_SINK_H_

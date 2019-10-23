#ifndef FAST_ANN_LOG_SINKS_CONSOLE_SINK_H_
#define FAST_ANN_LOG_SINKS_CONSOLE_SINK_H_

#include <iostream>
#include <string>

#include "fast_ann/log_sink.h"

namespace fast_ann {

class ConsoleSink : public LogSink {
   public:
    void write(const std::string& message) {
        // Printing using cout with one "<<" operator is thread safe.
        std::cout << message;
        // Flushes can interleave.
        std::cout.flush();
    }
};

}  // namespace fast_ann

#endif  // FAST_ANN_LOG_SINKS_CONSOLE_SINK_H_

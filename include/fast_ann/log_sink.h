#ifndef FAST_ANN_LOG_SINK_H_
#define FAST_ANN_LOG_SINK_H_

#include <string>

namespace fast_ann {

class LogSink {
   public:
    virtual void write(const std::string& message) = 0;
    virtual ~LogSink() {}
};

}  // namespace fast_ann

#endif  // FAST_ANN_LOG_SINK_H_

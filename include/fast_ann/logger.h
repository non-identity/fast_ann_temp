#ifndef FAST_ANN_LOGGER_H_
#define FAST_ANN_LOGGER_H_

#include <sstream>
#include <string>

#include "fast_ann/log_sink.h"
#include "fast_ann/log_sinks/null_sink.h"

namespace fast_ann {

const int kLogLineExtraReserveBytes = 64;

enum LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, FATAL = 4, NONE = 5 };

const char* log_level_names[] = {"[DEBUG] ", "[INFO] ", "[WARN] ", "[ERROR] ",
                                 "[FATAL] "};

class Logger {
   public:
    Logger()
        : log_sink_ptr_(new NullSink()), log_level_cutoff_(LogLevel::NONE) {}

    void set_log_sink(LogSink* ls_ptr) {
        delete log_sink_ptr_;
        log_sink_ptr_ = ls_ptr;
    }

    void set_log_level_cutoff(LogLevel ll_cutoff) {
        log_level_cutoff_ = ll_cutoff;
    }

    void log(LogLevel log_level, const char* file_name, int line_no,
             const char* function_name, const std::string& message) {
        if (log_level < log_level_cutoff_) {
            return;
        }
        std::string log_line;
        log_line.reserve(message.length() + kLogLineExtraReserveBytes);
        log_line.append(log_level_names[log_level]);
        log_line.append(file_name);
        log_line.append(":");
        log_line.append(std::to_string(line_no));
        log_line.append(" (");
        log_line.append(function_name);
        log_line.append(") ");
        log_line.append(message);
        log_line.push_back('\n');
        log_sink_ptr_->write(log_line);
        if (log_level == LogLevel::FATAL) {
            exit(1);
        }
    }

   private:
    LogSink* log_sink_ptr_;
    LogLevel log_level_cutoff_;
};

Logger& GetGlobalLogger() {
    static Logger global_logger;
    return global_logger;
}

void SetLogSink(LogSink* ls_ptr) { GetGlobalLogger().set_log_sink(ls_ptr); }

void SetLogLevel(LogLevel ll_cutoff) {
    GetGlobalLogger().set_log_level_cutoff(ll_cutoff);
}

#define LOG(Level_, Message_)                                              \
    GetGlobalLogger().log(Level_, __FILE__, __LINE__, __PRETTY_FUNCTION__, \
                          static_cast<std::ostringstream&>(                \
                              std::ostringstream().flush() << Message_)    \
                              .str())

#ifdef NDEBUG
#define LOG_DEBUG(_) \
    do {             \
    } while (0)
#define LOG_INFO(_) \
    do {            \
    } while (0)
#define LOG_WARN(_) \
    do {            \
    } while (0)
#define LOG_ERROR(_) \
    do {             \
    } while (0)
#define LOG_FATAL(_) \
    do {             \
    } while (0)
#else
#define LOG_DEBUG(Message_) LOG(LogLevel::DEBUG, Message_)
#define LOG_INFO(Message_) LOG(LogLevel::INFO, Message_)
#define LOG_WARN(Message_) LOG(LogLevel::WARN, Message_)
#define LOG_ERROR(Message_) LOG(LogLevel::ERROR, Message_)
#define LOG_FATAL(Message_) LOG(LogLevel::FATAL, Message_)
#endif

}  // namespace fast_ann

#endif  // FAST_ANN_LOGGER_H_

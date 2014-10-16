/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
 *   
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *   THE SOFTWARE.
 */

#ifndef _TEMPEST_LOGGING_HH_
#define _TEMPEST_LOGGING_HH_

#include "tempest/utils/assert.hh"
#include "tempest/utils/patterns.hh"

#include <vector>
#include <fstream>

namespace Tempest
{
enum class LogLevel
{
    Info,    //!< Information about the general execution. It can be safely ignored. Might contain hints for the end-user.
    Debug,   //!< Debug information used for displaying values and general hints about how to fix a particular issue.
    Warning, /*!< \brief Warnings about stuff that might be suspicious or just non-portable functionality that might fail in
              *          strange non-obvious fashion.
              */
    Error,   //!< General failures of execution that might be safely skipped; however, they might lead to unspecified behavior.
    Fatal    //!< Program execution failures that could plain crash the application or result in data corruption.
};

enum LogFlags
{
    TEMPEST_LOG_ASYNCHRONOUS = 1 << 0,
    TEMPEST_LOG_APPEND       = 1 << 1
};

struct LogMessage
{
    LogLevel    level;
    int64           timestamp;
    string          message;
};

void PrintMessage(std::ostream& _stream, const LogMessage& msg);

std::ostream& GetStdOutput(LogLevel level);

class LogFile: public Singleton<LogFile>
{
    std::fstream            m_LogFile;
    LogLevel            m_MinLogLevel;
    uint64                  m_Timestamp;

    typedef std::vector<LogMessage> LogMessages;
    LogMessages             m_LogMessages;
    size_t                  m_CurrentIndex;
    size_t                  m_Flags;
#ifdef _MSC_VER
    std::stringstream       m_MessageBuffer;
#endif
public:
    LogFile(uint32 flags);
    LogFile(const string& filename, LogLevel log_level = LogLevel::Info, uint32 flags = 0);
     ~LogFile();
    
    void flush();
    
    void setLogFile(const string& filename);
    
    void setMinLogLevel(LogLevel log_level);
    LogLevel getMinLogLevel();

    void pushMessage(LogLevel log_level, string msg);
    string readLog();

    inline static string read() { return LogFile::getSingleton().readLog(); }
private:
    void flushStream(std::ostream& os);
    void flushCurrentMessage();
    std::ostream& getOutputStream(LogLevel level);
};

inline void _LogPrintImpl(std::ostream& os) {}

template<class T, class... TArgs>
inline void _LogPrintImpl(std::ostream& os, T&& arg, TArgs&&... args)
{
    os << arg;
    _LogPrintImpl(os, args...);
}

void Log(LogLevel log_level, string msg);

template<class... TArgs>
void Log(LogLevel log_level, TArgs&&... args)
{
    std::stringstream ss;
    _LogPrintImpl(ss, args...);
    Log(log_level, ss.str());
}
}

#endif // _TEMPEST_LOGGING_HH_
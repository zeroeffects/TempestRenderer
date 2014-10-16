/*   The MIT License
 *   
 *   Tempest Common
 *   Copyright (c) 2010-2011 Zdravko Velinov
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

#include "tempest/utils/logging.hh"
#include "tempest/utils/types.hh"

#include <chrono>
#include <iostream>

// TODO: Optimize

namespace Tempest
{
string ConvertLogLevelToString(LogLevel level)
{
    switch(level)
    {
    case LogLevel::Error: return "[ERROR]"; break;
    case LogLevel::Fatal: return "[FATAL ERROR]"; break;
    case LogLevel::Warning: return "[WARNING]"; break;
    case LogLevel::Debug: return "[DEBUG]"; break;
    case LogLevel::Info: return "[INFO]"; break;
    default: TGE_ASSERT(false, "Unknown logging level");
    }
    return "[UNKNOWN]";
}

LogFile::LogFile(uint32 flags)
    :   m_MinLogLevel(LogLevel::Info),
		m_CurrentIndex(0),
		m_Flags(flags)
{
}

LogFile::LogFile(const string& filename, LogLevel log_level, uint32 flags)
    :   m_LogFile(filename.c_str(), std::ios::out | std::ios::trunc),
        m_MinLogLevel(log_level),
        m_CurrentIndex(0),
        m_Flags(flags)
{
}

LogFile::~LogFile()
{
    flush();
}

void LogFile::flush()
{
    for(auto iter = m_LogMessages.begin() + m_CurrentIndex, iter_end = m_LogMessages.end();
        iter != iter_end; ++iter)
    {
        auto& _stream = getOutputStream(iter->level);
        PrintMessage(_stream, *iter);
        flushStream(_stream.flush()); // TODO: nope
    }
    m_CurrentIndex = m_LogMessages.size();
}

void LogFile::setLogFile(const string& filename)
{
    m_LogFile.open(filename.c_str(), std::ios::out | std::ios::trunc);
}

void LogFile::setMinLogLevel(LogLevel log_level)
{
    m_MinLogLevel = log_level;
}

LogLevel LogFile::getMinLogLevel()
{
    return m_MinLogLevel;
}

std::ostream& GetStdOutput(LogLevel level)
{
    if(level == LogLevel::Error ||
       level == LogLevel::Fatal ||
       level == LogLevel::Warning)
        return std::cerr;
    return std::cout;
}

std::ostream& LogFile::getOutputStream(LogLevel level)
{
    if(m_LogFile.is_open())
        return m_LogFile;
    // On Windows it uses an intermediate stream to output to the Output Window.
#ifdef _MSC_VER
    return m_MessageBuffer;
#else
    return GetStdOutput(level);
#endif
}

void LogFile::flushStream(std::ostream& os)
{
    // HACK: It flushes the data to the Output Window in the case of Windows; otherwise, it
    // works in the usual fashion.
#ifdef _MSC_VER
    if(!m_LogFile.is_open())
    {
        auto str = m_MessageBuffer.str();
        OutputDebugString(str.c_str());
        m_MessageBuffer.str("");
    }
#else
    os.flush();
#endif
}

string LogFile::readLog()
{
    std::stringstream ss;
    for(auto& msg : m_LogMessages)
    {
        PrintMessage(ss, msg);
    }
    return ss.str();
}

void LogFile::pushMessage(LogLevel log_level, string msg)
{
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    m_LogMessages.push_back(LogMessage{ log_level, now.count(), msg });
    if((m_Flags & TEMPEST_LOG_ASYNCHRONOUS) == 0)
    {
#ifdef _MSC_VER
        PrintMessage(m_MessageBuffer, m_LogMessages.back());
        OutputDebugString(m_MessageBuffer.str().c_str());
        m_MessageBuffer.str("");
#else
        PrintMessage(getOutputStream(log_level), m_LogMessages.back());
#endif
        ++m_CurrentIndex;
    }
}

void PrintMessage(std::ostream& _stream, const LogMessage& msg)
{
    uint64 m = msg.timestamp / 60000000000LL;
    uint64 s = (msg.timestamp % 60000000000LL) / 1000000000LL;
    uint64 ns = (msg.timestamp % 1000000000LL);
    _stream << m << ":" << s << "." << ns << ": " << ConvertLogLevelToString(msg.level) << ": " << msg.message << "\n";
}

void Log(LogLevel log_level, string msg_str)
{
    auto* ptr = LogFile::getSingletonPtr();
    if(ptr)
    {
        ptr->pushMessage(log_level, msg_str);
    }
    else
    {
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        LogMessage msg{ log_level, now.count(), msg_str };
        PrintMessage(GetStdOutput(log_level), msg);
    }
}
}

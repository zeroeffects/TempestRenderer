/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2013 Zdravko Velinov
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

#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

#if !defined(NDEBUG) && !defined(_WIN32) && defined(HAS_QT4)
#   define QT_DEBUG_GUI
#endif

#if defined(_MSC_VER)
#   include <intrin.h>
#   define TGE_TRAP() __debugbreak()
#elif defined(LINUX)
#   include <signal.h>
#   include <execinfo.h>
#   include <cxxabi.h>
#   include <cstring>
#   include <vector>
#   define TGE_TRAP() __builtin_trap()
#endif

#ifdef QT_DEBUG_GUI
#   include <QMessageBox>
#endif


namespace Tempest
{
#ifdef TGE_LOG_ASSERTS
// These ones are for testing purposes only. They are a little bit too aggressive
DialogAnswer AssertMessageBox(const string& title, const string& doc_msg)
{
    Log(TGE_LOG_ERROR, title, ": ", doc_msg);
    FlustLog();
    TGE_TRAP();
    return TGE_ANSWER_ABORT;
}

void CrashMessageBox(const string& title, const string& doc_msg)
{
    Tempest::Log::stream(TGE_LOG_ERROR) << title << ": " << doc_msg << std::endl;
    Tempest::Log::getSingleton().flush();
    TGE_TRAP();
}
#elif defined(_WIN32)
DialogAnswer AssertMessageBox(const string& title, const string& doc_msg)
{
    auto res = MessageBox(nullptr, doc_msg.c_str(), title.c_str(), MB_ABORTRETRYIGNORE|MB_ICONERROR|MB_SYSTEMMODAL);
    switch(res)
    {
    default:
    case IDABORT: return TGE_ANSWER_ABORT;
    case IDRETRY: return TGE_ANSWER_RETRY;
    case IDIGNORE: return TGE_ANSWER_IGNORE;    
    }
}

void CrashMessageBox(const string& title, const string& doc_msg)
{
    MessageBox(nullptr, doc_msg.c_str(), title.c_str(), MB_OK|MB_ICONERROR|MB_SYSTEMMODAL);
}

#elif defined(LINUX)
// Qt because it is used for development in the other parts of the toolkit 
DialogAnswer AssertMessageBox(const string& title, const string& doc_msg)
{
#ifdef QT_DEBUG_GUI
    QMessageBox mb(QMessageBox::Critical, title.c_str(), doc_msg.c_str(), QMessageBox::Abort|QMessageBox::Retry|QMessageBox::Ignore);
    mb.setDetailedText(Tempest::Log::read().c_str());
    auto res = mb.exec();
    switch(res)
    {
    default:
    case QMessageBox::Abort: return TGE_ANSWER_ABORT;
    case QMessageBox::Retry: return TGE_ANSWER_RETRY;
    case QMessageBox::Ignore: return TGE_ANSWER_IGNORE;
    }
#else
    std::cerr << title << "\n" << doc_msg << std::endl;
    return TGE_ANSWER_ABORT;
#endif
}

void CrashMessageBox(const string& title, const string& doc_msg)
{
#ifdef QT_DEBUG_GUI
    QMessageBox mb(QMessageBox::Critical, title.c_str(), doc_msg.c_str(), QMessageBox::Ok);
    mb.setDetailedText(QString(Tempest::Log::read().c_str()));
    mb.exec();
#else
    std::cerr << title << "\n" << doc_msg << std::endl;
#endif
}
#else
#   error "Unsupported platform"
#endif
    
#if defined(_WIN32)
string Backtrace(size_t start_frame, size_t end_frame)
{
    TGE_ASSERT(false, "Unimplemented. TODO: Stack walk");
    return string();
}
#elif defined(LINUX)
string Backtrace(size_t start_frame, size_t end_frame)
{
    string result = "Backtrace:\n"
                    "==========\n";
    std::vector<void *> array(end_frame);
    size_t size;
    int status;
    
    auto strings = CREATE_SCOPED(char**, ::free);
    
    size = ::backtrace(&array.front(), array.size());
    strings = ::backtrace_symbols(&array.front(), size);
    
    size_t end = std::min(end_frame, size);
    for(size_t i = start_frame; i < end; ++i)
    {
        auto realname = CREATE_SCOPED(char*, ::free);
        auto* begin = ::strchr(strings[i], '(') + 1;
        if(!strings[i])
        {
            result += "??\n";
            continue;
        }
        auto* end = ::strchr(begin, '+');
        if(!end)
            end = ::strchr(begin, ')');
        string mangled_name(begin, end);
        realname = abi::__cxa_demangle(mangled_name.c_str(), 0, 0, &status);
        result.insert(result.end(), strings[i], begin);
        if(realname)
            result += realname;
        else
            result.insert(result.end(), begin, end);
        result.insert(result.size(), end);
        result += "\n";
    }

    return result;
}
#else
#   error "Unsupported platform"
#endif
}

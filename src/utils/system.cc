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

#include "tempest/utils/system.hh"
#include <cstdint>
#include <algorithm>
#include <mutex>

#ifdef _WIN32
#   include <windows.h>
#   ifdef __MINGW32__
#       define off64_t _off64_t
#   endif
#   include <io.h>
#   include <sys/utime.h>
#   include <Fcntl.h>
#   include <stdio.h>
#   include <tchar.h>
#   define ACCESS _access
#elif defined(LINUX)
#   include <unistd.h>
#   include <utime.h>
#   include <sys/sendfile.h>
#   include <sys/types.h>
#   include <sys/stat.h>
#   include <fcntl.h>
#	define ACCESS access
#else
#   error "Unsupported platform"
#endif

#ifndef _MSC_VER
#   include <cstdlib>
#endif

#include <thread>

namespace Tempest
{
namespace System
{
bool GetEnvironmentVariable(const std::string& name, std::string& result)
{
#ifdef _MSC_VER
    char    *buf;
    size_t  len;
    int err = _dupenv_s(&buf, &len, name.c_str());
    if(err)
        return false;
    result = buf;
    free(buf);
#else
    char* r = getenv(name.c_str());
    if(!r)
        return false;
    result = r;
#endif
    return true;
}

bool SetEnvironmentVariable(const std::string& name, const std::string& val)
{
#ifdef _WIN32
    if(::SetEnvironmentVariable(name.c_str(), val.c_str()))
        return false;
#elif defined(LINUX)
    if(setenv(name.c_str(), val.c_str(), 1))
        return false;
#else
#	error "Unsupported platform"
#endif
    return true;
}

std::string GetExecutablePath()
{
    char buffer[1024];
#ifdef _WIN32
    if(GetModuleFileName(nullptr, buffer, sizeof(buffer)) == 0)
        return std::string();
#elif defined(LINUX)
    size_t len = readlink("/proc/self/exe", buffer, 1023);
    if(len < 0)
        return std::string();
    buffer[len] = 0;
#else
#	error "Unsupported platform"
#endif
    return buffer;
}

bool Exists(const std::string& name)
{
    return ACCESS(name.c_str(), 0) == 0;
}

bool FileCopy(const std::string& source, const std::string& destination)
{
#ifdef _WIN32
	return CopyFile(source.c_str(), destination.c_str(), false) == TRUE;
#else
	auto input = CreateScoped(open(source.c_str(), O_RDONLY), [](int fd) { if(fd >= 0) { close(fd); } });
	if(input.get() < -1)
    {
        return false;
    }

	auto output = CreateScoped(open(destination.c_str(), O_RDWR | O_CREAT), [](int fd) { if(fd >= 0) { close(fd); } });
    if(output.get() < -1)
    {
        return false;
    }

#	if defined(__APPLE__) || defined(__FreeBSD__)
    int result = fcopyfile(input, output, 0, COPYFILE_ALL);
    if(result < 0)
    {
        return false;
    }
#	else
    off_t byte_count = 0;
    struct stat fileinfo = {0};
    fstat(input.get(), &fileinfo);
    int result = sendfile(output.get(), input.get(), &byte_count, fileinfo.st_size);
    if(result < 0)
    {
        return false;
    }
#	endif

	return true;
#endif
}

uint32_t GetNumberOfProcessors()
{
    uint32_t concurrency = std::thread::hardware_concurrency();
    if(concurrency)
        return concurrency;
#ifdef _WIN32
    return GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

bool Touch(const std::string& filename)
{
    auto cur_time = time(nullptr);

    utimbuf t;
    t.actime = cur_time;
    t.modtime = cur_time;

    int ret = utime(filename.c_str(), &t);
    return ret != 0;
}

#ifdef _WIN32
const DWORD MS_VC_EXCEPTION = 0x406D1388;
#pragma pack(push,8)
typedef struct tagTHREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
 } THREADNAME_INFO;
#pragma pack(pop)

// Well, Windows is stupid, so hack around the limitation
static std::mutex g_ThreadInfoMutex;

struct ThreadInfo
{
    std::thread::native_handle_type Handle;
    std::string                     Name;
};

static std::vector<ThreadInfo> g_ThreadInfo;
#endif

#ifdef _WIN32
void SetThreadName(std::thread::native_handle_type handle, const char* name)
{
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = name;
    info.dwThreadID = ::GetThreadId(static_cast<HANDLE>(handle));
    info.dwFlags = 0;
#   pragma warning(push)
#   pragma warning(disable: 6320 6322)
    __try{
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except (EXCEPTION_EXECUTE_HANDLER){
    }
#   pragma warning(pop)

}
#endif

void SetThreadName(std::thread::native_handle_type handle, const std::string& name)
{
#ifdef _WIN32
    SetThreadName(handle, name.c_str());

    {
    std::lock_guard<std::mutex> lock(g_ThreadInfoMutex);
    auto iter = std::find_if(g_ThreadInfo.begin(), g_ThreadInfo.end(), [handle](const ThreadInfo& info) { return info.Handle == handle; });
    if(iter != g_ThreadInfo.end())
    {
        iter->Name = name.c_str();
    }
    else
    {
        g_ThreadInfo.emplace_back(ThreadInfo{ handle, name.c_str() });
    }
    }

#else
    pthread_setname_np((pthread_t)handle, name.c_str());
#endif
}

std::string GetThreadName(std::thread::native_handle_type handle)
{
#ifdef _WIN32
    auto iter = std::find_if(g_ThreadInfo.begin(), g_ThreadInfo.end(), [handle](const ThreadInfo& info) { return info.Handle == handle; });
    if(iter != g_ThreadInfo.end())
    {
        return iter->Name;
    }
    else
    {
        return "Unknown Thread";
    }
#else
    char buffer[1024];

    pthread_getname_np((pthread_t)handle, buffer, TGE_FIXED_ARRAY_SIZE(buffer));
    return buffer;
#endif
}

void OpenConsoleConnection()
{
#ifdef _WIN32
    AllocConsole();

    auto win_stdout = GetStdHandle(STD_OUTPUT_HANDLE);
    auto stdout_hnd = _open_osfhandle((intptr_t)win_stdout, _O_WRONLY);
    if(stdout_hnd != -1)
    {
        *stdout = *_tfdopen(stdout_hnd, "a");
        std::cout.sync_with_stdio();
    }

    auto win_stderr = GetStdHandle(STD_ERROR_HANDLE);
    auto stderr_hnd = _open_osfhandle((intptr_t)win_stderr, _O_WRONLY);
    if(stderr_hnd != -1)
    {
        *stderr = *_tfdopen(stderr_hnd, "a");
        std::cerr.sync_with_stdio();
    }
#endif
}

std::thread::native_handle_type GetCurrentThreadNativeHandle()
{
#ifdef _WIN32
    return ::GetCurrentThread();
#else
    return pthread_self();
#endif
}
}
}

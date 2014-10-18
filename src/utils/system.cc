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
#include "tempest/utils/types.hh"

#ifdef _WIN32
#   include <windows.h>
#   ifdef __MINGW32__
#       define off64_t _off64_t
#   endif
#   include <io.h>
#elif defined(LINUX)
#   include <unistd.h>
#else
#   error "Unsupported platform"
#endif

#ifndef _MSC_VER
#   include <cstdlib>
#endif

namespace Tempest
{
namespace System
{
bool GetEnvironmentVariable(const string& name, string& result)
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

bool SetEnvironmentVariable(const string& name, const string& val)
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

string GetExecutablePath()
{
    char buffer[1024];
#ifdef _WIN32
    if(GetModuleFileName(nullptr, buffer, sizeof(buffer)) == 0)
        return string();
#elif defined(LINUX)
    size_t len = readlink("/proc/self/exe", buffer, 1023);
    if(len < 0)
        return string();
    buffer[len] = 0;
#else
#	error "Unsupported platform"
#endif
    return buffer;
}

bool Exists(const string& name)
{
#ifdef _WIN32
#   define _access access
#elif !defined(LINUX)
#   error "Unsupported platform"
#endif
    return access(name.c_str(), 0) == 0;
}
}
}

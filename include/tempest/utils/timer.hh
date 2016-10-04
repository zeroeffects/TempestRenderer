/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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

#ifndef _TEMPEST_TIMER_HH_
#define _TEMPEST_TIMER_HH_

#include <cstdint>

#ifdef LINUX
#   include <sys/time.h>
#elif defined(_WIN32)
#   define WIN32_LEAN_AND_MEAN 1
#   include <Windows.h>
#else
#   error "Unsupported platform"
#endif

namespace Tempest
{
#ifdef _WIN32
class TimeQuery
{
    LARGE_INTEGER m_Frequency;
public:
    TimeQuery()
    {
        if(!QueryPerformanceFrequency(&m_Frequency))
        {
            m_Frequency.QuadPart = 1LL;
        }
    }

    int64_t time()
    {
        LARGE_INTEGER time;
        if(!QueryPerformanceCounter(&time))
        {
            time.QuadPart = 0LL;
        }

        return time.QuadPart / (m_Frequency.QuadPart / 1000000ULL);
    }
};
#elif defined(LINUX)
class TimeQuery
{
public:
    TimeQuery() = default;

    int64_t time()
    {
        timeval tv;
        if(!gettimeofday(&tv, nullptr))
            return 0;
        return tv.tv_sec*10000000ULL + (uint64_t)tv.tv_usec;
    }
};
#else
#   error "Unsupported platform"
#endif

}

#endif // _TEMPEST_TIMER_HH_

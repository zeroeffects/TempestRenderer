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

#include <cstdint>
#include "tempest/utils/patterns.hh"

#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>

namespace Tempest
{
std::string GetLastErrorString()
{
    auto err = ::GetLastError();
    auto msg = CREATE_SCOPED(LPSTR, LocalFree);
    msg = nullptr;
    DWORD res = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM |
                                FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                FORMAT_MESSAGE_IGNORE_INSERTS,
                                nullptr,
                                err,
                                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                (LPTSTR)&msg,
                                0,
                                nullptr);
    if(res && msg)
        return static_cast<char*>(msg);
    else
    {
        std::stringstream ss;
        ss << "Generic error: 0x" << std::hex << err;
        return ss.str();
    }
}
}
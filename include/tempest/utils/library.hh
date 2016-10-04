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

#ifndef LIBRARY_HH_
#define LIBRARY_HH_

#ifdef _WIN32
#   include <windows.h>
#elif defined(LINUX)
#   include <dlfcn.h>
#else
#	error "Unsupported platform"
#endif

#include <string>

namespace Tempest
{

#ifdef _WIN32
    typedef HMODULE LibType;
#   define GetProcAddress(lib, symb) reinterpret_cast<ProcType>(GetProcAddress(lib, symb))
#elif defined(LINUX)
    typedef void* LibType;
#   define LoadLibrary(name) dlopen(name, RTLD_LAZY | RTLD_GLOBAL)
#   define GetProcAddress(lib, symb) dlsym(lib, symb)
#   define FreeLibrary(lib) dlclose(lib)
#else
#	error "Unsupported platform"
#endif

typedef void(*ProcType)(void);

class Library
{
    LibType m_Lib;
public:
    Library();
    Library(const std::string& name);
     ~Library();

    bool loaded() const { return m_Lib != nullptr; }
    bool load(const std::string& name);
    void free();

    ProcType getProcAddress(const std::string& str);
};
}

#endif /* LIBRARY_HH_ */

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

#include "tempest/utils/library.hh"
#include "tempest/utils/logging.hh"
#include <cassert>

namespace Tempest
{
Library::Library()
    :   m_Lib(0) {}

Library::Library(const string& name)
    :   m_Lib(LoadLibrary(name.c_str()))
{
}

Library::~Library()
{
    if(m_Lib)
        FreeLibrary(m_Lib);
}

bool Library::load(const string& name)
{
    this->free();
    m_Lib = LoadLibrary(name.c_str());
    return m_Lib != 0;
}

void Library::free()
{
    if(m_Lib)
        FreeLibrary(m_Lib);
    m_Lib = 0;
}

string GetLastErrorString();

ProcType Library::getProcAddress(const string& str)
{
    if(!m_Lib)
        return ProcType();
    union
    {
        ProcType proc;
        void* symbol;
    } ptr;
#ifdef _WIN32
    ptr.proc = GetProcAddress(m_Lib, str.c_str());
    if(ptr.proc == nullptr)
    {
        Log(LogLevel::Error, "Failed to find symbol within library ", str, ": ", GetLastErrorString());
        return nullptr;
    }
#elif defined(LINUX)
    ptr.symbol = GetProcAddress(m_Lib, str.c_str());
#else
#   error "Unsupported platform"
#endif
    return ptr.proc;
}
}

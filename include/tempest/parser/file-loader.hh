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

#ifndef TEMPEST_FILE_LOADER_HH_
#define TEMPEST_FILE_LOADER_HH_

#include <cstdint>
#include "tempest/utils/patterns.hh"

namespace Tempest
{
// Hint: if you need more data just concat it after the initial header
struct FileDescription
{
    const char* Content;
    size_t ContentSize;
};
    
class FileLoader
{
public:
    virtual FileDescription* loadFileContent(const std::string& name)=0;
    virtual void freeFileContent(FileDescription* ptr)=0;
};

#define CREATE_SCOPED_FILE(loader, name) CreateScoped<FileDescription*>(loader->loadFileContent(name), [=](FileDescription* ptr){ loader->freeFileContent(ptr); });
}

#endif // TEMPEST_FILE_LOADER_HH_
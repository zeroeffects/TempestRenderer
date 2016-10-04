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

#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/parser/file-loader.hh"

#include <fstream>

namespace Tempest
{
FileDescription* BasicFileLoader::loadFileContent(const std::string& name)
{
    std::fstream fs(name, std::ios::in|std::ios::binary);
    if(!fs)
        return nullptr;

    auto start = fs.tellg();
    fs.seekg(0, std::ios::end);
    size_t size = fs.tellg() - start;
    fs.seekg(0, std::ios::beg);
    
    char* data = reinterpret_cast<char*>(malloc(sizeof(FileDescription) + size + 1));
    FileDescription* header = reinterpret_cast<FileDescription*>(data);
    header->Content = data + sizeof(FileDescription);
    ((char*)header->Content)[size] = 0;
    header->ContentSize = size;

    fs.read((char*)header->Content, size);
    fs.close();        
    
    return header;
}

void BasicFileLoader::freeFileContent(FileDescription* ptr)
{
    free(ptr);
}

SubdirectoryFileLoader::SubdirectoryFileLoader(const std::string& subdir)
    :   m_Subdirectory(subdir + "/")
{
}

FileDescription* SubdirectoryFileLoader::loadFileContent(const std::string& name)
{
    std::fstream fs(m_Subdirectory + name, std::ios::in|std::ios::binary);
    if(!fs)
		fs.open(name, std::ios::in|std::ios::binary);

	if(!fs)
        return nullptr;

    auto start = fs.tellg();
    fs.seekg(0, std::ios::end);
    size_t size = fs.tellg() - start;
    fs.seekg(0, std::ios::beg);
    
    char* data = reinterpret_cast<char*>(malloc(sizeof(FileDescription) + size + 1));
    FileDescription* header = reinterpret_cast<FileDescription*>(data);
    header->Content = data + sizeof(FileDescription);
    ((char*)header->Content)[size] = 0;
    header->ContentSize = size;

    fs.read((char*)header->Content, size);
    fs.close();        
    
    return header;
}

void SubdirectoryFileLoader::freeFileContent(FileDescription* ptr)
{
    free(ptr);
}
}
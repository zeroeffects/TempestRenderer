/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2014 Zdravko Velinov
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

#include "tempest/mesh/obj-loader-driver.hh"
#include "tempest/mesh/obj-mtl-loader-driver.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/utils/file-system.hh"

namespace Tempest
{
namespace ObjLoader
{
void Driver::parseMaterialFile(const Location& loc, const string& name)
{
    ObjMtlLoader::Driver obj_mtl_driver(&m_Materials);
    if(m_FileLoader)
    {
        auto* file_descr = m_FileLoader->loadFileContent(m_Path + TGE_PATH_DELIM + name);
        auto* loader = m_FileLoader;
        CreateAtScopeExit([loader, file_descr]() { loader->freeFileContent(file_descr); });
        obj_mtl_driver.parseString(file_descr->Content, file_descr->ContentSize, name);
    }
    else
    {
        auto parse_ret = obj_mtl_driver.parseFile(m_Path + TGE_PATH_DELIM + name);
        if(!parse_ret)
        {
            std::stringstream ss;
            ss << "The application has failed to parse a material file (refer to the error log for more information): " << name << std::endl;
            error(loc, ss.str());
            TGE_ASSERT(parse_ret, ss.str());
        }
    }
}

void Driver::pushMaterial(const Location& loc, const string& name)
{
    auto beg_iter = std::begin(m_Materials), end_iter = std::end(m_Materials);
    auto iter = std::find_if(beg_iter, end_iter, [&name](const ObjMtlLoader::Material& mtl) { return mtl.Name == name; });
    if(iter == end_iter)
    {
        error(loc, "Unknown material: " + name);
        return;
    }
    m_CurrentMaterial = iter - beg_iter;
}
}
}
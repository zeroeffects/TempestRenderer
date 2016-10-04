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
 
#ifndef _OBJ_MTL_LOADER_DRIVER_HH_
#define _OBJ_MTL_LOADER_DRIVER_HH_

#include "tempest/parser/driver-base.hh"
#include "tempest/mesh/obj-mtl-loader.hh"

#include <vector>

namespace Tempest
{
namespace ObjMtlLoader
{
class Driver: public DriverBase
{
    ObjMtlLoader::Material*              m_CurrentMaterial = nullptr;
    std::vector<ObjMtlLoader::Material>* m_Materials;
    
public:
    Driver(FileLoader* loader, std::vector<ObjMtlLoader::Material>* out_materials)
        :   DriverBase(loader),
            m_Materials(out_materials) {}
     ~Driver()=default;
    
    void pushNewMaterial(std::string name)
	{
		std::transform(std::begin(name), std::end(name), std::begin(name), ::tolower);
		m_Materials->push_back(ObjMtlLoader::Material());
		m_Materials->back().Name = name; 
		m_CurrentMaterial = &m_Materials->back();
	}
    ObjMtlLoader::Material* getCurrentMaterial() { return m_CurrentMaterial; }
    
    bool parseFile(const std::string& filename);
    bool parseString(const char* str, size_t size, const std::string& filename);
};
}
}

#endif // _OBJ_MTL_LOADER_DRIVER_HH_
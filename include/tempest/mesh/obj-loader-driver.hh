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
 
#ifndef _OBJ_LOADER_DRIVER_HH_
#define _OBJ_LOADER_DRIVER_HH_

#include <cstddef>

#include <cstdint>
#include "tempest/parser/driver-base.hh"
#include "tempest/mesh/obj-mtl-loader.hh"

#include "tempest/math/vector4.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/vector2.hh"

#include <vector>

namespace Tempest
{
namespace ObjLoader
{
struct GroupHeader
{
    std::string Name;
    uint32_t    PositionStart,
                TexCoordStart,
                NormalStart,
                MaterialIndex;
};
    
class Driver: public DriverBase
{
protected:
            
    std::vector<Vector3>     m_Position;
    std::vector<Vector2>     m_TexCoord;
    std::vector<Vector3>     m_Normal;
    
    std::vector<int32_t>     m_PositionIndices;
    std::vector<int32_t>     m_TexCoordIndices;
    std::vector<int32_t>     m_NormalIndices;
    std::vector<GroupHeader> m_Groups;
    
    uint32_t                 m_CurrentMaterial = 0;
    
    std::vector<ObjMtlLoader::Material> m_Materials;
    
    std::string              m_Path;
    
public:
    Driver(std::string path, FileLoader* loader)
        :   DriverBase(loader),
            m_Path(path) {}
     ~Driver()=default;

    FileLoader* getFileLoader() { return m_FileLoader; }
     
    void pushPosition(float px, float py, float pz, float pw = 1.0f) { float factor = 1.0f / pw; m_Position.emplace_back(Vector3{px * factor, py * factor, pz * factor}); }
    void pushTexCoord(float tx, float ty) { m_TexCoord.push_back(Tempest::Vector2{tx, ty}); }
    void pushNormal(float nx, float ny, float nz) { m_Normal.push_back(Tempest::Vector3{nx, ny, nz}); }
    
    void pushMaterial(const Location& loc, const std::string& name);
    void pushGroup(const std::string& name) { m_Groups.push_back(GroupHeader{ name, (uint32_t)m_PositionIndices.size(), (uint32_t)m_TexCoordIndices.size(), (uint32_t)m_NormalIndices.size(), m_CurrentMaterial }); }
    void pushPositionIndex(int32_t idx) { m_PositionIndices.push_back(idx >= 0 ? idx - 1: ((int32_t)m_Position.size() + idx)); }
    void pushTexCoordIndex(int32_t idx) { m_TexCoordIndices.push_back(idx >= 0 ? idx - 1: ((int32_t)m_TexCoord.size() + idx)); }
    void pushNormalIndex(int32_t idx) { m_NormalIndices.push_back(idx >= 0 ? idx - 1: ((int32_t)m_Normal.size() + idx)); }
    
    const std::vector<Vector3>& getPositions() const { return m_Position; }
    const std::vector<Vector2>& getTexCoords() const { return m_TexCoord; }
    const std::vector<Vector3>& getNormals() const { return m_Normal; }
    
    const std::vector<int32_t>& getPositionIndices() const { return m_PositionIndices; }
    const std::vector<int32_t>& getTexCoordIndices() const { return m_TexCoordIndices; }
    const std::vector<int32_t>& getNormalIndices() const { return m_NormalIndices; }
    
    const std::vector<ObjMtlLoader::Material>& getMaterials() const { return m_Materials; }
    
    void parseMaterialFile(const Location& loc, const std::string& name);
    
    const std::vector<GroupHeader>& getGroups() const { return m_Groups; }
    
    bool parseFile(const std::string& filename);
    bool parseString(const char* str, size_t size, const std::string& filename);
private:
};
}
}

#endif // _OBJ_LOADER_DRIVER_HH_
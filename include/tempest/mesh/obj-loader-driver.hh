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

#include "location.hh"

#include "tempest/utils/types.hh"
#include "tempest/parser/driver-base.hh"
#include "tempest/mesh/obj-mtl-loader.hh"

#include "tempest/math/vector4.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/vector2.hh"

namespace Tempest
{
class FileLoader;
    
namespace ObjLoader
{
struct GroupHeader
{
    string Name;
    uint32 PositionStart,
           TexCoordStart,
           NormalStart,
           MaterialIndex;
};
    
class Driver: public DriverBase
{
protected:
    size_t                   m_ErrorCount = 0,
                             m_WarningCount = 0;
                            
    std::vector<Vector4>     m_Position;
    std::vector<Vector3>     m_TexCoord;
    std::vector<Vector3>     m_Normal;
    
    std::vector<int32>       m_PositionIndices;
    std::vector<int32>       m_TexCoordIndices;
    std::vector<int32>       m_NormalIndices;
    std::vector<GroupHeader> m_Groups;
    
    uint32                   m_CurrentMaterial = 0;
    
    std::vector<ObjMtlLoader::Material> m_Materials;
    
    FileLoader*              m_FileLoader;
    string                   m_Path;
    
public:
    Driver(string path, FileLoader* loader)
        :   m_FileLoader(loader),
            m_Path(path) {}
     ~Driver()=default;

    FileLoader* getFileLoader() { return m_FileLoader; }
     
    void pushPosition(float px, float py, float pz, float pw = 1.0f) { m_Position.push_back(Tempest::Vector4(px, py, pz, pw)); }
    void pushTexCoord(float tx, float ty, float tz = 0.0f) { m_TexCoord.push_back(Tempest::Vector3(tx, ty, tz)); }
    void pushNormal(float nx, float ny, float nz) { m_Normal.push_back(Tempest::Vector3(nx, ny, nz)); }
    
    void pushMaterial(const Location& loc, const string& name);
    void pushGroup(const string& name) { m_Groups.push_back(GroupHeader{ name, (uint32)m_PositionIndices.size(), (uint32)m_TexCoordIndices.size(), (uint32)m_NormalIndices.size(), m_CurrentMaterial }); }
    void pushPositionIndex(int32 idx) { m_PositionIndices.push_back(idx); }
    void pushTexCoordIndex(int32 idx) { m_TexCoordIndices.push_back(idx); }
    void pushNormalIndex(int32 idx) { m_NormalIndices.push_back(idx); }
    
    const std::vector<Vector4>& getPositions() const { return m_Position; }
    const std::vector<Vector3>& getTexCoords() const { return m_TexCoord; }
    const std::vector<Vector3>& getNormals() const { return m_Normal; }
    
    const std::vector<int32>& getPositionIndices() const { return m_PositionIndices; }
    const std::vector<int32>& getTexCoordIndices() const { return m_TexCoordIndices; }
    const std::vector<int32>& getNormalIndices() const { return m_NormalIndices; }
    
    const std::vector<ObjMtlLoader::Material>& getMaterials() const { return m_Materials; }
    
    void parseMaterialFile(const Location& loc, const string& name);
    
    void normalizeIndices() { normalizeIndices(m_Position.size(), m_PositionIndices); normalizeIndices(m_TexCoord.size(), m_TexCoordIndices); normalizeIndices(m_Normal.size(), m_NormalIndices); }
    
    const std::vector<GroupHeader>& getGroups() const { return m_Groups; }
    
    bool parseFile(const string& filename);
    bool parseString(const char* str, size_t size, const string& filename);
private:
    void normalizeIndices(size_t elems, std::vector<int32>& vec)
    {
        for(auto& ind : vec)
        {
            ind = ind < 0 ? elems + ind : ind - 1;
        }
    }
};
}
}

#endif // _OBJ_LOADER_DRIVER_HH_
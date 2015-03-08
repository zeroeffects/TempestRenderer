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

#ifndef _TEMPEST_GL_SHADER_HH_
#define _TEMPEST_GL_SHADER_HH_

#ifdef _WIN32
    #include <windows.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"

#include <vector>
#include <algorithm>
#include <memory>

#include "tempest/utils/types.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/graphics/shader.hh"
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/shader/shader-common.hh"

namespace Tempest
{
/*! \brief Resource tables are intermediate objects created to make data assignment easier.
 * 
 *  The main idea behind resource table is to collect all values and create an object that
 *  is capable of assigning them in a single API call, if possible.
 */
class GLTexture;
class Matrix4;
struct Vector4;
struct Vector3;
struct Vector2;

template<class T> struct UniformValueBinding;
#define UNIFORM_VALUE_BINDING(type, value) \
    template<> struct UniformValueBinding<type> { \
        static const UniformValueType value_type = value; };

UNIFORM_VALUE_BINDING(GLTexture, UniformValueType::Texture);
UNIFORM_VALUE_BINDING(Matrix4, UniformValueType::Matrix4);
UNIFORM_VALUE_BINDING(Vector4, UniformValueType::Vector4);
UNIFORM_VALUE_BINDING(Vector3, UniformValueType::Vector3);
UNIFORM_VALUE_BINDING(Vector2, UniformValueType::Vector2);
UNIFORM_VALUE_BINDING(float, UniformValueType::Float);
UNIFORM_VALUE_BINDING(int32, UniformValueType::Integer);
UNIFORM_VALUE_BINDING(uint32, UniformValueType::UnsignedInteger);
UNIFORM_VALUE_BINDING(bool, UniformValueType::Boolean);

class GLResourceTable
{
    ResourceTableDescription* m_ResourceTable;

    BakedResourceTable   m_BakedResourceTable;
    size_t               m_ExtendedUnits;
public:
    GLResourceTable(ResourceTableDescription* desc, size_t extended)
        :   m_ResourceTable(desc),
            m_BakedResourceTable(desc->BufferSize + desc->ExtendablePart*extended),
            m_ExtendedUnits(extended) {}
    
    inline size_t getResourceCount() const { return m_ResourceTable->Uniforms.Count; }
    
    ResourceIndex getResourceIndex(const string& name);
    
    void setResource(ResourceIndex index, const GLTexture& tex);
    
    template<class T>
    void setResource(ResourceIndex index, const T& val)
    {
        TGE_ASSERT(index.ResourceTableIndex < m_ResourceTable->Uniforms.Count || m_ResourceTable->Uniforms.Count == std::numeric_limits<size_t>::max(), "Unknown index");
        if(index.ResourceTableIndex >= m_ResourceTable->Uniforms.Count)
            return;
        TGE_ASSERT(index.BaseOffset < m_ResourceTable->BufferSize + m_ExtendedUnits*m_ResourceTable->ExtendablePart, "Buffer overflow");
        TGE_ASSERT(m_BakedResourceTable, "The baked table is already extracted");
    #ifndef NDEBUG
        TGE_ASSERT(UniformValueBinding<T>::value_type == m_ResourceTable->Uniforms.Values[index.ResourceTableIndex].Type, "Mismatching uniform variable types.");
    #endif
        m_BakedResourceTable.setValue(index.BaseOffset, val);
    }
    
    template<class T>
    void setResource(const string& name, const T& val)
    {
        setResource(getResourceIndex(name), val);
    }
    
    DataDescription* getResourceDescription(ResourceIndex index)
    {
        TGE_ASSERT(index.ResourceTableIndex < m_ResourceTable->Uniforms.Count || m_ResourceTable->Uniforms.Count == std::numeric_limits<size_t>::max(), "Unknown index");
        if(index.ResourceTableIndex >= m_ResourceTable->Uniforms.Count)
            return nullptr;
        return m_ResourceTable->Uniforms.Values + index.ResourceTableIndex;
    }

    DataDescription* getResourceDescription(const string& name)
    {
        return getResourceDescription(getResourceIndex(name));
    }

    /*! \brief It gives you the table without the rest of the data.
     * 
     *  \remarks It is allocated in the usual fashion, so no special deallocation procedure is required. Just call delete.
     *           Also, it is completely throw-away. You might deallocate it at any time. The data is transferred to separate
     *           constant buffer.
     */
    BakedResourceTable* extractBakedTable() { return new BakedResourceTable(std::move(m_BakedResourceTable)); }
    
    BakedResourceTable* getBakedTable() { return &m_BakedResourceTable; }

    void resetBakedTable() { m_BakedResourceTable.reset(); }
};

class GLShaderProgram;
class GLRenderingBackend;
class GLInputLayout;
struct GLBufferTableEntry;

class GLShaderProgram
{
    GLuint                                       m_Program;
    GLInputLayout*                               m_InputLayout;
    std::unique_ptr<ResourceTableDescription*[]> m_ResourceTables;
    uint32                                       m_ResourceTableCount;

public:
    typedef GLResourceTable     ResourceTableType;
    
    explicit GLShaderProgram(GLuint shader_program, GLInputLayout* input_signature, ResourceTableDescription* resource_tables[], uint32 res_table_count);
     ~GLShaderProgram();
    
    GLShaderProgram(const GLShaderProgram&)=delete;
    GLShaderProgram& operator=(const GLShaderProgram&)=delete;
    GLShaderProgram(GLShaderProgram&&)=delete;
    GLShaderProgram& operator=(GLShaderProgram&&)=delete;
 
    void bind(GLBufferTableEntry* table_entry) const;
    
    const GLInputLayout* getInputLayout() const { return m_InputLayout; }

    GLResourceTable* createResourceTable(const string& name, size_t extended = 0);
};

}

#endif // _TEMPEST_GL_SHADER_HH_
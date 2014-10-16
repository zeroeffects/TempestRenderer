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

#include <GL/gl.h>

#include <vector>
#include <algorithm>
#include <memory>

#include "tempest/utils/types.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
/*! \brief Resource tables are intermediate objects created to make data assignment easier.
 * 
 *  The main idea behind resource table is to collect all values and create an object that
 *  is capable of assigning them in a single API call, if possible.
 */
class GLTexture;
class Matrix4;
class Vector4;
class Vector3;
class Vector2;

template<class T> struct UniformValueBinding;
#define UNIFORM_VALUE_BINDING(type, value) \
    template<> struct UniformValueBinding<type> { \
        static constexpr UniformValueType value_type = value; };

UNIFORM_VALUE_BINDING(GLTexture, UniformValueType::Texture);
UNIFORM_VALUE_BINDING(Matrix4, UniformValueType::Matrix4);
UNIFORM_VALUE_BINDING(Vector4, UniformValueType::Vector4);
UNIFORM_VALUE_BINDING(Vector3, UniformValueType::Vector3);
UNIFORM_VALUE_BINDING(Vector2, UniformValueType::Vector2);
UNIFORM_VALUE_BINDING(float, UniformValueType::Float);
UNIFORM_VALUE_BINDING(int32, UniformValueType::Integer);
UNIFORM_VALUE_BINDING(uint32, UniformValueType::UnsignedInteger);
UNIFORM_VALUE_BINDING(bool, UniformValueType::Boolean);

struct DataDescription
{
    string           Name;
    UniformValueType Type;
    uint16           ElementSize;
    uint16           ElementCount;
    uint32           Offset;
};

class GLBakedResourceTable
{
    char*                   m_Table;
    size_t                  m_Size;
public:
    GLBakedResourceTable(size_t size)
        :   m_Table(new char[size]),
            m_Size(size) {}
    ~GLBakedResourceTable() { delete m_Table; }
    
    GLBakedResourceTable(GLBakedResourceTable&& table)
    {
        m_Table = table.m_Table;
        m_Size = table.m_Size;
        table.m_Table = nullptr;
        table.m_Size = 0;
    }
    
    GLBakedResourceTable& operator=(GLBakedResourceTable&& table)
    {
        m_Table = table.m_Table;
        m_Size = table.m_Size;
        table.m_Table = nullptr;
        table.m_Size = 0;
    }
    
    template<class T>
    void setValue(size_t offset, const T& val)
    {
        *reinterpret_cast<T*>(m_Table + offset) = val;
    }
    
    operator bool() const { return m_Table != nullptr; }
    
    const char* get() const { return m_Table; }
    
    size_t getSize() const { return m_Size; }
};

struct ResourceTableDescription
{
    string                       Name;
    uint32                       BindPoint;
    uint32                       BufferSize;
    uint32                       ExtendablePart;
    PACKED_DATA(DataDescription) Uniforms;
    
    ResourceTableDescription(const ResourceTableDescription&)=delete;
    ResourceTableDescription& operator=(const ResourceTableDescription&)=delete;
    
private:
    ResourceTableDescription(size_t count, size_t extendable_part, string name, uint32 bind_point)
        :   ExtendablePart(extendable_part),
            Name(name),
            BindPoint(bind_point),
            Uniforms(count) {}
     ~ResourceTableDescription()=default;
};

struct ResourceIndex
{
    uint32 ResourceTableIndex = 0;
    uint32 BaseOffset = 0;
};

class GLResourceTable
{
    ResourceTableDescription* m_ResourceTable;

    GLBakedResourceTable m_BakedResourceTable;
    size_t               m_ExtendedUnits;
public:
    GLResourceTable(ResourceTableDescription* desc, size_t extended)
        :   m_ResourceTable(desc),
            m_BakedResourceTable(desc->BufferSize + desc->ExtendablePart*extended),
            m_ExtendedUnits(extended) {}
    
    typedef GLBakedResourceTable BakedResourceTableType;
    
    inline size_t getResourceCount() const { return m_ResourceTable->Uniforms.Count; }
    
    ResourceIndex getResourceIndex(const string& name);
    
    inline void setSubroutine(ResourceIndex index, ResourceIndex subroutine_func)
    {
        TGE_ASSERT(index.ResourceTableIndex < m_ResourceTable->Uniforms.Count || m_ResourceTable->Uniforms.Count == std::numeric_limits<size_t>::max(), "Unknown index");
        if(index.ResourceTableIndex >= m_ResourceTable->Uniforms.Count)
            return;
        TGE_ASSERT(index.BaseOffset < m_ResourceTable->BufferSize + m_ExtendedUnits*m_ResourceTable->ExtendablePart, "Buffer overflow");
        TGE_ASSERT(m_BakedResourceTable, "The baked table is already extracted");
        m_BakedResourceTable.setValue(index.BaseOffset, subroutine_func.ResourceTableIndex);
    }
    
    inline void setSubroutine(const string& name, const string& func)
    {
        setSubroutine(getResourceIndex(name), getResourceIndex(func));
    }
    
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
    
    /*! \brief It gives you the table without the rest of the data.
     * 
     *  \remarks It is allocated in the usual fashion, so no special deallocation procedure is required. Just call delete.
     *           Also, it is completely throw-away. You might deallocate it at any time. The data is transferred to separate
     *           constant buffer.
     */
    GLBakedResourceTable* extractBakedTable() { return new GLBakedResourceTable(std::move(m_BakedResourceTable)); }
    
    GLBakedResourceTable* getBakedTable() { return &m_BakedResourceTable; }
};

class GLShaderProgram;
class GLInputLayout;
class GLRenderingBackend;
struct VertexAttributeDescription;

class GLLinkedShaderProgram
{
    friend GLShaderProgram;
    GLuint                m_Program;
    GLBakedResourceTable* m_Baked;
public:
    GLLinkedShaderProgram(GLuint prog, GLBakedResourceTable* table)
        :   m_Program(prog),
            m_Baked(table) {}
    
    void bind();
};

typedef std::vector<std::unique_ptr<GLLinkedShaderProgram>> DynamicLinkageCache;

class GLShaderProgram
{
    GLuint                                       m_Program;
    DynamicLinkageCache                          m_Programs;
    std::unique_ptr<ResourceTableDescription*[]> m_ResourceTables;
    GLint                                        m_ResourceTableCount;

public:
    typedef GLInputLayout InputLayoutType;
    typedef GLResourceTable ResourceTableType;
    
    explicit GLShaderProgram(GLuint shader_program, ResourceTableDescription* resource_tables[], size_t res_table_count);
     ~GLShaderProgram();
    
    GLShaderProgram(const GLShaderProgram&)=delete;
    GLShaderProgram& operator=(const GLShaderProgram&)=delete;
    GLShaderProgram(GLShaderProgram&&)=delete;
    GLShaderProgram& operator=(GLShaderProgram&&)=delete;
 
    //! \remarks Don't deallocate
    GLLinkedShaderProgram* getUniqueLinkage(GLBakedResourceTable* _table);
    
    GLInputLayout* createInputLayout(GLRenderingBackend* backend, const VertexAttributeDescription* arr, size_t count);
    void destroyRenderResource(GLRenderingBackend* backend, GLInputLayout* input_layout);
    
    GLResourceTable* createResourceTable(const string& name, size_t extended = 0);
    void destroyRenderResource(GLResourceTable* buffer);
};

}

#endif // _TEMPEST_GL_SHADER_HH_
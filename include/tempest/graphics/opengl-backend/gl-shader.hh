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

#include <algorithm>
#include <memory>

#include "tempest/utils/types.hh"
#include "tempest/utils/assert.hh"
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
    size_t           ElementCount;
    size_t           Offset;
};

class GLBakedResourceTable
{
    std::unique_ptr<char[]> m_Table;
    size_t                  m_Size;
public:
    template<class T>
    void setValue(size_t offset, const T& val)
    {
        *reinterpret_cast<T*>(m_Table[offset]) = val;
    }
    
    operator bool() const { return m_Table != nullptr; }
    
    size_t getSize() const { return m_Size; }
};

struct ResourceTableDescription
{
    string          Name;
    uint32          BindPoint;
    uint32          Count;
    uint32          ResourceTableSize;
    DataDescription UniformValue[];
};

class GLResourceTable
{
    ResourceTableDescription* m_ResourceTable;

    GLBakedResourceTable m_BakedResourceTable;
public:
    typedef GLBakedResourceTable BakedResourceTableType;
    
    inline size_t getResourceIndex(const string& name)
    {
        auto iter = std::find_if(m_ResourceTable->UniformValue, m_ResourceTable->UniformValue + m_ResourceTable->Count, [&name](const DataDescription& data) { return data.Name == name; });
        TGE_ASSERT(iter != m_ResourceTable->UniformValue + m_ResourceTable->Count, "Unknown variable");
        return iter != m_ResourceTable->UniformValue + m_ResourceTable->Count ? iter-m_ResourceTable->UniformValue : std::numeric_limits<size_t>::max(); // So we don't crash on reloads and in general.
    }
    
    void setResource(size_t index, const GLTexture& tex);
    
    template<class T>
    void setResource(size_t index, const T& val)
    {
        TGE_ASSERT(index < m_ResourceTable->Count || m_ResourceTable->Count == std::numeric_limits<size_t>::max(), "Unknown index");
        if(index >= m_ResourceTable->Count)
            return;
        TGE_ASSERT(m_BakedResourceTable, "The baked table is already extracted");
    #ifndef NDEBUG
        TGE_ASSERT(UniformValueBinding<T>::value_type == m_ResourceTable->UniformValue[index].Type, "Mismatching uniform variable types.");
    #endif
        m_BakedResourceTable.setValue(m_ResourceTable->UniformValue[index].Offset, val);
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

class GLInputLayout;
class GLRenderingBackend;
struct VertexAttributeDescription;



class GLShaderProgram
{
    GLuint                                       m_Id;
    std::unique_ptr<ResourceTableDescription*[]> m_ResourceTables;
    GLint                                        m_ResourceTableCount;

public:
    typedef GLInputLayout InputLayoutType;
    typedef GLResourceTable ResourceTableType;
    
    explicit GLShaderProgram(GLuint id);
     ~GLShaderProgram();
    
    GLShaderProgram(const GLShaderProgram&)=delete;
    GLShaderProgram& operator=(const GLShaderProgram&)=delete;
    GLShaderProgram(GLShaderProgram&&)=delete;
    GLShaderProgram& operator=(GLShaderProgram&&)=delete;
    
    void bind();
    void setupInputLayout(GLInputLayout* layout);
    
    GLInputLayout* createInputLayout(GLRenderingBackend* backend, const VertexAttributeDescription* arr, size_t count);
    void destroyRenderResource(GLRenderingBackend* backend, GLInputLayout* input_layout);
    
    GLResourceTable* createResourceTable(const string& name);
    void destroyRenderResource(GLResourceTable* buffer);
};

}

#endif // _TEMPEST_GL_SHADER_HH_
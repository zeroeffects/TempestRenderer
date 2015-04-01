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

#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/assert.hh"

namespace Tempest
{
void GLResourceTable::setResource(ResourceIndex index, const GLTexture& tex)
{
    TGE_ASSERT(index.ResourceTableIndex < m_ResourceTable->Uniforms.Count || m_ResourceTable->Uniforms.Count == std::numeric_limits<size_t>::max(), "Unknown index");
    if(index.ResourceTableIndex >= m_ResourceTable->Uniforms.Count)
        return;
    TGE_ASSERT(index.BaseOffset < m_ResourceTable->BufferSize + m_ExtendedUnits*m_ResourceTable->ExtendablePart, "Buffer overflow");
    TGE_ASSERT(m_BakedResourceTable, "The baked table is already extracted");
    #ifndef NDEBUG
        TGE_ASSERT(UniformValueType::Texture == m_ResourceTable->Uniforms.Values[index.ResourceTableIndex].Type, "Mismatching uniform variable types.");
    #endif
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
    {
        m_BakedResourceTable.setValue(index.BaseOffset, tex.getGPUHandle());
    }
    else
#endif
    {
        m_BakedResourceTable.setValue(index.BaseOffset, tex.getCPUHandle());
    }
}
    
ResourceIndex GLResourceTable::getResourceIndex(const string& name)
{
    ResourceIndex res_idx;
    string sliced_string;
    sliced_string.reserve(name.size());
    int array_index = 0, array_index_base = 1;
    for(size_t idx = 0; idx < name.size(); ++idx)
    {
        char c = name[idx];
        // Array indexing gets chopped of because it hinders search.
        if(c == '[')
        {
            TGE_ASSERT(array_index == 0, "Double arrays are not supported");
            for(;;)
            {
                ++idx;
                TGE_ASSERT(idx < name.size(), "Array ended prematurely");
                c = name[idx];
                if(isdigit(c))
                {
                    array_index += (c - '0')*array_index_base;
                    array_index_base *= 10;
                }
                else if(c == ']')
                {
                    break;
                }
            }
        }
        else
        {
            if(c == '.')
            {
                auto iter = std::find_if(m_ResourceTable->Uniforms.Values, m_ResourceTable->Uniforms.Values + m_ResourceTable->Uniforms.Count, [&sliced_string](const DataDescription& data) { return data.Name == sliced_string; });
                TGE_ASSERT(iter != m_ResourceTable->Uniforms.Values + m_ResourceTable->Uniforms.Count, "Unknown variable");
                res_idx.BaseOffset += iter->Offset + iter->ElementSize*array_index;
                array_index = 0, array_index_base = 1;
            }
            else
            {
                TGE_ASSERT(array_index_base == 1, "Array indexing should be followed by member access operator");
            }
            sliced_string.push_back(c);
        }
    }
    auto iter = std::find_if(m_ResourceTable->Uniforms.Values, m_ResourceTable->Uniforms.Values + m_ResourceTable->Uniforms.Count, [&sliced_string](const DataDescription& data) { return data.Name == sliced_string; });
    TGE_ASSERT(iter != m_ResourceTable->Uniforms.Values + m_ResourceTable->Uniforms.Count, "Unknown variable");
    res_idx.BaseOffset += iter->Offset + iter->ElementSize*array_index;
    res_idx.ResourceTableIndex = iter != m_ResourceTable->Uniforms.Values + m_ResourceTable->Uniforms.Count ? static_cast<uint32>(iter - m_ResourceTable->Uniforms.Values) : std::numeric_limits<uint32>::max(); // So we don't crash on reloads and in general.
    return res_idx;
}
    
GLShaderProgram::GLShaderProgram(GLuint prog, GLInputLayout* input_layout, ResourceTableDescription* res_tables[], uint32 res_table_count)
    :   m_Program(prog),
        m_InputLayout(input_layout),
        m_ResourceTableCount(res_table_count),
        m_ResourceTables(res_tables) {}

GLShaderProgram::~GLShaderProgram()
{
    DestroyPackedData(m_InputLayout);
    for(uint32 i = 0; i < m_ResourceTableCount; ++i)
    {
        DestroyPackedData(m_ResourceTables[i]);
    }
    glDeleteProgram(m_Program);
}

GLResourceTable* GLShaderProgram::createResourceTable(const string& name, size_t extended)
{
    auto iter = std::find_if(m_ResourceTables.get(), m_ResourceTables.get() + m_ResourceTableCount,
                             [&name](const ResourceTableDescription* desc) { return desc->Name == name; });
    return iter != m_ResourceTables.get() + m_ResourceTableCount ? new GLResourceTable(*iter, extended) : nullptr;
}

void GLShaderProgram::bind(GLBufferTableEntry* table_entry) const
{
    glUseProgram(m_Program);
    if(m_InputLayout)
    {
        m_InputLayout->bind(table_entry);
    }
}
}
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
#include "tempest/utils/logging.hh"
#include "tempest/utils/assert.hh"

namespace Tempest
{
void GLResourceTable::setResource(size_t index, const GLTexture& tex)
{
    TGE_ASSERT(false, "Stub");
}

GLShaderProgram::GLShaderProgram(GLuint id, ResourceTableDescription* res_tables[])
    :   m_Id(id),
        m_ResourceTables(res_tables)
{
}

GLShaderProgram::~GLShaderProgram()
{
    for(size_t i = 0; i < m_ResourceTableCount; ++i)
    {
        auto& res_table = m_ResourceTables[i];
        for(size_t j = 0; j < res_table->Count; ++j)
        {
            res_table->UniformValue[j].~DataDescription();
        }
        m_ResourceTables[i]->~ResourceTableDescription();
        free(m_ResourceTables[i]);
    }
    glDeleteProgram(m_Id);
}

GLResourceTable* GLShaderProgram::createResourceTable(const string& name)
{
    auto iter = std::find_if(m_ResourceTables.get(), m_ResourceTables.get() + m_ResourceTableCount, [name](const ResourceTableDescription* desc) { return desc->Name == name; });
    return iter != m_ResourceTables.get() + m_ResourceTableCount ? new GLResourceTable(*iter) : nullptr;
}

void GLShaderProgram::destroyRenderResource(GLResourceTable* buffer)
{
    delete buffer;
}

void GLShaderProgram::bind()
{
    glUseProgram(m_Id);
}

void GLShaderProgram::setupInputLayout(GLInputLayout* layout)
{
    if(!layout)
        return;
    for(size_t i = 0, iend = layout->getAttributeCount(); i < iend; ++i)
    {
        auto* vert_attr = layout->getAttribute(i);
        glVertexAttribFormat(i, vert_attr->Size, vert_attr->Type, vert_attr->Normalized, vert_attr->Offset);
        glBindVertexBuffer(i, 0, 0, vert_attr->Stride);
        glVertexAttribBinding(i, vert_attr->Binding);
        glEnableVertexAttribArrayARB(i);
    }
    CheckOpenGL();
}

GLInputLayout* GLShaderProgram::createInputLayout(GLRenderingBackend* backend, const VertexAttributeDescription* arr, size_t count)
{
    return new (malloc(sizeof(size_t) + count*sizeof(GLVertexAttributeDescription))) GLInputLayout(arr, count);
}

void GLShaderProgram::destroyRenderResource(GLRenderingBackend* backend, GLInputLayout* input_layout)
{
    input_layout->~GLInputLayout();
    free(input_layout);
}
}
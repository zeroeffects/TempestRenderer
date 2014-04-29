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

GLShaderProgram::GLShaderProgram(GLuint id)
    :   m_Id(id)
{
    glGetProgramiv(m_Id, GL_ACTIVE_UNIFORM_BLOCKS, &m_ResourceTableCount);
    GLint uniform_block_max_len;
    glGetProgramiv(m_Id, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &uniform_block_max_len);
    GLint uniform_max_len;
    glGetProgramiv(m_Id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &uniform_max_len);
    uniform_max_len = std::max(uniform_max_len, uniform_block_max_len) + 1;

    m_ResourceTables = std::unique_ptr<ResourceTableDescription*[]>(new ResourceTableDescription*[m_ResourceTableCount]);
    
    std::unique_ptr<GLchar[]> name(new GLchar[uniform_max_len]);

    std::vector<GLint> indices;
    std::vector<GLint> types;
    std::vector<GLint> offsets;
    std::vector<GLint> sizes;
    
    GLint   size;
    GLenum  type;
    
  
    for(GLint block_idx = 0; block_idx < m_ResourceTableCount; ++block_idx)
    {
        auto* uniform_block = m_ResourceTables[block_idx];

        GLint count;
        glGetActiveUniformBlockiv(m_Id, block_idx, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &count);
        
        m_ResourceTables[block_idx] = new (malloc(sizeof(ResourceTableDescription) + count*sizeof(DataDescription))) ResourceTableDescription;
        uniform_block->Count = count;
        GLint bind_point;
        glGetActiveUniformBlockiv(m_Id, block_idx, GL_UNIFORM_BLOCK_BINDING, &bind_point);
        uniform_block->BindPoint = bind_point;
        
        if(count > indices.size())
        {
            indices.resize(count);
            types.resize(count);
        }
        
        GLsizei size;
        glGetActiveUniformBlockName(m_Id, block_idx, uniform_max_len, &size, name.get());
        uniform_block->Name = name.get();
        
        glGetActiveUniformBlockiv(m_Id, block_idx, GL_UNIFORM_BLOCK_DATA_SIZE, &count);
        uniform_block->ResourceTableSize = count;
        glGetActiveUniformBlockiv(m_Id, block_idx, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, &indices.front()); 

        glGetActiveUniformsiv(m_Id, count, (GLuint*)&indices.front(), GL_UNIFORM_TYPE, &types.front());
        glGetActiveUniformsiv(m_Id, count, (GLuint*)&indices.front(), GL_UNIFORM_OFFSET, &offsets.front());
        glGetActiveUniformsiv(m_Id, count, (GLuint*)&indices.front(), GL_UNIFORM_SIZE, &sizes.front());
        
        for(size_t uniform_idx = 0; uniform_idx < count; ++uniform_idx)
        {
            auto& desc = uniform_block->UniformValue[uniform_idx];
            switch(type)
            {
            case GL_FLOAT: desc.Type = UniformValueType::Float; break;
            case GL_FLOAT_VEC2: desc.Type = UniformValueType::Vector2; break;
            case GL_FLOAT_VEC3: desc.Type = UniformValueType::Vector3; break;
            case GL_FLOAT_VEC4: desc.Type = UniformValueType::Vector4; break;
            case GL_INT: desc.Type = UniformValueType::Integer; break;
            case GL_INT_VEC2: desc.Type = UniformValueType::IntegerVector2; break;
            case GL_INT_VEC3: desc.Type = UniformValueType::IntegerVector3; break;
            case GL_INT_VEC4: desc.Type = UniformValueType::IntegerVector4; break;
            case GL_UNSIGNED_INT: desc.Type = UniformValueType::UnsignedInteger; break;
            case GL_UNSIGNED_INT_VEC2: desc.Type = UniformValueType::UnsignedIntegerVector2; break;
            case GL_UNSIGNED_INT_VEC3: desc.Type = UniformValueType::UnsignedIntegerVector3; break;
            case GL_UNSIGNED_INT_VEC4: desc.Type = UniformValueType::UnsignedIntegerVector4; break;
            case GL_BOOL: desc.Type = UniformValueType::Boolean; break;
            case GL_BOOL_VEC2: desc.Type = UniformValueType::BooleanVector2; break;
            case GL_BOOL_VEC3: desc.Type = UniformValueType::BooleanVector3; break;
            case GL_BOOL_VEC4: desc.Type = UniformValueType::BooleanVector4; break;
            case GL_FLOAT_MAT2: desc.Type = UniformValueType::Matrix2; break;
            case GL_FLOAT_MAT3: desc.Type = UniformValueType::Matrix3; break;
            case GL_FLOAT_MAT4: desc.Type = UniformValueType::Matrix4; break;
            case GL_FLOAT_MAT2x3: desc.Type = UniformValueType::Matrix2x3; break;
            case GL_FLOAT_MAT2x4: desc.Type = UniformValueType::Matrix2x4; break;
            case GL_FLOAT_MAT3x2: desc.Type = UniformValueType::Matrix3x2; break;
            case GL_FLOAT_MAT3x4: desc.Type = UniformValueType::Matrix3x4; break;
            case GL_FLOAT_MAT4x2: desc.Type = UniformValueType::Matrix4x2; break;
            case GL_FLOAT_MAT4x3: desc.Type = UniformValueType::Matrix4x3; break;
            default:
                Log(LogLevel::Error, "Unknown data type \"", type, "\": within the following uniform block: ", name.get());
            }
            desc.Offset = offsets[uniform_idx];
            desc.ElementCount = sizes[uniform_idx];
            glGetActiveUniformName(m_Id, indices[uniform_idx], uniform_max_len, &size, name.get());
            desc.Name = name.get();
        }
    }
}

GLShaderProgram::~GLShaderProgram()
{
    for(size_t i = 0; i < m_ResourceTableCount; ++i)
        free(m_ResourceTables[i]);
    glDeleteProgram(m_Id);
}

GLResourceTable* GLShaderProgram::createResourceTable(const string& name)
{
    TGE_ASSERT(false, "Stub");
}

void GLShaderProgram::destroyRenderResource(GLResourceTable* buffer)
{
    TGE_ASSERT(false, "Stub");
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
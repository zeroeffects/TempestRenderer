
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

#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/shader/shader-common.hh"
#include "tempest/utils/assert.hh"

namespace Tempest
{
static void TranslateDataFormat(DataFormat vtype, GLsizei* elements, GLType* type, GLboolean* normalized)
{
    switch(vtype)
    {
    default: TGE_ASSERT(false, "Unknown variable *type"); break;
    case DataFormat::R32F: *elements = 1, *type = GLType::GL_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::RG32F: *elements = 2, *type = GLType::GL_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::RGB32F: *elements = 3, *type = GLType::GL_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::RGBA32F: *elements = 4, *type = GLType::GL_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::R16F: *elements = 1, *type = GLType::GL_HALF_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::RG16F: *elements = 2, *type = GLType::GL_HALF_FLOAT, *normalized = GL_FALSE; break;
//  case DataFormat::RGB16F: *elements = 3, *type = GLType::GL_HALF_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::RGBA16F: *elements = 4, *type = GLType::GL_HALF_FLOAT, *normalized = GL_FALSE; break;
    case DataFormat::R32: *elements = 1, *type = GLType::GL_INT, *normalized = GL_FALSE; break;
    case DataFormat::RG32: *elements = 2, *type = GLType::GL_INT, *normalized = GL_FALSE; break;
    case DataFormat::RGB32: *elements = 3, *type = GLType::GL_INT, *normalized = GL_FALSE; break;
    case DataFormat::RGBA32: *elements = 4, *type = GLType::GL_INT, *normalized = GL_FALSE; break;
    case DataFormat::R16: *elements = 1, *type = GLType::GL_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::RG16: *elements = 2, *type = GLType::GL_SHORT, *normalized = GL_FALSE; break;
//  case DataFormat::RGB16: *elements = 3, *type = GLType::GL_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::RGBA16: *elements = 4, *type = GLType::GL_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::R8: *elements = 1, *type = GLType::GL_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::RG8: *elements = 2, *type = GLType::GL_BYTE, *normalized = GL_FALSE; break;
//  case DataFormat::RGB8: *elements = 3, *type = GLType::GL_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::RGBA8: *elements = 4, *type = GLType::GL_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::uR32: *elements = 1, *type = GLType::GL_UNSIGNED_INT, *normalized = GL_FALSE; break;
    case DataFormat::uRG32: *elements = 2, *type = GLType::GL_UNSIGNED_INT, *normalized = GL_FALSE; break;
    case DataFormat::uRGB32: *elements = 3, *type = GLType::GL_UNSIGNED_INT, *normalized = GL_FALSE; break;
    case DataFormat::uRGBA32: *elements = 4, *type = GLType::GL_UNSIGNED_INT, *normalized = GL_FALSE; break;
    case DataFormat::uR16: *elements = 1, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::uRG16: *elements = 2, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_FALSE; break;
//  case DataFormat::uRGB16: *elements = 3, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::uRGBA16: *elements = 4, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_FALSE; break;
    case DataFormat::uR8: *elements = 1, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::uRG8: *elements = 2, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_FALSE; break;
//  case DataFormat::uRGB8: *elements = 3, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::uRGBA8: *elements = 4, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_FALSE; break;
    case DataFormat::R16SNorm: *elements = 1, *type = GLType::GL_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::RG16SNorm: *elements = 2, *type = GLType::GL_SHORT, *normalized = GL_TRUE; break;
//  case DataFormat::RGB16SNorm: *elements = 3, *type = GLType::GL_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::RGBA16SNorm: *elements = 4, *type = GLType::GL_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::R8SNorm: *elements = 1, *type = GLType::GL_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::RG8SNorm: *elements = 2, *type = GLType::GL_BYTE, *normalized = GL_TRUE; break;
//  case DataFormat::RGB8SNorm: *elements = 3, *type = GLType::GL_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::RGBA8SNorm: *elements = 4, *type = GLType::GL_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::R16UNorm: *elements = 1, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::RG16UNorm: *elements = 2, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_TRUE; break;
//  case DataFormat::RGB16UNorm: *elements = 3, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::RGBA16UNorm: *elements = 4, *type = GLType::GL_UNSIGNED_SHORT, *normalized = GL_TRUE; break;
    case DataFormat::R8UNorm: *elements = 1, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::RG8UNorm: *elements = 2, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_TRUE; break;
//  case DataFormat::RGB8UNorm: *elements = 3, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::RGBA8UNorm: *elements = 4, *type = GLType::GL_UNSIGNED_BYTE, *normalized = GL_TRUE; break;
    case DataFormat::D16: *elements = 1, *type = GLType::GL_DEPTH_COMPONENT16, *normalized = GL_TRUE; break;
    case DataFormat::D24S8: *elements = 2, *type = GLType::GL_UNSIGNED_INT_24_8, *normalized = GL_TRUE; break; // GL_DEPTH_STENCIL
    case DataFormat::D32: *elements = 1, *type = GLType::GL_DEPTH_COMPONENT32, *normalized = GL_TRUE; break;
    case DataFormat::R10G10B10A2: *elements = 4, *type = GLType::GL_INT_2_10_10_10_REV, *normalized = GL_FALSE; break;
    case DataFormat::uR10G10B10A2: *elements = 4, *type = GLType::GL_UNSIGNED_INT_2_10_10_10_REV, *normalized = GL_FALSE; break;
    }
}

GLInputLayout::GLInputLayout(uint32 count, const Shader::VertexAttributeDescription* arr)
    :   m_Attributes(count)
{
    for(size_t i = 0; i < count; ++i)
    {
        auto& vert_attr = m_Attributes.Values[i];
        GLsizei elements;
        GLType type;
        GLboolean normalized;
        TranslateDataFormat(arr[i].Format, &elements, &type, &normalized);
        vert_attr.Binding = arr[i].BufferId;
        vert_attr.Normalized = normalized;
        vert_attr.Offset = arr[i].Offset;
        vert_attr.Size = elements;
        vert_attr.Type = type;
    }
}

void GLInputLayout::bind(GLBufferTableEntry* buffer_table) const
{
    for(GLuint i = 0, iend = m_Attributes.Count; i < iend; ++i)
    {
        auto& vert_attr = m_Attributes.Values[i];
        if(buffer_table)
        {
            // GL 2.0 style - should work everywhere.
            glVertexAttribPointer(i, vert_attr.Size, vert_attr.Type, vert_attr.Normalized,
                                 buffer_table[i].Stride,
                                 reinterpret_cast<GLvoid*>(reinterpret_cast<GLchar*>(nullptr) + vert_attr.Offset + buffer_table[i].Offset));
        }
        else
        {
            glVertexAttribFormat(i, vert_attr.Size, vert_attr.Type, vert_attr.Normalized, vert_attr.Offset);
            glVertexAttribBinding(i, vert_attr.Binding);
        }
        glEnableVertexAttribArray(i);
    }
    CheckOpenGL();
}
}
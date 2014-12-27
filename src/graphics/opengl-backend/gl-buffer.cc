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

#include "tempest/graphics/opengl-backend/gl-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/utils/assert.hh"

namespace Tempest
{
GLUsageMode TranslateUsage(size_t usage)
{
    switch(usage & RESOURCE_USAGE_MASK)
    {
    default: TGE_ASSERT(false, "Unknown usage");
    case RESOURCE_STATIC_DRAW: return GLUsageMode::GL_STATIC_DRAW;
    case RESOURCE_STATIC_READ: return GLUsageMode::GL_STATIC_READ;
    case RESOURCE_STATIC_COPY: return GLUsageMode::GL_STATIC_COPY;
    case RESOURCE_STREAM_DRAW: return GLUsageMode::GL_STREAM_DRAW;
    case RESOURCE_STREAM_READ: return GLUsageMode::GL_STREAM_READ;
    case RESOURCE_STREAM_COPY: return GLUsageMode::GL_STREAM_COPY;
    case RESOURCE_DYNAMIC_DRAW: return GLUsageMode::GL_DYNAMIC_DRAW;
    case RESOURCE_DYNAMIC_READ: return GLUsageMode::GL_DYNAMIC_READ;
    case RESOURCE_DYNAMIC_COPY: return GLUsageMode::GL_DYNAMIC_COPY;
    }
}

static GLBufferTarget TranslateVBType(VBType vb_type)
{
    switch(vb_type)
    {
    default: TGE_ASSERT(false, "Unknown video buffer bind type"); // fall-through
    case VBType::VertexBuffer: return GLBufferTarget::GL_ARRAY_BUFFER;
    case VBType::IndexBuffer: return GLBufferTarget::GL_ELEMENT_ARRAY_BUFFER;
    }
}

GLBuffer::GLBuffer(size_t size, VBType vb_type, size_t usage, const void* data)
    :   m_Size(size)
{
    auto gl_vbt = TranslateVBType(vb_type);
    glGenBuffers(1, &m_Buffer);
    glBindBuffer(gl_vbt, m_Buffer);
    glBufferData(gl_vbt, size, data, TranslateUsage(usage));
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_MDI_BINDLESS))
    {
        glGetBufferParameterui64vNV(gl_vbt, GLBufferParameterNV::GL_BUFFER_GPU_ADDRESS_NV, &m_GPUAddress);
        glMakeBufferResidentNV(gl_vbt, GLAccessMode::GL_READ_ONLY);
    }
    CheckOpenGL();
}

GLBuffer::~GLBuffer()
{
    glDeleteBuffers(1, &m_Buffer);
}

void GLBuffer::bindVertexBuffer(GLuint bind_slot, GLintptr offset, GLsizei stride)
{
    glBindBuffer(GLBufferTarget::GL_ARRAY_BUFFER, m_Buffer);
    //glBindVertexBuffer(bind_slot, m_Buffer, offset, stride);
}

void GLBuffer::bindIndexBuffer()
{
    glBindBuffer(GLBufferTarget::GL_ELEMENT_ARRAY_BUFFER, m_Buffer);
}
}
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

#include "tempest/graphics/opengl-backend/gl-storage.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/texture.hh"

#include <cstring>

namespace Tempest
{
GLBufferTarget TranslateStorage(StorageMode storage_type)
{
    switch(storage_type)
    {
    default: TGE_ASSERT(false, "Unknown storage type");
    case StorageMode::BufferWrite: return GLBufferTarget::GL_COPY_WRITE_BUFFER;
    case StorageMode::BufferRead: return GLBufferTarget::GL_COPY_READ_BUFFER;
    case StorageMode::PixelPack: return GLBufferTarget::GL_PIXEL_PACK_BUFFER;
    case StorageMode::PixelUnpack: return GLBufferTarget::GL_PIXEL_UNPACK_BUFFER;
    }
}

GLbitfield TranslateAccessBit(StorageMode storage_type)
{
    switch(storage_type)
    {
    default: TGE_ASSERT(false, "Unknown storage type");
    case StorageMode::BufferWrite:
    case StorageMode::PixelPack: return GL_MAP_READ_BIT;
    case StorageMode::BufferRead:
    case StorageMode::PixelUnpack: return GL_MAP_WRITE_BIT;
    }
}

GLUsageMode TranslateAccess(StorageMode storage_type)
{
    switch(storage_type)
    {
    default: TGE_ASSERT(false, "Unknown storage type");
    case StorageMode::BufferWrite:
    case StorageMode::PixelPack: return GLUsageMode::GL_STREAM_READ;
    case StorageMode::BufferRead:
    case StorageMode::PixelUnpack: return GLUsageMode::GL_STREAM_DRAW;
    }
}

GLStorage::GLStorage(StorageMode storage_type, uint32 size)
    :   m_Size(size),
        m_Target(TranslateStorage(storage_type)),
        m_Access(TranslateAccessBit(storage_type))
{
    glGenBuffers(1, &m_Storage);
    glBindBuffer(m_Target, m_Storage);
    glBufferData(m_Target, size, 0, TranslateAccess(storage_type));
    CheckOpenGL();
    glBindBuffer(m_Target, 0);
}

GLStorage::~GLStorage()
{
    glDeleteBuffers(1, &m_Storage);
}

void GLStorage::storeLinearBuffer(uint32 offset, uint32 size, const void* data)
{
    glBindBuffer(m_Target, m_Storage);
    auto* data_ptr = static_cast<uint8*>(glMapBufferRange(m_Target, offset, size,
                                                          m_Access));
        memcpy(data_ptr, data, size);
    glUnmapBuffer(m_Target);
    glBindBuffer(m_Target, 0);
}

void GLStorage::storeTexture(uint32 offset, const TextureDescription& tex_desc, const void* data)
{
    GLsizeiptr size = tex_desc.Height*tex_desc.Width*tex_desc.Depth*DataFormatElementSize(tex_desc.Format);
    glBindBuffer(m_Target, m_Storage);
    auto* data_ptr = static_cast<uint8*>(glMapBufferRange(m_Target, offset, size,
        m_Access));
        memcpy(data_ptr, data, size);
    glUnmapBuffer(m_Target);
    glBindBuffer(m_Target, 0);
}

void GLStorage::extractLinearBuffer(uint32 offset, uint32 size, void* data)
{
    glBindBuffer(m_Target, m_Storage);
    auto* data_ptr = static_cast<uint8*>(glMapBufferRange(m_Target, offset, size,
        m_Access));
        memcpy(data, data_ptr, size);
    glUnmapBuffer(m_Target);
    glBindBuffer(m_Target, 0);
}

void GLStorage::extractTexture(uint32 offset, const TextureDescription& tex_desc, void* data)
{
    glBindBuffer(m_Target, m_Storage);
    GLsizeiptr size = tex_desc.Height*tex_desc.Width*tex_desc.Depth*DataFormatElementSize(tex_desc.Format);
    auto* data_ptr = static_cast<uint8*>(glMapBufferRange(m_Target, offset, size,
        m_Access));
        memcpy(data, data_ptr, size);
    glUnmapBuffer(m_Target);
    glBindBuffer(m_Target, 0);
}
}

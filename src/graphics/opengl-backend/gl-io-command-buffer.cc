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

#include "tempest/graphics/opengl-backend/gl-io-command-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-storage.hh"
#include "tempest/graphics/opengl-backend/gl-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"

namespace Tempest
{
GLIOCommandBuffer::GLIOCommandBuffer(const IOCommandBufferDescription& cmd_desc)
    :   m_IOCommandCount(cmd_desc.CommandCount),
        m_IOCommands(new GLIOCommand[cmd_desc.CommandCount])
{
    glGenFramebuffers(1, &m_FBO);
}

GLIOCommandBuffer::~GLIOCommandBuffer()
{
    glDeleteFramebuffers(1, &m_FBO);
}

void GLIOCommandBuffer::_executeCommandBuffer()
{
    for(uint32 i = 0, iend = m_IOCurrentCommand; i < iend; ++i)
    {
        auto& cmd = m_IOCommands[i];
        switch(cmd.CommandType)
        {
        case IOCommandMode::CopyBuffer:
        {
            cmd.Source.Buffer->bindToTarget(GLBufferTarget::GL_COPY_READ_BUFFER);
            cmd.Destination.Buffer->bindToTarget(GLBufferTarget::GL_COPY_WRITE_BUFFER);
            glCopyBufferSubData(GLBufferTarget::GL_COPY_READ_BUFFER, GLBufferTarget::GL_COPY_WRITE_BUFFER, cmd.SourceX, cmd.DestinationX, cmd.Width);
        } break;
        case IOCommandMode::CopyTexture:
        {
            auto& src_desc = cmd.Source.Texture->getDescription();
            auto& dst_desc = cmd.Destination.Texture->getDescription();
            glBindFramebuffer(GLFramebufferTarget::GL_READ_FRAMEBUFFER, m_FBO);
            if(dst_desc.Depth > 1)
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width &&
                           cmd.SourceY + cmd.Height <= src_desc.Height &&
                           cmd.SourceZ + cmd.Depth <= src_desc.Depth &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width &&
                           cmd.DestinationY + cmd.Height <= dst_desc.Height &&
                           cmd.DestinationZ + cmd.Depth <= dst_desc.Depth,
                           "Invalid coordinates specified");
                GLTextureTarget dst_target = dst_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_2D_ARRAY : GLTextureTarget::GL_TEXTURE_3D;
                GLTextureTarget src_target = src_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_2D_ARRAY : GLTextureTarget::GL_TEXTURE_3D;
                for(uint16 cur_depth = 0, end_depth = cmd.Depth; cur_depth < end_depth; ++cur_depth)
                {
                    glFramebufferTexture3D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), src_target, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip, cmd.SourceZ + cur_depth);
#ifndef NDEBUG
                    auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                    TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                    glCopyTexSubImage3D(dst_target, cmd.DestinationMip, cmd.DestinationX, cmd.DestinationY, cmd.DestinationZ + cur_depth, cmd.SourceX, cmd.SourceY, cmd.Width, cmd.Height);
                }
            }
            else if(dst_desc.Height > 1)
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width && 
                           cmd.SourceY + cmd.Height <= src_desc.Height &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width &&
                           cmd.DestinationY + cmd.Height <= dst_desc.Height,
                           "Invalid coordinates specified");
                GLTextureTarget dst_target = dst_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_1D_ARRAY : GLTextureTarget::GL_TEXTURE_2D;
                GLTextureTarget src_target = src_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_1D_ARRAY : GLTextureTarget::GL_TEXTURE_2D;
                glFramebufferTexture2D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), src_target, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip);
#ifndef NDEBUG
                auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                glCopyTexSubImage2D(dst_target, cmd.DestinationMip, cmd.DestinationX, cmd.DestinationY, cmd.SourceX, cmd.SourceY, cmd.Width, cmd.Height);
            }
            else
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width,
                           "Invalid coordinates specified");
                glFramebufferTexture1D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), GLTextureTarget::GL_TEXTURE_1D, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip);
#ifndef NDEBUG
                auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                glCopyTexSubImage1D(GLTextureTarget::GL_TEXTURE_1D, cmd.DestinationMip, cmd.DestinationX, cmd.SourceX, cmd.SourceY, cmd.Width);
            }
            glBindFramebuffer(GLFramebufferTarget::GL_READ_FRAMEBUFFER, 0);
        } break;
        case IOCommandMode::CopyStorageToBuffer:
        {
            cmd.Source.Storage->bindToTarget(GLBufferTarget::GL_COPY_READ_BUFFER);
            cmd.Destination.Buffer->bindToTarget(GLBufferTarget::GL_COPY_WRITE_BUFFER);
            glCopyBufferSubData(GLBufferTarget::GL_COPY_READ_BUFFER, GLBufferTarget::GL_COPY_WRITE_BUFFER, cmd.SourceX, cmd.DestinationX, cmd.Width);
        } break;
        case IOCommandMode::CopyStorageToTexture:
        {
            cmd.Source.Storage->bindToTarget(GLBufferTarget::GL_PIXEL_UNPACK_BUFFER);
            auto& dst_desc = cmd.Destination.Texture->getDescription();
            auto line_size = cmd.Width*DataFormatElementSize(dst_desc.Format);

            glPixelStorei(GLPixelStoreMode::GL_UNPACK_ROW_LENGTH, cmd.Width);
            glPixelStorei(GLPixelStoreMode::GL_UNPACK_IMAGE_HEIGHT, cmd.Height);

            auto tex_info = TranslateTextureInfo(dst_desc.Format);

            if(dst_desc.Depth > 1)
            {
                // TODO: check whether it works
                TGE_ASSERT(cmd.SourceX + cmd.Height*cmd.Depth*line_size <= cmd.Source.Storage->getSize() &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width &&
                           cmd.DestinationY + cmd.Height <= dst_desc.Height &&
                           cmd.DestinationZ + cmd.Depth <= dst_desc.Depth,
                           "Invalid coordinates specified");
                GLTextureTarget target = dst_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_2D_ARRAY : GLTextureTarget::GL_TEXTURE_3D;
                glTexSubImage3D(target, cmd.DestinationMip, cmd.DestinationX, cmd.DestinationY, cmd.DestinationZ, cmd.Width, cmd.Height, cmd.Depth, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.SourceX);
            }
            else if(dst_desc.Height > 1)
            {
                TGE_ASSERT(cmd.SourceX + cmd.Height*line_size <= cmd.Source.Storage->getSize() &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width &&
                           cmd.DestinationY + cmd.Height <= dst_desc.Height,
                           "Invalid coordinates specified");
                GLTextureTarget target = dst_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_1D_ARRAY : GLTextureTarget::GL_TEXTURE_2D;
                glTexSubImage2D(target, cmd.DestinationMip, cmd.DestinationX, cmd.DestinationY, cmd.Width, cmd.Height, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.SourceX);
            }
            else
            {
                TGE_ASSERT(cmd.SourceX + line_size <= cmd.Source.Storage->getSize() &&
                           cmd.DestinationX + cmd.Width <= dst_desc.Width,
                           "Invalid coordinates specified");
                glTexSubImage1D(GLTextureTarget::GL_TEXTURE_1D, cmd.DestinationMip, cmd.DestinationX, cmd.Width, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.SourceX);
            }
            glBindBuffer(GLBufferTarget::GL_PIXEL_UNPACK_BUFFER, 0);
        } break;
        case IOCommandMode::CopyBufferToStorage:
        {
            cmd.Source.Buffer->bindToTarget(GLBufferTarget::GL_COPY_READ_BUFFER);
            cmd.Destination.Storage->bindToTarget(GLBufferTarget::GL_COPY_WRITE_BUFFER);
            glCopyBufferSubData(GLBufferTarget::GL_COPY_READ_BUFFER, GLBufferTarget::GL_COPY_WRITE_BUFFER, cmd.SourceX, cmd.DestinationX, cmd.Width);
        } break;
        case IOCommandMode::CopyTextureToStorage:
        {
            cmd.Destination.Storage->bindToTarget(GLBufferTarget::GL_PIXEL_PACK_BUFFER);

            glPixelStorei(GLPixelStoreMode::GL_PACK_ROW_LENGTH, cmd.Width);
            glPixelStorei(GLPixelStoreMode::GL_PACK_IMAGE_HEIGHT, cmd.Height);

            auto& src_desc = cmd.Source.Texture->getDescription();
            auto tex_info = TranslateTextureInfo(src_desc.Format);
            auto line_size = cmd.Width*DataFormatElementSize(src_desc.Format);
            glBindFramebuffer(GLFramebufferTarget::GL_READ_FRAMEBUFFER, m_FBO);
            if(src_desc.Depth > 1)
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width &&
                           cmd.SourceY + cmd.Height <= src_desc.Height &&
                           cmd.SourceZ + cmd.Depth <= src_desc.Depth &&
                           cmd.DestinationX + cmd.Height*cmd.Depth*line_size <= cmd.Destination.Storage->getSize(),
                           "Invalid coordinates specified");
                GLTextureTarget target = src_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_2D_ARRAY : GLTextureTarget::GL_TEXTURE_3D;
                for(uint16 cur_depth = 0, end_depth = cmd.Depth; cur_depth < end_depth; ++cur_depth)
                {
                    glFramebufferTexture3D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), target, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip, cmd.SourceZ + cur_depth);
                    glReadBuffer(UINT_TO_GL_BUFFER_COLOR_ATTACHMENT(0));
#ifndef NDEBUG
                    auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                    TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                    glReadPixels(cmd.SourceX, cmd.SourceY, cmd.Width, cmd.Height, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.DestinationX + cur_depth*cmd.Height*cur_depth);
                }
            }
            else if(src_desc.Height > 1)
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width &&
                           cmd.SourceY + cmd.Height <= src_desc.Height &&
                           cmd.DestinationX + cmd.Height*line_size <= cmd.Destination.Storage->getSize(),
                           "Invalid coordinates specified");
                GLTextureTarget target = src_desc.Tiling == TextureTiling::Array ? GLTextureTarget::GL_TEXTURE_1D_ARRAY : GLTextureTarget::GL_TEXTURE_2D;
                glFramebufferTexture2D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), target, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip);
                glReadBuffer(UINT_TO_GL_BUFFER_COLOR_ATTACHMENT(0));
#ifndef NDEBUG
                auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                glReadPixels(cmd.SourceX, cmd.SourceY, cmd.Width, cmd.Height, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.DestinationX);
                CheckOpenGL();
            }
            else
            {
                TGE_ASSERT(cmd.SourceX + cmd.Width <= src_desc.Width &&
                           cmd.DestinationX + line_size <= cmd.Destination.Storage->getSize(),
                           "Invalid coordinates specified");
                glFramebufferTexture1D(GLFramebufferTarget::GL_READ_FRAMEBUFFER, UINT_TO_GL_COLOR_ATTACHMENT(0), GLTextureTarget::GL_TEXTURE_1D, cmd.Source.Texture->getCPUHandle(), cmd.SourceMip);
                glReadBuffer(UINT_TO_GL_BUFFER_COLOR_ATTACHMENT(0));
#ifndef NDEBUG
                auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_READ_FRAMEBUFFER);
                TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Framebuffer is broken");
#endif
                glReadPixels(cmd.SourceX, cmd.SourceY, cmd.Width, cmd.Height, tex_info.Format, tex_info.Type, static_cast<char*>(nullptr) + cmd.DestinationX);
            }
            glBindFramebuffer(GLFramebufferTarget::GL_READ_FRAMEBUFFER, 0);
            glBindBuffer(GLBufferTarget::GL_PIXEL_PACK_BUFFER, 0);
        } break;
        default: TGE_ASSERT(false, "Unsupported command type");
        }
        CheckOpenGL();
    }
}
}
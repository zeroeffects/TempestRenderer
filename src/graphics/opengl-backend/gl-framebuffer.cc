/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#include "tempest/graphics/opengl-backend/gl-framebuffer.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/utils/memory.hh"

namespace Tempest
{
GLFramebuffer::GLFramebuffer(GLRenderTarget** color, uint32_t color_rt_count, GLRenderTarget* depth)
{
    glGenFramebuffers(1, &m_FBO);

	auto* draw_buffers = TGE_TYPED_ALLOCA(GLAttachmentIndex, color_rt_count);

    GLint cur_fb;
    glGetIntegerv(GLParameterType::GL_FRAMEBUFFER_BINDING, &cur_fb);

    glBindFramebuffer(GLFramebufferTarget::GL_FRAMEBUFFER, m_FBO);

    for(uint32_t rt_idx = 0; rt_idx < color_rt_count; ++rt_idx)
    {
        GLAttachmentIndex gl_att = UINT_TO_GL_COLOR_ATTACHMENT(rt_idx);

        auto rt = color[rt_idx];

        switch(rt->getTarget())
        {
        case GLTextureTarget::GL_TEXTURE_1D:
        {
            glFramebufferTexture1D(GLFramebufferTarget::GL_FRAMEBUFFER, gl_att, GLTextureTarget::GL_TEXTURE_1D, rt->getCPUHandle(), 0);
        } break;
        case GLTextureTarget::GL_TEXTURE_2D:
        {
            glFramebufferTexture2D(GLFramebufferTarget::GL_FRAMEBUFFER, gl_att, GLTextureTarget::GL_TEXTURE_2D, rt->getCPUHandle(), 0);
        } break;
        case GLTextureTarget::GL_TEXTURE_2D_MULTISAMPLE:
        {
            glFramebufferTexture2D(GLFramebufferTarget::GL_FRAMEBUFFER, gl_att, GLTextureTarget::GL_TEXTURE_2D_MULTISAMPLE, rt->getCPUHandle(), 0);
        } break;
        default:
            TGE_ASSERT(false, "Unsupported attachment type");
            return;
        }

		draw_buffers[rt_idx] = gl_att;
    }

    glReadBuffer(GLBufferMode::GL_NONE);
	glDrawBuffers(color_rt_count, draw_buffers);

    if(depth)
    {
        switch(depth->getTarget())
        {
        case GLTextureTarget::GL_TEXTURE_1D:
        {
            glFramebufferTexture1D(GLFramebufferTarget::GL_FRAMEBUFFER, GLAttachmentIndex::GL_DEPTH_ATTACHMENT, GLTextureTarget::GL_TEXTURE_1D, depth->getCPUHandle(), 0);
        } break;
        case GLTextureTarget::GL_TEXTURE_2D:
        {
            glFramebufferTexture2D(GLFramebufferTarget::GL_FRAMEBUFFER, GLAttachmentIndex::GL_DEPTH_ATTACHMENT, GLTextureTarget::GL_TEXTURE_2D, depth->getCPUHandle(), 0);
        } break;
        case GLTextureTarget::GL_TEXTURE_2D_MULTISAMPLE:
        {
            glFramebufferTexture2D(GLFramebufferTarget::GL_FRAMEBUFFER, GLAttachmentIndex::GL_DEPTH_ATTACHMENT, GLTextureTarget::GL_TEXTURE_2D_MULTISAMPLE, depth->getCPUHandle(), 0);
        } break;
        default:
            TGE_ASSERT(false, "Unsupported attachment type");
            return;
        }
    }

    auto status = glCheckFramebufferStatus(GLFramebufferTarget::GL_FRAMEBUFFER);
    TGE_ASSERT(status == GLFramebufferStatus::GL_FRAMEBUFFER_COMPLETE, "Invalid framebuffer");

	glBindFramebuffer(GLFramebufferTarget::GL_FRAMEBUFFER, cur_fb);
}

GLFramebuffer::~GLFramebuffer()
{
    glDeleteFramebuffers(1, &m_FBO);
}

void GLFramebuffer::_bind()
{
    glBindFramebuffer(GLFramebufferTarget::GL_FRAMEBUFFER, m_FBO);
}
}
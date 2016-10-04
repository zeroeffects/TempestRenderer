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

#include "tempest/utils/config.hh"

#ifdef _MSC_VER
#   pragma warning(disable : 4503)
#endif

#ifndef DISABLE_CUDA
#   include <surface_types.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-backend.hh"
#include "tempest/graphics/opengl-backend/gl-command-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-state-object.hh"
#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/graphics/opengl-backend/gl-config.hh"
#include "tempest/graphics/opengl-backend/gl-storage.hh"
#include "tempest/graphics/opengl-backend/gl-framebuffer.hh"
#include "tempest/graphics/opengl-backend/gl-io-command-buffer.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/utils/memory.hh"
#include "tempest/utils/logging.hh"

#include <cassert>
#include <sstream>
#include <algorithm>
#include <cstring>

#include "xxhash/xxhash.h"

#undef GL_NO_ERROR
#undef GL_TEXTURE_2D
#undef GL_COLOR

namespace Tempest
{
template<class T, class TDeleter>
bool GLRenderingBackend::CompareIndirect<T, TDeleter>::operator()(const std::unique_ptr<T, TDeleter>& lhs, const std::unique_ptr<T, TDeleter>& rhs) const
{
    return *lhs == *rhs;
}

template<class T, class TDeleter>
size_t GLRenderingBackend::HashIndirect<T, TDeleter>::operator()(const std::unique_ptr<T, TDeleter>& ptr) const
{
    return XXH32(ptr.get(), sizeof(T), 0xEF1C1337);
}

#define BUFFER_OFFSET(offset) reinterpret_cast<GLubyte*>(offset)

#ifndef NDEBUG
    
const char* TranslateDebugSource(GLDebugSourceType source)
{
    switch(source)
    {
    case GLDebugSourceType::GL_DEBUG_SOURCE_API: return "OpenGL API";
    case GLDebugSourceType::GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "Window system";
    case GLDebugSourceType::GL_DEBUG_SOURCE_SHADER_COMPILER: return "Shader compiler";
    case GLDebugSourceType::GL_DEBUG_SOURCE_THIRD_PARTY: return "Third-party application";
    case GLDebugSourceType::GL_DEBUG_SOURCE_APPLICATION: return "Application";
    case GLDebugSourceType::GL_DEBUG_SOURCE_OTHER: break;
    default: break;
    }
    return "Unknown source";
}

const char* TranslateDebugType(GLDebugType type)
{
    switch(type)
    {
    case GLDebugType::GL_DEBUG_TYPE_ERROR: return "Error";
    case GLDebugType::GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "Deprecated";
    case GLDebugType::GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "Undefined behavior";
    case GLDebugType::GL_DEBUG_TYPE_PORTABILITY: return "Portability issue";
    case GLDebugType::GL_DEBUG_TYPE_PERFORMANCE: return "Performance issue";
    case GLDebugType::GL_DEBUG_TYPE_MARKER: return "Marker";
    case GLDebugType::GL_DEBUG_TYPE_PUSH_GROUP: return "Group push";
    case GLDebugType::GL_DEBUG_TYPE_POP_GROUP: return "Group pop";
    case GLDebugType::GL_DEBUG_TYPE_OTHER: break;
    default: break;
    }
    return "Unknown type";
}

LogLevel TranslateDebugSeverity(GLSeverityType severity)
{
    switch(severity)
    {
    case GLSeverityType::GL_DEBUG_SEVERITY_HIGH: return LogLevel::Fatal;
    case GLSeverityType::GL_DEBUG_SEVERITY_MEDIUM: return LogLevel::Error;
    case GLSeverityType::GL_DEBUG_SEVERITY_LOW: return LogLevel::Warning;
    case GLSeverityType::GL_DEBUG_SEVERITY_NOTIFICATION: break;
    default: break;
    }
    return LogLevel::Info;
}

static const LogLevel MinSeverity = LogLevel::Error;

void APIENTRY DebugLoggingCallback(GLDebugSourceType source, GLDebugType type, GLuint id, GLSeverityType severity, GLsizei length, const GLchar* message, const void*)
{
    auto log_level = TranslateDebugSeverity(severity);
    if(log_level < MinSeverity)
        return;
    Log(log_level, "[Source: ", TranslateDebugSource(source), "; Type: ", TranslateDebugType(type), "] ", message);
}

#endif

GLRenderingBackend::GLRenderingBackend()
{
}

GLRenderingBackend::~GLRenderingBackend()
{
    if(m_TexturesTable)
    {
        glDeleteBuffers(1, &m_TexturesTable);
    }

#ifdef _WIN32
    if(m_HGLRC)
    {
        Tempest::wglMakeCurrent(m_DC, nullptr);
        Tempest::wglDeleteContext(m_HGLRC);
    }
#else
    glXMakeCurrent(m_Display->nativeHandle(), 0, 0);
    glXDestroyContext(m_Display->nativeHandle(), m_GLXContext);
#endif
}

bool GLRenderingBackend::attach(OSWindowSystem& wnd_sys, GLWindow& gl_wnd)
{
#ifdef _WIN32
    wnd_sys;
    int flags = 0;
#ifndef NDEBUG
    flags |= WGL_CONTEXT_DEBUG_BIT_ARB;
#endif
    static const int ctx_attr_list[] =
    {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
        WGL_CONTEXT_MINOR_VERSION_ARB, 0,
        WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
        WGL_CONTEXT_FLAGS_ARB, flags,
        0
    };

    if(!m_HGLRC)
    {
        m_HGLRC = w32hackCreateContextAttribs(gl_wnd.getDC(), nullptr, ctx_attr_list);
        if(!m_HGLRC)
        {
            Log(LogLevel::Warning, "Failed to create OpenGL 3 context");
            return false;
        }
    }
    m_DC = gl_wnd.getDC();
    Tempest::wglMakeCurrent(m_DC, m_HGLRC);
#else
    auto fbconf = gl_wnd.getFBConfig();

    m_Display = &wnd_sys;
    auto display = m_Display->nativeHandle();

    if(!m_FBConfig)
    {
        m_FBConfig = fbconf;
        if(glXCreateContextAttribsARB)
        {
            int ctx_attr_list[] =
            {
                GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                GLX_CONTEXT_MINOR_VERSION_ARB, 0,
                None
            };

            m_GLXContext = glXCreateContextAttribsARB(display, *fbconf, 0, True, ctx_attr_list);
            if(!m_GLXContext)
            {
                Log(LogLevel::Error, "the application has failed to initialize an OpenGL 3.0 compatible GLX rendering context");
                return false;
            }
        }
        else
        {
            m_GLXContext = glXCreateNewContext(display, *fbconf, GLX_RGBA_TYPE, 0, True);
            XSync(display, False);
            if(!m_GLXContext)
            {
                Log(LogLevel::Error, "the application has failed to initialize an OpenGL GLX rendering context");
                return false;
            }
        }
    }
    else if(m_FBConfig != fbconf)
    {
        Log(LogLevel::Error, "For performance reasons context recreation is not supported.");
        return false;
    }

    glXMakeCurrent(display, gl_wnd.getWindowId(), m_GLXContext);
#endif
    return true;
}

void GLRenderingBackend::init()
{
    // Requires the library to be loaded!
#ifndef NDEBUG
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_430))
    {
        glEnable(GLCapabilityMode::GL_DEBUG_OUTPUT);
        glDebugMessageCallback(&DebugLoggingCallback, nullptr);
    }
#endif
#if !defined(TEMPEST_DISABLE_MDI_BINDLESS) && !defined(TEMPEST_DISABLE_MDI)
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_MDI_BINDLESS))
    {
        glEnableClientState(GLClientState::GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
        glEnableClientState(GLClientState::GL_ELEMENT_ARRAY_UNIFIED_NV);
    }
#endif
    RasterizerStates default_rast_state;
    BlendStates default_blend_state;
    DepthStencilStates default_depth_stencil_state;
    TranslateRasterizerStates(&default_rast_state, &m_DefaultRasterizerStates);
    TranslateBlendStates(&default_blend_state, &m_DefaultBlendState);
    TranslateDepthStencilStates(&default_depth_stencil_state, &m_DefaultDepthStencilStates);
    auto opengl_err = glGetError();
    if(opengl_err != GLErrorCode::GL_NO_ERROR)
    {
        Log(LogLevel::Error, "OpenGL: error: ", ConvertGLErrorToString(opengl_err));
        TGE_ASSERT(opengl_err == GLErrorCode::GL_NO_ERROR, "An error has occurred while using OpenGL");
        return;
    }
}

GLRenderTarget* GLRenderingBackend::createRenderTarget(const TextureDescription& desc, uint32_t flags)
{
    return new GLRenderTarget(desc, flags);
}

GLFramebuffer* GLRenderingBackend::createFramebuffer(GLRenderTarget** color, uint32_t color_rt_count, GLRenderTarget* depth)
{
    return new GLFramebuffer(color, color_rt_count, depth);
}

void GLRenderingBackend::destroyRenderResource(GLRenderTarget* render_target)
{
    delete render_target;
}

void GLRenderingBackend::setFramebuffer(GLFramebuffer* framebuffer)
{
    if(framebuffer)
		framebuffer->_bind();
	else
		glBindFramebuffer(GLFramebufferTarget::GL_FRAMEBUFFER, 0);
}
   
void GLRenderingBackend::destroyRenderResource(GLFramebuffer* framebuffer)
{
    delete framebuffer;
}

GLCommandBuffer* GLRenderingBackend::createCommandBuffer(const CommandBufferDescription& cmd_buf_desc)
{
    return new GLCommandBuffer(cmd_buf_desc);
}
    
void GLRenderingBackend::submitCommandBuffer(GLCommandBuffer* cmd_buffer)
{
    cmd_buffer->_executeCommandBuffer(this);
}

GLIOCommandBuffer* GLRenderingBackend::createIOCommandBuffer(const IOCommandBufferDescription& cmd_buf_desc)
{
    return new GLIOCommandBuffer(cmd_buf_desc);
}

void GLRenderingBackend::destroyRenderResource(GLIOCommandBuffer* cmd_buffer)
{
    delete cmd_buffer;
}

void GLRenderingBackend::submitCommandBuffer(GLIOCommandBuffer* cmd_buffer)
{
    cmd_buffer->_executeCommandBuffer();
}

void GLRenderingBackend::destroyRenderResource(GLCommandBuffer* buffer)
{
    delete buffer;
}

void GLRenderingBackend::setActiveTextures(uint32_t num_textures)
{
    if(m_ActiveTextures == num_textures)
        return;
    m_ActiveTextures = num_textures;

#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
    {
        if(m_TexturesTable)
        {
            glDeleteBuffers(1, &m_TexturesTable);
        }

        GLsizeiptr size = num_textures * AlignAddress(sizeof(GLuint64), 4 * sizeof(GLfloat));
        glGenBuffers(1, &m_TexturesTable);
        glBindBuffer(GLBufferTarget::GL_UNIFORM_BUFFER, m_TexturesTable);
        glBufferData(GLBufferTarget::GL_UNIFORM_BUFFER, size, nullptr, GLUsageMode::GL_DYNAMIC_DRAW);
    }
#endif
}

void GLRenderingBackend::setTextures(const BakedResourceTable* resource_table)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
    {
        TGE_ASSERT(resource_table->getSize() / (4 * sizeof(GLfloat)) <= m_ActiveTextures, "Texture descriptor overflow");
        glBindBuffer(GLBufferTarget::GL_UNIFORM_BUFFER, m_TexturesTable);
        auto* res_buf = reinterpret_cast<char*>(glMapBufferRange(GLBufferTarget::GL_UNIFORM_BUFFER, 0, resource_table->getSize(), GL_MAP_INVALIDATE_BUFFER_BIT|GL_MAP_WRITE_BIT));
        memcpy(res_buf, resource_table->get(), resource_table->getSize());
        glUnmapBuffer(GLBufferTarget::GL_UNIFORM_BUFFER);
        glBindBufferRange(GLBufferTarget::GL_UNIFORM_BUFFER, 0, m_TexturesTable, 0, m_ActiveTextures*sizeof(GLuint64));
    }
    else
#endif
    {
        auto* table_ptr = resource_table->get();
        size_t entry_count = resource_table->getSize() / (4 * sizeof(GLfloat));
        for(size_t i = 0; i < entry_count; ++i)
        {
            auto* tex_info = reinterpret_cast<const GLTextureBindInfo*>(table_ptr);
            glActiveTexture(UINT_TO_GL_TEXTURE(i));
            if(tex_info->handle != 0)
            {
                glBindTexture(tex_info->target, tex_info->handle);
            }
            else
            {
                glBindTexture(GLTextureTarget::GL_TEXTURE_2D, 0);
            }
            table_ptr += 4 * sizeof(GLfloat);
        }
        CheckOpenGL();
    }
}

uint32_t GLRenderingBackend::getTextureHandleSize()
{
    static_assert(sizeof(GLuint64) == sizeof(GLTextureBindInfo), "Invalid bind info size");
    return sizeof(GLuint64);
}

GLBuffer* GLRenderingBackend::createBuffer(size_t size, ResourceBufferType buffer_type, uint32_t flags, const void* data)
{
    return new GLBuffer(size, buffer_type, flags, data);
}

void GLRenderingBackend::destroyRenderResource(GLBuffer* buffer)
{
    delete buffer;
}

void GLRenderingBackend::setConstantBuffer(size_t idx, const GLBuffer* buf)
{
    buf->bindConstantBuffer(TEMPEST_UBO_START + (GLint)idx, 0, buf->getSize());
}

GLTexture* GLRenderingBackend::createTexture(const TextureDescription& desc, uint32_t flags, const void* data)
{
    return new GLTexture(desc, flags, data);
}

void GLRenderingBackend::destroyRenderResource(GLTexture* texture)
{
    delete texture;
}

GLStateObject* GLRenderingBackend::createStateObject(DataFormat*,
                                                     size_t,
                                                     DataFormat,
                                                     GLShaderProgram* shader_program,
                                                     DrawModes primitive_type,
                                                     const RasterizerStates* rasterizer_states,
                                                     const BlendStates* blend_states,
                                                     const DepthStencilStates* depth_stencil_states)
{
    // We kind of need a complete translation before doing the comparisons; otherwise we are going to make translation when
    // comparing it with each element.
    auto* gl_rast_states = &m_DefaultRasterizerStates;
    auto* gl_blend_states = &m_DefaultBlendState;
    auto* gl_depth_stencil_states = &m_DefaultDepthStencilStates;
    if(rasterizer_states)
    {
        gl_rast_states = new GLRasterizerStates;
        TranslateRasterizerStates(rasterizer_states, gl_rast_states);
        gl_rast_states = m_RasterizerStates.emplace(gl_rast_states).first->get();
    }
    if(blend_states)
    {
        gl_blend_states = new GLBlendStates;
        TranslateBlendStates(blend_states, gl_blend_states);
        gl_blend_states = m_BlendStates.emplace(gl_blend_states).first->get();
    }
    if(depth_stencil_states)
    {
        gl_depth_stencil_states = new GLDepthStencilStates;
        TranslateDepthStencilStates(depth_stencil_states, gl_depth_stencil_states);
        gl_depth_stencil_states = m_DepthStencilStates.emplace(gl_depth_stencil_states).first->get();
    }

    return m_StateObjects.emplace(new GLStateObject(shader_program, primitive_type, gl_rast_states, gl_blend_states, gl_depth_stencil_states)).first->get();
}

GLStorage* GLRenderingBackend::createStorageBuffer(StorageMode storage_type, uint32_t size)
{
    return new GLStorage(storage_type, size);
}

void GLRenderingBackend::destroyRenderResource(GLStorage* storage)
{
    delete storage;
}

void GLRenderingBackend::setScissorRect(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    glScissor(x, y, width, height);
}

void GLRenderingBackend::setViewportRect(uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    glViewport(x, y, w, h);
}

void GLRenderingBackend::clearColorBuffer(uint32_t idx, const Vector4& color)
{
    glClearBufferfv(GLBufferContentType::GL_COLOR, idx, Array(color));
    CheckOpenGL();
}

void GLRenderingBackend::clearDepthStencilBuffer(float depth, uint8_t stencil)
{
    glClearBufferfi(GLBufferContentType::GL_DEPTH_STENCIL, 0, depth, stencil);
    CheckOpenGL();
}

GLRenderingBackend::FenceType* GLRenderingBackend::createFence()
{
    return new GLsync{ 0 };
}

void GLRenderingBackend::pushFence(GLRenderingBackend::FenceType* fence)
{
    if(*fence)
        glDeleteSync(*fence);
    *fence = glFenceSync(GLSyncCondition::GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void GLRenderingBackend::destroyRenderResource(GLRenderingBackend::FenceType* fence)
{
    if(*fence)
        glDeleteSync(*fence);
    delete fence;
}

void GLRenderingBackend::waitFence(FenceType* fence)
{
    if(*fence == 0)
        return;
    glClientWaitSync(*fence, GL_SYNC_FLUSH_COMMANDS_BIT, std::numeric_limits<uint64_t>::max());
    *fence = 0;
}

#undef GL_NEAREST
#undef GL_LINEAR

void GLRenderingBackend::blitAttachmentToScreen(AttachmentType att, uint32_t idx,
												uint32_t src_x, uint32_t src_y, uint32_t dst_x, uint32_t dst_y, uint32_t w, uint32_t h)
{
	TGE_ASSERT(att != AttachmentType::Depth || idx == 0, "invalid attachment specified");
	glReadBuffer(att == AttachmentType::Color ? UINT_TO_GL_BUFFER_COLOR_ATTACHMENT(idx) : static_cast<GLBufferMode>(GLAttachmentIndex::GL_DEPTH_ATTACHMENT));

	glBindFramebuffer(GLFramebufferTarget::GL_DRAW_FRAMEBUFFER, 0);
	glBlitFramebuffer(src_x, src_y, src_x + w, src_y + h, dst_x, dst_y, dst_x + w, dst_y + h, att == AttachmentType::Color ? GL_COLOR_BUFFER_BIT : GL_DEPTH_BUFFER_BIT, GLFilterMode::GL_NEAREST);
}
}

#ifndef DISABLE_CUDA
#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"

namespace Tempest
{
void GLRenderingBackend::mapToCudaTexture(GLTexture* tex, uint32_t flags, CUDATextureResource* cuda_tex)
{
    auto target = tex->getTarget();
    auto handle = tex->getCPUHandle();
    auto err = cudaGraphicsGLRegisterImage(&cuda_tex->Resource, handle, (GLenum)target, flags);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to register image in CUDA: ", cudaGetErrorString(err));
        cuda_tex->Resource = nullptr;
        return;
    }
    cuda_tex->Description = tex->getDescription();
    CheckOpenGL();
}

void GLRenderingBackend::unmapCudaTexture(CUDATextureResource* cuda_tex)
{
    
}

void GLRenderingBackend::mapToCudaSurface(GLTexture* tex, uint32_t flags, CUDASurfaceResource* cuda_tex)
{
    mapToCudaTexture(tex, flags, &cuda_tex->Texture);

    auto err = cudaGraphicsMapResources(1, &cuda_tex->Texture.Resource, 0);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to map graphics resource: ", cudaGetErrorString(err));
        return;
    }

    err = cudaGraphicsSubResourceGetMappedArray(&cuda_tex->Array, cuda_tex->Texture.Resource, 0, 0);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create backbuffer surface: ", cudaGetErrorString(err));
        return;
    }

    cudaResourceDesc res_desc{};
    res_desc.res.array.array = cuda_tex->Array;
    res_desc.resType = cudaResourceTypeArray;
    err = cudaCreateSurfaceObject(&cuda_tex->Surface, &res_desc);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create backbuffer surface: ", cudaGetErrorString(err));
        return;
    }
}

void GLRenderingBackend::unmapCudaSurface(CUDASurfaceResource* cuda_tex)
{
    cudaDestroySurfaceObject(cuda_tex->Surface);
    cudaGraphicsUnmapResources(1, &cuda_tex->Texture.Resource, 0);
}
}
#endif


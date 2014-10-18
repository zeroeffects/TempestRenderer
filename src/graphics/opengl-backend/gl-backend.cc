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

#ifdef _MSC_VER
#   pragma warning(disable : 4503)
#endif

#include "tempest/graphics/opengl-backend/gl-backend.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-command-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/utils/logging.hh"

#include "GL/gl.h"

#include <cassert>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace Tempest
{

#define BUFFER_OFFSET(offset) reinterpret_cast<GLubyte*>(offset)

#ifndef NDEBUG
    
const char* TranslateDebugSource(GLenum source)
{
    switch(source)
    {
    case GL_DEBUG_SOURCE_API: return "OpenGL API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "Window system";
    case GL_DEBUG_SOURCE_SHADER_COMPILER: return "Shader compiler";
    case GL_DEBUG_SOURCE_THIRD_PARTY: return "Third-party application";
    case GL_DEBUG_SOURCE_APPLICATION: return "Application";
    case GL_DEBUG_SOURCE_OTHER: break;
    default: break;
    }
    return "Unknown source";
}

const char* TranslateDebugType(GLenum type)
{
    switch(type)
    {
    case GL_DEBUG_TYPE_ERROR: return "Error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "Deprecated";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "Undefined behavior";
    case GL_DEBUG_TYPE_PORTABILITY: return "Portability issue";
    case GL_DEBUG_TYPE_PERFORMANCE: return "Performance issue";
    case GL_DEBUG_TYPE_MARKER: return "Marker";
    case GL_DEBUG_TYPE_PUSH_GROUP: return "Group push";
    case GL_DEBUG_TYPE_POP_GROUP: return "Group pop";
    case GL_DEBUG_TYPE_OTHER: break;
    default: break;
    }
    return "Unknown type";
}

LogLevel TranslateDebugSeverity(GLenum severity)
{
    switch(severity)
    {
    case GL_DEBUG_SEVERITY_HIGH: return LogLevel::Fatal;
    case GL_DEBUG_SEVERITY_MEDIUM: return LogLevel::Error;
    case GL_DEBUG_SEVERITY_LOW: return LogLevel::Warning;
    case GL_DEBUG_SEVERITY_NOTIFICATION: break;
    default: break;
    }
    return LogLevel::Info;
}

void DebugLoggingCallback(GLenum source, GLenum type, GLenum id, GLenum severity, GLsizei length, const GLchar* message, const void*)
{
    Log(TranslateDebugSeverity(severity), "[Source: ", TranslateDebugSource(source), "; Type: ", TranslateDebugType(type), "] ", message);
}

#endif

GLRenderingBackend::GLRenderingBackend()
{
#ifndef NDEBUG
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(&DebugLoggingCallback, nullptr);
#endif
    glEnableClientState(GL_VERTEX_ATTRIB_ARRAY_UNIFIED_NV);
    glEnableClientState(GL_ELEMENT_ARRAY_UNIFIED_NV);
}

GLRenderingBackend::~GLRenderingBackend()
{
}

GLRenderTarget* GLRenderingBackend::createRenderTarget(const TextureDescription& desc, uint32 flags)
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}
    
GLFramebuffer* GLRenderingBackend::createFramebuffer()
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}
    
void GLRenderingBackend::setFramebuffer(GLFramebuffer* rt_batch)
{
    TGE_ASSERT(false, "Stub");
}
    
GLCommandBuffer* GLRenderingBackend::createCommandBuffer()
{
    return new GLCommandBuffer;
}
    
void GLRenderingBackend::submitCommandBuffer(GLCommandBuffer* cmd_buffer)
{
    cmd_buffer->_executeCommandBuffer(this);
}

void GLRenderingBackend::destroyRenderResource(GLCommandBuffer* buffer)
{
    delete buffer;
}
    
GLBuffer* GLRenderingBackend::createBuffer(size_t size, VBType vb_type, uint32 flags, const void* data)
{
    return new GLBuffer(size, vb_type, flags, data);
}
    
void GLRenderingBackend::destroyRenderResource(GLBuffer* buffer)
{
    delete buffer;
}
    
GLTexture* GLRenderingBackend::createTexture(const TextureDescription& desc, uint32 flags, const void* data)
{
    return new GLTexture(desc, flags, data);
}
    
void GLRenderingBackend::destroyRenderResource(GLTexture* texture)
{
    delete texture;
}
    
GLStateObject* GLRenderingBackend::createStateObject(const RasterizerStates* rasterizer_states, const BlendStates* blend_states, const DepthStencilStates* depth_stencil_states)
{
    TGE_ASSERT(false, "Stub");
    return nullptr;
}
    
void GLRenderingBackend::setStateObject(const GLStateObject* state_obj)
{
    TGE_ASSERT(false, "Stub");
}
    
void GLRenderingBackend::setScissorRect(uint32 x, uint32 y, uint32 width, uint32 height)
{
    glScissor(x, y, width, height);
}

void GLRenderingBackend::setViewportRect(uint32 x, uint32 y, uint32 w, uint32 h)
{
    glViewport(0, 0, w, h);
}
    
void GLRenderingBackend::clearColorBuffer(uint32 idx, const Vector4& color)
{
    glClearBufferfv(GL_COLOR, idx, color.elem);
    CheckOpenGL();
}
    
void GLRenderingBackend::clearDepthStencilBuffer(float depth, uint8 stencil)
{
    glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil);
}

void GLRenderingBackend::bindInputLayout(GLInputLayout* layout)
{
    if(!layout)
        return;
    for(size_t i = 0, iend = layout->getAttributeCount(); i < iend; ++i)
    {
        auto* vert_attr = layout->getAttribute(i);
        glVertexAttribFormat(i, vert_attr->Size, vert_attr->Type, vert_attr->Normalized, vert_attr->Offset);
        glVertexAttribBinding(i, vert_attr->Binding);
        glEnableVertexAttribArrayARB(i);
    }
    CheckOpenGL();
}
}

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

#include <algorithm>

#include "tempest/graphics/opengl-backend/gl-command-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-shader.hh"
#include "tempest/graphics/opengl-backend/gl-buffer.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-input-layout.hh"
#include "tempest/graphics/opengl-backend/gl-backend.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/utils/assert.hh"

//#define DISABLE_NV_OPTIMIZATION

namespace Tempest
{
struct DrawElementsIndirectCommand
{
    uint32  count;
    uint32  instanceCount;
    uint32  firstIndex;
    uint32  baseVertex;
    uint32  baseInstance;
};

struct BindlessPtrNV
{
    GLuint   index;
    GLuint   reserved; 
    GLuint64 address;
    GLuint64 length;
}; 

struct DrawElementsIndirectBindlessCommandNV
{
    DrawElementsIndirectCommand cmd;
    GLuint                      reserved; 
    BindlessPtrNV               indexBuffer;
    BindlessPtrNV               vertexBuffers[];
};

static GLenum TranslateDrawMode(DrawModes mode)
{
    switch(mode)
    {
    default: TGE_ASSERT(false, "Unknown draw topology"); return (GLenum)0;
    case DrawModes::PointList: return GL_POINTS;
    case DrawModes::LineStrip: return GL_LINE_STRIP;
    case DrawModes::LineList: return GL_LINES;
    case DrawModes::TriangleStrip: return GL_TRIANGLE_STRIP;
    case DrawModes::TriangleList: return GL_TRIANGLES;
    case DrawModes::LineStripAdjacency: return GL_LINE_STRIP_ADJACENCY;
    case DrawModes::LineListAdjacency: return GL_LINES_ADJACENCY;
    case DrawModes::TriangleStripAdjacency: return GL_TRIANGLE_STRIP_ADJACENCY;
    case DrawModes::TriangleListAdjacency: return GL_TRIANGLES_ADJACENCY;
    }
}

GLCommandBuffer::GLCommandBuffer()
{
    m_CommandBuffer.reserve(10000);
}

GLCommandBuffer::~GLCommandBuffer()
{
    if(m_GPUCommandBufferSize)
        glDeleteBuffers(1, &m_GPUCommandBuffer);
    if(m_ConstantBufferRingSize)
        glDeleteBuffers(1, &m_ConstantBufferRing);
    if(m_GPUFence)
        glDeleteSync(m_GPUFence);
}

void GLCommandBuffer::clear()
{
    m_CommandBuffer.clear();
    m_ConstantBufferReqRingSize = 0;
    m_CommandBufferReqSize = 0;
}

void GLCommandBuffer::enqueueBatch(const GLDrawBatch& draw_batch)
{
    if(draw_batch.LinkedShaderProgram == nullptr)
        return; // We don't care about broken programs.
    m_CommandBuffer.push_back(draw_batch);
    if(draw_batch.ResourceTable)
        m_ConstantBufferReqRingSize += draw_batch.ResourceTable->getSize();
#ifndef DISABLE_NV_OPTIMIZATION
    if(glMultiDrawElementsIndirectBindlessNV)
    {
        m_CommandBufferReqSize += sizeof(DrawElementsIndirectCommand) + 
                                sizeof(GLuint) +
                                sizeof(BindlessPtrNV)*(draw_batch.InputLayout ? (draw_batch.InputLayout->getAttributeCount() + 1) : 1);
    }
    else
#endif
    {
        m_CommandBufferReqSize += sizeof(DrawElementsIndirectCommand);
    }
}

void GLCommandBuffer::prepareCommandBuffer()
{
    std::sort(m_CommandBuffer.begin(), m_CommandBuffer.end(),
              [](const GLDrawBatch& lhs, const GLDrawBatch& rhs)
              {
                  return lhs.LinkedShaderProgram == rhs.LinkedShaderProgram ?
                      lhs.SortKey < rhs.SortKey :
                      lhs.LinkedShaderProgram < rhs.LinkedShaderProgram;
              });
                  
}

static void AllocateBuffer(GLenum type, size_t req_size, size_t* size, GLuint* gpu_buf, void** gpu_buf_ptr)
{
    if(req_size > *size)
    {
        if(*size)
        {
            glDeleteBuffers(1, gpu_buf);
        }
        glGenBuffers(1, gpu_buf);
        glBindBuffer(type, *gpu_buf);
        *size = 2*req_size;
        glBufferStorage(type, *size, 0,
                        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT);
        *gpu_buf_ptr = glMapBufferRange(type, 0, *size,
                                        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        CheckOpenGL();
    }
    else
    {
        glBindBuffer(type, *gpu_buf);
    }
}

static void ExecuteCommandBufferNV(GLRenderingBackend* backend, const std::vector<GLDrawBatch>& cpu_cmd_buf, GLvoid* gpu_cmd_buf_ptr, GLuint const_buf_ring, GLvoid* const_buf_ptr)
{
    // Naive to start with. TODO: Ring buffer.
    char *cmd_buf = reinterpret_cast<char*>(gpu_cmd_buf_ptr),
         *cmd_start = cmd_buf,
         *res_buf = reinterpret_cast<char*>(const_buf_ptr),
         *res_start = res_buf;
    GLuint cnt = 0;
    auto& first = cpu_cmd_buf.front();
    GLLinkedShaderProgram* prev_prog = first.LinkedShaderProgram;
    GLInputLayout* prev_layout = first.InputLayout;
    DrawModes prev_mode = first.PrimitiveType;
    GLVertexBufferDescription prev_vert_buffers[MAX_VERTEX_BUFFERS];
    std::copy_n(first.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);
    for(GLuint i = 0; i < MAX_VERTEX_BUFFERS; ++i)
    {
        auto& vb = first.VertexBuffers[i];
        if(vb.VertexBuffer)
        {
            glBindVertexBuffer(i, 0, vb.Offset, vb.Stride);
        }
    }
    
    prev_prog->bind();
    backend->bindInputLayout(prev_layout);
    
    for(auto& cpu_cmd : cpu_cmd_buf)
    {
        auto& gpu_cmd = *reinterpret_cast<DrawElementsIndirectBindlessCommandNV*>(cmd_buf);
        bool vb_not_equal = !std::equal(prev_vert_buffers, prev_vert_buffers + MAX_VERTEX_BUFFERS, cpu_cmd.VertexBuffers,
                                        [](const GLVertexBufferDescription& lhs, const GLVertexBufferDescription& rhs)
                                        {
                                            return lhs.Stride == rhs.Stride &&
                                                   lhs.Offset == rhs.Offset;
                                        });
        if(prev_prog != cpu_cmd.LinkedShaderProgram ||
           prev_mode != cpu_cmd.PrimitiveType ||
           prev_layout != cpu_cmd.InputLayout ||
           !vb_not_equal)
        {
            GLuint layout_size = (GLuint)(prev_layout ? prev_layout->getAttributeCount() : 0);
            
            if(res_buf != res_start)
            {
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
            }
            
            glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(prev_mode), GL_UNSIGNED_SHORT,
                                                  (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr)),
                                                  cnt, 0, layout_size);
            CheckOpenGL();
            cmd_start = cmd_buf;
            res_start = res_buf;
            cnt = 0;
            
            if(prev_prog != cpu_cmd.LinkedShaderProgram)
            {
                cpu_cmd.LinkedShaderProgram->bind();
                prev_prog = cpu_cmd.LinkedShaderProgram;
            }
            if(prev_layout != cpu_cmd.InputLayout)
            {
                backend->bindInputLayout(cpu_cmd.InputLayout);
                prev_layout = cpu_cmd.InputLayout;
            }
            if(vb_not_equal)
            {
                for(GLuint i = 0; i < MAX_VERTEX_BUFFERS; ++i)
                {
                    auto& vb = cpu_cmd.VertexBuffers[i];
                    if(vb.VertexBuffer)
                    {
                        glBindVertexBuffer(i, 0, vb.Offset, vb.Stride);
                    }
                }
                std::copy_n(cpu_cmd.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);
            }
            
            prev_mode = cpu_cmd.PrimitiveType;
        }
        
        size_t layout_size = cpu_cmd.InputLayout ? cpu_cmd.InputLayout->getAttributeCount() : 0;
        
        if(cpu_cmd.ResourceTable)
        {
            auto size = cpu_cmd.ResourceTable->getSize();
            std::copy_n(cpu_cmd.ResourceTable->get(), size, res_buf);
            res_buf += size;
        }
        
        gpu_cmd.cmd.count = cpu_cmd.VertexCount;
        gpu_cmd.cmd.instanceCount = 1;
        gpu_cmd.cmd.firstIndex = 0;
        gpu_cmd.cmd.baseVertex = cpu_cmd.BaseVertex;
        gpu_cmd.cmd.baseInstance = 0;
        gpu_cmd.indexBuffer.index = 0;
        gpu_cmd.indexBuffer.reserved = 0;
        gpu_cmd.indexBuffer.address = cpu_cmd.IndexBuffer->getGPUAddress();
        gpu_cmd.indexBuffer.length = cpu_cmd.IndexBuffer->getSize();
        for(size_t i = 0; i < layout_size; ++i)
        {
            auto* attr = prev_layout->getAttribute(i);
            auto bind_point = cpu_cmd.InputLayout->getAttribute(i)->Binding;
            auto& vb = cpu_cmd.VertexBuffers[bind_point];
            auto& gpu_vb = gpu_cmd.vertexBuffers[i];
            gpu_vb.index = bind_point;
            gpu_vb.reserved = 0;
            if(vb.VertexBuffer)
            {
                gpu_vb.address = vb.VertexBuffer->getGPUAddress();
                gpu_vb.length = vb.VertexBuffer->getSize();
            }
            else
            {
                gpu_vb.address = 0;
                gpu_vb.length = 0;
            }
        }

        cmd_buf += sizeof(DrawElementsIndirectCommand) + 
                   sizeof(GLuint) +
                   sizeof(BindlessPtrNV)*(layout_size + 1);
        ++cnt;
    }
    
    if(cnt)
    {
        auto* layout = cpu_cmd_buf.back().InputLayout;
        auto offset = (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr));
        
        if(res_buf != res_start)
        {
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
        }
        
        glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(prev_mode), GL_UNSIGNED_SHORT,
                                              offset, cnt, 0, static_cast<GLuint>(layout ? layout->getAttributeCount() : 0));
        CheckOpenGL();
    }
}

static void ExecuteCommandBufferARB(GLRenderingBackend* backend, const std::vector<GLDrawBatch>& cpu_cmd_buf, GLvoid* gpu_cmd_buf_ptr, GLuint const_buf_ring, GLvoid* const_buf_ptr)
{
    char *cmd_buf = reinterpret_cast<char*>(gpu_cmd_buf_ptr),
         *cmd_start = cmd_buf,
         *res_buf = reinterpret_cast<char*>(const_buf_ptr),
         *res_start = res_buf;
    GLuint cnt = 0;
    auto& first = cpu_cmd_buf.front();
    GLLinkedShaderProgram* prev_prog = first.LinkedShaderProgram;
    GLInputLayout* prev_layout = first.InputLayout;
    GLBuffer* prev_index_buffer = first.IndexBuffer;
    GLVertexBufferDescription prev_vert_buffers[MAX_VERTEX_BUFFERS];
    std::copy_n(first.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);
    DrawModes prev_mode = first.PrimitiveType;
    
    prev_prog->bind();
    prev_index_buffer->bindIndexBuffer();
    for(GLuint vb_idx = 0; vb_idx < MAX_VERTEX_BUFFERS; ++vb_idx)
    {
        auto& vb = prev_vert_buffers[vb_idx];
        if(vb.VertexBuffer)
        {
            vb.VertexBuffer->bindVertexBuffer(vb_idx, vb.Offset, vb.Stride);
        }
    }
    backend->bindInputLayout(prev_layout);
    
    for(auto& cpu_cmd : cpu_cmd_buf)
    {
        bool vb_not_equal = !std::equal(prev_vert_buffers, prev_vert_buffers + MAX_VERTEX_BUFFERS, cpu_cmd.VertexBuffers,
                                        [](const GLVertexBufferDescription& lhs, const GLVertexBufferDescription& rhs)
                                        {
                                            return lhs.VertexBuffer == rhs.VertexBuffer &&
                                                    lhs.Stride == rhs.Stride &&
                                                    lhs.Offset == rhs.Offset;
                                        });
        auto& gpu_cmd = *reinterpret_cast<DrawElementsIndirectCommand*>(cmd_buf);
        if(prev_prog != cpu_cmd.LinkedShaderProgram ||
           prev_mode != cpu_cmd.PrimitiveType ||
           prev_layout != cpu_cmd.InputLayout ||
           prev_index_buffer != cpu_cmd.IndexBuffer ||
           vb_not_equal)
        {
            size_t layout_size = prev_layout ? prev_layout->getAttributeCount() : 0;
            
            if(res_buf != res_start)
            {
                glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
            }
            
            glMultiDrawElementsIndirect(TranslateDrawMode(prev_mode), GL_UNSIGNED_SHORT,
                                        (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr)),
                                        cnt, 0);
            CheckOpenGL();
            cmd_start = cmd_buf;
            res_start = res_buf;
            cnt = 0;
            
            if(prev_prog != cpu_cmd.LinkedShaderProgram)
            {
                cpu_cmd.LinkedShaderProgram->bind();
                prev_prog = cpu_cmd.LinkedShaderProgram;    
            }
            if(prev_index_buffer != cpu_cmd.IndexBuffer)
            {
                cpu_cmd.IndexBuffer->bindIndexBuffer();
                prev_index_buffer = cpu_cmd.IndexBuffer;
            }
            if(vb_not_equal)
            {
                for(GLuint vb_idx = 0; vb_idx < MAX_VERTEX_BUFFERS; ++vb_idx)
                {
                    auto& prev_vb = prev_vert_buffers[vb_idx];
                    auto& cur_vb = cpu_cmd.VertexBuffers[vb_idx];
                    if(prev_vb.VertexBuffer != cur_vb.VertexBuffer ||
                    prev_vb.Offset != cur_vb.Offset ||
                    prev_vb.Stride != cur_vb.Stride)
                    {
                        cur_vb.VertexBuffer->bindVertexBuffer(vb_idx, cur_vb.Offset, cur_vb.Stride);
                        prev_vb = cur_vb;
                    }
                }
            }
            if(prev_layout != cpu_cmd.InputLayout)
            {
                backend->bindInputLayout(cpu_cmd.InputLayout);
                prev_layout = cpu_cmd.InputLayout;
            }
            
            prev_mode = cpu_cmd.PrimitiveType;
        }
        
        size_t layout_size = cpu_cmd.InputLayout ? cpu_cmd.InputLayout->getAttributeCount() : 0;
        
        if(cpu_cmd.ResourceTable)
        {
            auto size = cpu_cmd.ResourceTable->getSize();
            std::copy_n(cpu_cmd.ResourceTable->get(), size, res_buf);
            res_buf += size;
        }
        
        gpu_cmd.count = cpu_cmd.VertexCount;
        gpu_cmd.instanceCount = 1;
        gpu_cmd.firstIndex = 0;
        gpu_cmd.baseVertex = cpu_cmd.BaseVertex;
        gpu_cmd.baseInstance = 0;

        cmd_buf += sizeof(DrawElementsIndirectCommand);
        ++cnt;
    }
    
    if(cnt)
    {
        auto* layout = cpu_cmd_buf.back().InputLayout;
        auto offset = (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr));
        
        if(res_buf != res_start)
        {
            glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
        }
        
        glMultiDrawElementsIndirect(TranslateDrawMode(prev_mode), GL_UNSIGNED_SHORT,
                                    offset, cnt, 0);
        CheckOpenGL();
    }
}

void GLCommandBuffer::_executeCommandBuffer(GLRenderingBackend* backend)
{
    // Early out, don't bother with empty stuff.
    if(m_CommandBuffer.empty())
        return;
    
    AllocateBuffer(GL_DRAW_INDIRECT_BUFFER, m_CommandBufferReqSize, &m_GPUCommandBufferSize, &m_GPUCommandBuffer, &m_GPUCommandBufferPtr);
    AllocateBuffer(GL_SHADER_STORAGE_BUFFER, m_ConstantBufferReqRingSize, &m_ConstantBufferRingSize, &m_ConstantBufferRing, &m_ConstantBufferPtr);
    
    // Well, we pretty much wait forever, so don't bother with loops.
    if(m_GPUFence)
    {
        glClientWaitSync(m_GPUFence, GL_SYNC_FLUSH_COMMANDS_BIT, std::numeric_limits<uint64>::max());
        glDeleteSync(m_GPUFence);
    }
#ifndef DISABLE_NV_OPTIMIZATION
    if(glMultiDrawElementsIndirectBindlessNV)
    {
        ExecuteCommandBufferNV(backend, m_CommandBuffer, m_GPUCommandBufferPtr, m_ConstantBufferRing, m_ConstantBufferPtr);
    }
    else
#endif
    {
        ExecuteCommandBufferARB(backend, m_CommandBuffer, m_GPUCommandBufferPtr, m_ConstantBufferRing, m_ConstantBufferPtr);
    }   

    m_GPUFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}
}
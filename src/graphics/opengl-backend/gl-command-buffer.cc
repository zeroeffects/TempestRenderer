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
#include "tempest/graphics/opengl-backend/gl-config.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/memory.hh"



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

static GLDrawMode TranslateDrawMode(DrawModes mode)
{
    switch(mode)
    {
    default: TGE_ASSERT(false, "Unknown draw topology"); return (GLDrawMode)0;
    case DrawModes::PointList: return GLDrawMode::GL_POINTS;
    case DrawModes::LineStrip: return GLDrawMode::GL_LINE_STRIP;
    case DrawModes::LineList: return GLDrawMode::GL_LINES;
    case DrawModes::TriangleStrip: return GLDrawMode::GL_TRIANGLE_STRIP;
    case DrawModes::TriangleList: return GLDrawMode::GL_TRIANGLES;
    case DrawModes::LineStripAdjacency: return GLDrawMode::GL_LINE_STRIP_ADJACENCY;
    case DrawModes::LineListAdjacency: return GLDrawMode::GL_LINES_ADJACENCY;
    case DrawModes::TriangleStripAdjacency: return GLDrawMode::GL_TRIANGLE_STRIP_ADJACENCY;
    case DrawModes::TriangleListAdjacency: return GLDrawMode::GL_TRIANGLES_ADJACENCY;
    }
}

#define MAX_LAYOUT_SIZE 16

GLCommandBuffer::GLCommandBuffer(const CommandBufferDescription& desc)
    :   m_CommandBuffer(new GLDrawBatch[desc.CommandCount]),
        m_ConstantBufferSize(desc.ConstantsBufferSize),
        m_CommandBufferSize(desc.ConstantsBufferSize)
{
    GLuint cmd_buf_size = desc.CommandCount*sizeof(DrawElementsIndirectCommand);

    GLint alignment;

#ifndef TEMPEST_DISABLE_MDI
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_440))
    {
        GLuint buffers[2];
        glGenBuffers(2, buffers);
        m_GPUCommandBuffer = buffers[0];
        m_ConstantBuffer = buffers[1];
        glBindBuffer(GLBufferTarget::GL_DRAW_INDIRECT_BUFFER, m_GPUCommandBuffer);
#ifndef TEMPEST_DISABLE_MDI_BINDLESS
        if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_MDI_BINDLESS))
        {
            cmd_buf_size += desc.CommandCount*(sizeof(GLuint) + MAX_LAYOUT_SIZE*sizeof(BindlessPtrNV));
        }
#endif
        glBufferStorage(GLBufferTarget::GL_DRAW_INDIRECT_BUFFER, cmd_buf_size, 0,
                        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT);
        m_GPUCommandBufferPtr = glMapBufferRange(GLBufferTarget::GL_DRAW_INDIRECT_BUFFER, 0, cmd_buf_size,
                                                 GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        CheckOpenGL();
        GLuint const_buf_size = desc.ConstantsBufferSize;
        glBindBuffer(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, m_ConstantBuffer);
        glBufferStorage(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, const_buf_size, 0,
                        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_DYNAMIC_STORAGE_BIT);
        m_ConstantBufferPtr = glMapBufferRange(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, 0, const_buf_size,
                                               GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

        glGetIntegerv(GLParameterType::GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment);
        
        CheckOpenGL();
    }
    else
#endif
    {
        glGetIntegerv(GLParameterType::GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &alignment);
        
        glGenBuffers(1, &m_ConstantBuffer);
        glBindBuffer(GLBufferTarget::GL_UNIFORM_BUFFER, m_ConstantBuffer);
        glBufferData(GLBufferTarget::GL_UNIFORM_BUFFER, m_ConstantBufferSize, nullptr, GLUsageMode::GL_DYNAMIC_DRAW);
        CheckOpenGL();
    }

    m_Alignment = alignment;
}

GLCommandBuffer::~GLCommandBuffer()
{
    if(m_GPUFence)
    {
        glDeleteSync(m_GPUFence);
    }
#ifndef TEMPEST_DISABLE_MDI
    if(m_GPUCommandBuffer)
    {
        GLuint buffers[] = { m_GPUCommandBuffer, m_ConstantBuffer };
        glDeleteBuffers(TGE_FIXED_ARRAY_SIZE(buffers), buffers);
    }
    else
#endif
    {
        glDeleteBuffers(1, &m_ConstantBuffer);
    }
}

void GLCommandBuffer::clear()
{
    m_ConstantBufferReqSize = 0;
    m_CommandCount = 0;
}

bool GLCommandBuffer::enqueueBatch(const GLDrawBatch& draw_batch)
{
    if(draw_batch.PipelineState == nullptr)
        return true; // We don't care about broken pipeline state.

    if(m_CommandCount == m_CommandBufferSize)
        return false;

    if(draw_batch.ResourceTable)
    {
        auto size = draw_batch.ResourceTable->getSize();
        if(m_ConstantBufferReqSize + size > m_ConstantBufferSize)
            return false;

        #ifndef TEMPEST_DISABLE_MDI
            if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_440))
            {
                m_ConstantBufferReqSize += static_cast<uint32>(draw_batch.ResourceTable->getSize());
            }
            else
        #endif
            {
                m_ConstantBufferReqSize += AlignAddress(static_cast<uint32>(draw_batch.ResourceTable->getSize()), m_Alignment);
            }
    }

    m_CommandBuffer[m_CommandCount++] = draw_batch;
    return true;
}

void GLCommandBuffer::prepareCommandBuffer()
{
    std::sort(m_CommandBuffer.get(), m_CommandBuffer.get() + m_CommandCount,
              [](const GLDrawBatch& lhs, const GLDrawBatch& rhs)
              {
                  return lhs.PipelineState == rhs.PipelineState ?
                         lhs.SortKey < rhs.SortKey :
                         lhs.PipelineState < rhs.PipelineState;
              });
}

static void ExecuteCommandBufferNV(GLRenderingBackend* backend, GLDrawBatch* cpu_cmd_buf, uint32 cpu_cmd_buf_size, GLvoid* gpu_cmd_buf_ptr, size_t alignment, GLuint const_buf_ring, GLvoid* const_buf_ptr)
{
    // Naive to start with. TODO: Ring buffer.
    char *cmd_buf = reinterpret_cast<char*>(gpu_cmd_buf_ptr),
         *cmd_start = cmd_buf,
         *res_buf = reinterpret_cast<char*>(const_buf_ptr),
         *res_start = res_buf;
    GLuint cnt = 0;
    auto& first = *cpu_cmd_buf;
    auto* prev_state = first.PipelineState;
    DrawModes prev_mode = first.PipelineState->getPrimitiveType();
    GLVertexBufferDescription prev_vert_buffers[MAX_VERTEX_BUFFERS];
    std::copy_n(first.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);
    for(GLuint i = 0; i < MAX_VERTEX_BUFFERS; ++i)
    {
        auto& vb = first.VertexBuffers[i];
        if(vb.VertexBuffer)
        {
            glBindVertexBuffer(i, 0, 0, vb.Stride);
        }
    }
    
    prev_state->setup(nullptr, nullptr);
    
    for(uint32 cmd_idx = 0; cmd_idx < cpu_cmd_buf_size; ++cmd_idx)
    {
        auto& cpu_cmd = cpu_cmd_buf[cmd_idx];
        auto& gpu_cmd = *reinterpret_cast<DrawElementsIndirectBindlessCommandNV*>(cmd_buf);
        bool vb_not_equal = !std::equal(prev_vert_buffers, prev_vert_buffers + MAX_VERTEX_BUFFERS, cpu_cmd.VertexBuffers,
                                        [](const GLVertexBufferDescription& lhs, const GLVertexBufferDescription& rhs)
                                        {
                                            return lhs.Stride == rhs.Stride &&
                                                   lhs.Offset == rhs.Offset;
                                        });
        if(prev_state != cpu_cmd.PipelineState ||
           vb_not_equal)
        {
            GLuint layout_size = (GLuint)(prev_state->getInputLayout() ? prev_state->getInputLayout()->getAttributeCount() : 0);
            /*
            auto* cur_input_layout = cpu_cmd.PipelineState->getInputLayout();
            for(size_t i = cur_input_layout ? cur_input_layout->getAttributeCount() : 0, iend = layout_size; i < iend; ++i)
            {
                glDisableVertexAttribArray(i);
            }
            */
            if(res_buf != res_start)
            {
                glBindBufferRange(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, TEMPEST_GLOBALS_BUFFER, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
            }
            
            glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(prev_mode), GLType::GL_UNSIGNED_SHORT,
                                                  (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr)),
                                                  cnt, 0, layout_size);
            CheckOpenGL();
            cmd_start = cmd_buf;
            res_buf = AlignAddress(res_buf, alignment);
            res_start = res_buf;
            cnt = 0;
            
            if(prev_state != cpu_cmd.PipelineState)
            {
                cpu_cmd.PipelineState->setup(prev_state, nullptr);
                prev_state = cpu_cmd.PipelineState;
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
            
            prev_mode = cpu_cmd.PipelineState->getPrimitiveType();
        }
        
        size_t layout_size = prev_state->getInputLayout() ? prev_state->getInputLayout()->getAttributeCount() : 0;
        
        if(cpu_cmd.ResourceTable)
        {
            auto size = cpu_cmd.ResourceTable->getSize();
            std::copy_n(cpu_cmd.ResourceTable->get(), size, res_buf);
            res_buf += size;
        }
        
        gpu_cmd.cmd.count = cpu_cmd.VertexCount;
        gpu_cmd.cmd.instanceCount = 1;
        gpu_cmd.cmd.firstIndex = cpu_cmd.BaseIndex;
        gpu_cmd.cmd.baseVertex = cpu_cmd.BaseVertex;
        gpu_cmd.cmd.baseInstance = 0;
        gpu_cmd.indexBuffer.index = 0;
        gpu_cmd.indexBuffer.reserved = 0;
        if(cpu_cmd.IndexBuffer)
        {
            gpu_cmd.indexBuffer.address = cpu_cmd.IndexBuffer->getGPUAddress();
            gpu_cmd.indexBuffer.length = cpu_cmd.IndexBuffer->getSize();
        }
        else
        {
            gpu_cmd.indexBuffer.address = 0ULL;
            gpu_cmd.indexBuffer.length = 0ULL;
        }
        for(size_t i = 0; i < layout_size; ++i)
        {
            auto* attr = prev_state->getInputLayout()->getAttribute(i);
            auto bind_point = cpu_cmd.PipelineState->getInputLayout()->getAttribute(i)->Binding;
            auto& vb = cpu_cmd.VertexBuffers[bind_point];
            auto& gpu_vb = gpu_cmd.vertexBuffers[i];
            gpu_vb.index = bind_point;
            gpu_vb.reserved = 0;
            if(vb.VertexBuffer)
            {
                gpu_vb.address = vb.VertexBuffer->getGPUAddress() + vb.Offset;
                gpu_vb.length = vb.VertexBuffer->getSize();
            }
            else
            {
                gpu_vb.address = 0;
                gpu_vb.length = 0;
            }
        }

        TGE_ASSERT(layout_size < MAX_LAYOUT_SIZE, "Layout is capped at 16 attributes");

        cmd_buf += sizeof(DrawElementsIndirectCommand) + 
                   sizeof(GLuint) +
                   sizeof(BindlessPtrNV)*(layout_size + 1);
        ++cnt;
    }
    
    if(cnt)
    {
        auto* layout = prev_state->getInputLayout();
        auto offset = (char*)nullptr + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr));
        
        if(res_buf != res_start)
        {
            glBindBufferRange(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, TEMPEST_GLOBALS_BUFFER, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
        }
        
        glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(prev_mode), GLType::GL_UNSIGNED_SHORT,
                                              offset, cnt, 0, static_cast<GLuint>(layout ? layout->getAttributeCount() : 0));
        CheckOpenGL();
    }
}

static void ExecuteCommandBufferARB(GLRenderingBackend* backend, GLDrawBatch* cpu_cmd_buf, uint32 cpu_cmd_buf_size,
                                    GLvoid* gpu_cmd_buf_ptr, size_t alignment, GLuint const_buf_ring, GLvoid* const_buf_ptr)
{
    char *cmd_buf = reinterpret_cast<char*>(gpu_cmd_buf_ptr),
         *cmd_start = cmd_buf,
         *res_buf = reinterpret_cast<char*>(const_buf_ptr),
         *res_start = res_buf;
    GLuint cnt = 0;
    auto& first = *cpu_cmd_buf;
    auto* prev_state = first.PipelineState;
    GLBuffer* prev_index_buffer = first.IndexBuffer;
    GLVertexBufferDescription prev_vert_buffers[MAX_VERTEX_BUFFERS];
    std::copy_n(first.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);
    DrawModes prev_mode = first.PipelineState->getPrimitiveType();
    
    GLBufferTableEntry buffer_table[MAX_VERTEX_BUFFERS];
    memset(buffer_table, 0, sizeof(buffer_table));

    if(prev_index_buffer)
    {
        prev_index_buffer->bindIndexBuffer();
    }
    for(GLuint vb_idx = 0; vb_idx < MAX_VERTEX_BUFFERS; ++vb_idx)
    {
        auto& vb = prev_vert_buffers[vb_idx];
        if(vb.VertexBuffer)
        {
            vb.VertexBuffer->bindVertexBuffer(vb_idx, vb.Offset, vb.Stride);
            buffer_table[vb_idx].Offset = vb.Offset;
            buffer_table[vb_idx].Stride = vb.Stride;
        }
    }
    
    prev_state->setup(nullptr, buffer_table);

    for(uint32 cmd_idx = 0; cmd_idx < cpu_cmd_buf_size; ++cmd_idx)
    {
        auto& cpu_cmd = cpu_cmd_buf[cmd_idx];
        bool vb_not_equal = !std::equal(prev_vert_buffers, prev_vert_buffers + MAX_VERTEX_BUFFERS, cpu_cmd.VertexBuffers,
                                        [](const GLVertexBufferDescription& lhs, const GLVertexBufferDescription& rhs)
                                        {
                                            return lhs.VertexBuffer == rhs.VertexBuffer &&
                                                   lhs.Stride == rhs.Stride &&
                                                   lhs.Offset == rhs.Offset;
                                        });
        auto& gpu_cmd = *reinterpret_cast<DrawElementsIndirectCommand*>(cmd_buf);
        if(prev_state != cpu_cmd.PipelineState ||
           prev_index_buffer != cpu_cmd.IndexBuffer ||
           vb_not_equal)
        {
            if(res_buf != res_start)
            {
                glBindBufferRange(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, TEMPEST_GLOBALS_BUFFER, const_buf_ring, res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
            }
            
            glMultiDrawElementsIndirect(TranslateDrawMode(prev_mode), GLType::GL_UNSIGNED_SHORT,
                                        reinterpret_cast<char*>(nullptr) + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr)),
                                        cnt, 0);
            CheckOpenGL();
            cmd_start = cmd_buf;
            res_buf = AlignAddress(res_buf, alignment);
            res_start = res_buf;
            cnt = 0;
            
            if(prev_index_buffer != cpu_cmd.IndexBuffer && cpu_cmd.IndexBuffer)
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
                    if((prev_vb.VertexBuffer != cur_vb.VertexBuffer ||
                       prev_vb.Offset != cur_vb.Offset ||
                       prev_vb.Stride != cur_vb.Stride) &&
                       cur_vb.VertexBuffer)
                    {
                        cur_vb.VertexBuffer->bindVertexBuffer(vb_idx, cur_vb.Offset, cur_vb.Stride);
                        buffer_table[vb_idx].Offset = cur_vb.Offset;
                        buffer_table[vb_idx].Stride = cur_vb.Stride;
                        prev_vb = cur_vb;
                    }
                }
            }
            if(prev_state != cpu_cmd.PipelineState)
            {
                cpu_cmd.PipelineState->setup(prev_state, buffer_table);
                prev_state = cpu_cmd.PipelineState;
            }
            else if(vb_not_equal && prev_state->getInputLayout())
            {
                // Force rebind, if buffers changed.
                prev_state->getInputLayout()->bind(buffer_table);
            }

            prev_mode = cpu_cmd.PipelineState->getPrimitiveType();
        }
        
        if(cpu_cmd.ResourceTable)
        {
            auto size = cpu_cmd.ResourceTable->getSize();
            std::copy_n(cpu_cmd.ResourceTable->get(), size, res_buf);
            res_buf += size;
        }
        
        gpu_cmd.count = cpu_cmd.VertexCount;
        gpu_cmd.instanceCount = 1;
        gpu_cmd.firstIndex = cpu_cmd.BaseIndex;
        gpu_cmd.baseVertex = cpu_cmd.BaseVertex;
        gpu_cmd.baseInstance = 0;

        cmd_buf += sizeof(DrawElementsIndirectCommand);
        ++cnt;
    }
    
    if(cnt)
    {
        auto* offset = reinterpret_cast<char*>(nullptr) + (cmd_start - reinterpret_cast<char*>(gpu_cmd_buf_ptr));
        
        if(res_buf != res_start)
        {
            glBindBufferRange(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, TEMPEST_GLOBALS_BUFFER, const_buf_ring,
                              res_start - reinterpret_cast<char*>(const_buf_ptr), res_buf - res_start);
        }
        
        glMultiDrawElementsIndirect(TranslateDrawMode(prev_mode), GLType::GL_UNSIGNED_SHORT,
                                    offset, cnt, 0);
        CheckOpenGL();
    }
}

static void ExecuteCommandBufferOldStyle(GLRenderingBackend* backend, uint32 alignment, GLDrawBatch* cpu_cmd_buf, uint32 cpu_cmd_buf_size, GLuint const_buf_ring)
{
    auto& first = *cpu_cmd_buf;
    auto* prev_state = first.PipelineState;
    GLBuffer* prev_index_buffer = first.IndexBuffer;
    GLVertexBufferDescription prev_vert_buffers[MAX_VERTEX_BUFFERS];
    std::copy_n(first.VertexBuffers, MAX_VERTEX_BUFFERS, prev_vert_buffers);

    GLBufferTableEntry buffer_table[MAX_VERTEX_BUFFERS];
    memset(buffer_table, 0, sizeof(buffer_table));

    if(prev_index_buffer)
    {
        prev_index_buffer->bindIndexBuffer();
    }
    for(GLuint vb_idx = 0; vb_idx < MAX_VERTEX_BUFFERS; ++vb_idx)
    {
        auto& vb = prev_vert_buffers[vb_idx];
        if(vb.VertexBuffer)
        {
            auto& buf_table_entry = buffer_table[vb_idx];
            buf_table_entry.Offset = vb.Offset;
            buf_table_entry.Stride = vb.Stride;
            buf_table_entry.Buffer = vb.VertexBuffer->getCPUHandle();
        }
    }

    prev_state->setup(nullptr, buffer_table);

    GLintptr offset = 0;

    for(uint32 cmd_idx = 0; cmd_idx < cpu_cmd_buf_size; ++cmd_idx)
    {
        auto& cpu_cmd = cpu_cmd_buf[cmd_idx];
        bool vb_not_equal = !std::equal(prev_vert_buffers, prev_vert_buffers + MAX_VERTEX_BUFFERS, cpu_cmd.VertexBuffers,
                                        [](const GLVertexBufferDescription& lhs, const GLVertexBufferDescription& rhs)
        {
            return lhs.VertexBuffer == rhs.VertexBuffer &&
                lhs.Stride == rhs.Stride &&
                lhs.Offset == rhs.Offset;
        });
      
        if(prev_index_buffer != cpu_cmd.IndexBuffer && cpu_cmd.IndexBuffer)
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
                if((prev_vb.VertexBuffer != cur_vb.VertexBuffer ||
                    prev_vb.Offset != cur_vb.Offset ||
                    prev_vb.Stride != cur_vb.Stride) && cur_vb.VertexBuffer)
                {
                    cur_vb.VertexBuffer->bindVertexBuffer(vb_idx, cur_vb.Offset, cur_vb.Stride);
                    buffer_table[vb_idx].Offset = cur_vb.Offset;
                    buffer_table[vb_idx].Stride = cur_vb.Stride;
                    prev_vb = cur_vb;
                }
            }
        }
        if(prev_state != cpu_cmd.PipelineState)
        {
            cpu_cmd.PipelineState->setup(prev_state, buffer_table);
            prev_state = cpu_cmd.PipelineState;
        }
        else if(vb_not_equal)
        {
            prev_state->getInputLayout()->bind(buffer_table);
        }
        if(cpu_cmd.ResourceTable)
        {
            auto real_size = cpu_cmd.ResourceTable->getSize();
            auto size = AlignAddress(static_cast<uint32>(real_size), alignment);
            glBindBufferRange(GLBufferTarget::GL_UNIFORM_BUFFER, TEMPEST_GLOBALS_BUFFER, const_buf_ring, offset, real_size);
            offset += size;
        }

        auto mode = TranslateDrawMode(prev_state->getPrimitiveType());
        glDrawElementsBaseVertex(mode, cpu_cmd.VertexCount, GLType::GL_UNSIGNED_SHORT,
                                 reinterpret_cast<char*>(nullptr) + cpu_cmd.BaseIndex*sizeof(GLushort), cpu_cmd.BaseVertex);
        CheckOpenGL();
    }
}

void GLCommandBuffer::_executeCommandBuffer(GLRenderingBackend* backend)
{
    // Early out, don't bother with empty stuff.
    if(m_CommandCount == 0)
        return;
    
#ifndef TEMPEST_DISABLE_MDI
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_440))
    {
        glBindBuffer(GLBufferTarget::GL_DRAW_INDIRECT_BUFFER, m_GPUCommandBuffer);
        glBindBuffer(GLBufferTarget::GL_SHADER_STORAGE_BUFFER, m_ConstantBuffer);

        // Well, we pretty much wait forever, so don't bother with loops.
        if(m_GPUFence)
        {
            glClientWaitSync(m_GPUFence, GL_SYNC_FLUSH_COMMANDS_BIT, std::numeric_limits<uint64>::max());
            glDeleteSync(m_GPUFence);
        }
#ifndef TEMPEST_DISABLE_MDI_BINDLESS
        if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_MDI_BINDLESS))
        {
            ExecuteCommandBufferNV(backend, m_CommandBuffer.get(), m_CommandCount, m_GPUCommandBufferPtr, m_Alignment, m_ConstantBuffer, m_ConstantBufferPtr);
        }
        else
#endif
        {
            ExecuteCommandBufferARB(backend, m_CommandBuffer.get(), m_CommandCount, m_GPUCommandBufferPtr, m_Alignment, m_ConstantBuffer, m_ConstantBufferPtr);
        }

        m_GPUFence = glFenceSync(GLSyncCondition::GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
    else
#endif
    {
        // Complete state dump for this version before scheduling
        glBindBuffer(GLBufferTarget::GL_UNIFORM_BUFFER, m_ConstantBuffer);
        auto* res_buf = reinterpret_cast<char*>(glMapBuffer(GLBufferTarget::GL_UNIFORM_BUFFER, GLAccessMode::GL_WRITE_ONLY));
        auto* cpu_cmd_buf = m_CommandBuffer.get();
        for(uint32 cmd_idx = 0, cmd_idx_end = m_CommandCount; cmd_idx < cmd_idx_end; ++cmd_idx)
        {
            auto& cpu_cmd = cpu_cmd_buf[cmd_idx];
            if(cpu_cmd.ResourceTable)
            {
                auto real_size = cpu_cmd.ResourceTable->getSize();
                auto offset = AlignAddress(real_size, m_Alignment);
                std::copy_n(cpu_cmd.ResourceTable->get(), real_size, res_buf);
                res_buf += offset;
            }
        }
        glUnmapBuffer(GLBufferTarget::GL_UNIFORM_BUFFER);

        ExecuteCommandBufferOldStyle(backend, m_Alignment, m_CommandBuffer.get(), m_CommandCount, m_ConstantBuffer);
    }
}
}
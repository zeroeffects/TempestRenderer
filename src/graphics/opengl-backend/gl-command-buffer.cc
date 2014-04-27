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
#include "tempest/graphics/opengl-backend/gl-utils.hh"
#include "tempest/utils/assert.hh"

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
    if(draw_batch.ShaderProgram == nullptr)
        return; // We don't care about broken programs.
    m_CommandBuffer.push_back(draw_batch);
    if(draw_batch.ResourceTable)
        m_ConstantBufferReqRingSize += draw_batch.ResourceTable->getSize();
    m_CommandBufferReqSize += sizeof(DrawElementsIndirectCommand) + 
                              sizeof(GLuint) +
                              sizeof(BindlessPtrNV)*(draw_batch.InputLayout ? (draw_batch.InputLayout->getAttributeCount() + 1) : 1);
}

void GLCommandBuffer::prepareCommandBuffer()
{
    std::sort(m_CommandBuffer.begin(), m_CommandBuffer.end(),
              [](const GLDrawBatch& lhs, const GLDrawBatch& rhs)
              {
                  return lhs.ShaderProgram == rhs.ShaderProgram ?
                      lhs.SortKey < rhs.SortKey :
                      lhs.ShaderProgram < rhs.ShaderProgram;
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
}

void GLCommandBuffer::_executeCommandBuffer()
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
    
    // Naive to start with. TODO: Ring buffer.
    char *cmd_buf = reinterpret_cast<char*>(m_GPUCommandBufferPtr),
         *cmd_start = cmd_buf;
    size_t cnt = 0;
    auto& first = m_CommandBuffer.front();
    GLShaderProgram* prev_prog = first.ShaderProgram;
    GLInputLayout* layout = first.InputLayout;
    DrawModes mode = first.PrimitiveType;
    
    prev_prog->bind();
    prev_prog->setupInputLayout(layout);
    
    for(auto& cpu_cmd : m_CommandBuffer)
    {
        auto& gpu_cmd = *reinterpret_cast<DrawElementsIndirectBindlessCommandNV*>(cmd_buf);
        if(prev_prog != cpu_cmd.ShaderProgram ||
           mode != cpu_cmd.PrimitiveType ||
           layout != cpu_cmd.InputLayout)
        {
            size_t layout_size = layout ? layout->getAttributeCount() : 0;
            
            glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(mode), GL_UNSIGNED_SHORT,
                                                  (char*)nullptr + (cmd_start - reinterpret_cast<char*>(m_GPUCommandBufferPtr)),
                                                  cnt, 0, layout_size);
            CheckOpenGL();
            cmd_start = cmd_buf;
            cnt = 0;
            mode = cpu_cmd.PrimitiveType;
            layout = cpu_cmd.InputLayout;
            prev_prog = cpu_cmd.ShaderProgram;
            
            if(prev_prog != cpu_cmd.ShaderProgram)
                prev_prog->bind();
            if(layout != cpu_cmd.InputLayout)
                prev_prog->setupInputLayout(layout);
        }
        
        size_t layout_size = cpu_cmd.InputLayout ? cpu_cmd.InputLayout->getAttributeCount() : 0;
        
        // TODO: SSBO, stride
        
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
            auto* attr = layout->getAttribute(i);
            gpu_cmd.vertexBuffers[i].index = i;
            gpu_cmd.vertexBuffers[i].reserved = 0;
            gpu_cmd.vertexBuffers[i].address = cpu_cmd.VertexBuffers[i]->getGPUAddress();
            gpu_cmd.vertexBuffers[i].length = cpu_cmd.VertexBuffers[i]->getSize();
        }

        cmd_buf += sizeof(DrawElementsIndirectCommand) + 
                   sizeof(GLuint) +
                   sizeof(BindlessPtrNV)*(layout_size + 1);
        ++cnt;
    }
    
    if(cnt)
    {
        auto* layout = m_CommandBuffer.back().InputLayout;
        auto offset = (char*)nullptr + (cmd_start - reinterpret_cast<char*>(m_GPUCommandBufferPtr));
        glMultiDrawElementsIndirectBindlessNV(TranslateDrawMode(mode), GL_UNSIGNED_SHORT,
                                              offset, cnt, 0, layout ? layout->getAttributeCount() : 0);
        CheckOpenGL();
    }

    m_GPUFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}
}
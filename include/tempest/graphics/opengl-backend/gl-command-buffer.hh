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

#ifndef _GL_COMMAND_BUFFER_HH_
#define _GL_COMMAND_BUFFER_HH_

#include "tempest/graphics/rendering-definitions.hh"

#ifdef _WIN32
    #include <windows.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"

#include <cstddef>
#include <vector>
#include <memory>

namespace Tempest
{
class GLRenderingBackend;
class GLBakedResourceTable;
class GLLinkedShaderProgram;
class GLStateObject;
class GLBuffer;

struct GLVertexBufferDescription
{
    GLBuffer*               VertexBuffer = nullptr;
    uint32                  Stride = 0;
    uint32                  Offset = 0;
};

// TODO: make it cache friendlier, by pool allocating and using offsets instead
struct GLDrawBatch
{
    uint64                    SortKey = 0;
    uint32                    VertexCount = 0;
    uint32                    BaseVertex = 0;
    GLBakedResourceTable*     ResourceTable = nullptr;
    GLStateObject*            PipelineState = nullptr;
    GLBuffer*                 IndexBuffer = nullptr;
    GLVertexBufferDescription VertexBuffers[MAX_VERTEX_BUFFERS];
};

class GLCommandBuffer
{
    std::unique_ptr<GLDrawBatch[]> m_CommandBuffer;

    uint32                         m_CommandBufferSize     = 0;
    uint32                         m_ConstantBufferSize    = 0;

    uint32                         m_CommandCount          = 0;
    uint32                         m_ConstantBufferReqSize = 0;

    GLsync                         m_GPUFence              = 0;

    GLuint                         m_ConstantBuffer        = 0;
    GLvoid*                        m_ConstantBufferPtr     = nullptr;
    
    GLuint                         m_GPUCommandBuffer      = 0;
    GLvoid*                        m_GPUCommandBufferPtr   = nullptr;
public:
    typedef GLDrawBatch DrawBatchType;
    
    explicit GLCommandBuffer(const CommandBufferDescription& cmd_buf_desc);
     ~GLCommandBuffer();
    
    //! \remarks Can be used outside of the rendering thread.
    void clear();
    
    //! \remarks Can be used outside of the rendering thread.
    bool enqueueBatch(const GLDrawBatch& draw_batch);
    
    //! Called by the user to prepare the command buffer for submission.
    void prepareCommandBuffer();
    
    //! Called by the rendering backend to initiate buffer transfer.
    void _executeCommandBuffer(GLRenderingBackend* backend);
};
}

#endif //_GL_COMMAND_BUFFER_HH_
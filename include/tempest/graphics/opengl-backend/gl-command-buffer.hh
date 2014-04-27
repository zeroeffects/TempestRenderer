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

#include "GL/gl.h"

#include <cstddef>
#include <vector>

#include "tempest/graphics/rendering-definitions.hh"

#include <iostream>

namespace Tempest
{
class GLBakedResourceTable;
class GLShaderProgram;
class GLInputLayout;
class GLBuffer;

#ifndef MAX_VERTEX_BUFFERS
    // Ok, you can have two. More is just going to ruin your performance
    #define MAX_VERTEX_BUFFERS 2
#endif

// 64 bytes, if you don't change anything. Should be cache friendly
struct GLDrawBatch
{
    DrawModes               PrimitiveType = DrawModes::TriangleList;
    uint16                  VertexCount   = 0;
    uint16                  BaseVertex    = 0;
    GLBakedResourceTable*   ResourceTable = nullptr;
    GLShaderProgram*        ShaderProgram = nullptr;
    GLInputLayout*          InputLayout   = nullptr;
    uint64                  SortKey       = 0;
    GLBuffer*               IndexBuffer   = nullptr;
    GLBuffer*               VertexBuffers[MAX_VERTEX_BUFFERS];
};

class GLCommandBuffer
{
    std::vector<GLDrawBatch>     m_CommandBuffer;
    size_t                       m_CommandBufferReqSize      = 0;
    size_t                       m_ConstantBufferReqRingSize = 0;
    
    GLuint                       m_ConstantBufferRing        = 0;
    GLvoid*                      m_ConstantBufferPtr         = nullptr;
    size_t                       m_ConstantBufferRingSize    = 0;
    
    GLsync                       m_GPUFence                  = 0;
    
    GLuint                       m_GPUCommandBuffer          = 0;
    GLvoid*                      m_GPUCommandBufferPtr       = nullptr;
    size_t                       m_GPUCommandBufferSize      = 0;
    size_t                       m_GPUCommandBufferStart     = 0;
public:
    explicit GLCommandBuffer();
     ~GLCommandBuffer();
    
    //! \remarks Can be used outside of the rendering thread.
    void clear();
    
    //! \remarks Can be used outside of the rendering thread.
    void enqueueBatch(const GLDrawBatch& draw_batch);
    
    //! Called by the user to prepare the command buffer for submission.
    void prepareCommandBuffer();
    
    //! Called by the rendering backend to initiate buffer transfer.
    void _executeCommandBuffer();
};
}

#endif //_GL_COMMAND_BUFFER_HH_
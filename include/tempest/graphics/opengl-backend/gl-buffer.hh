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

#ifndef _TEMPEST_GL_BUFFER_HH_
#define _TEMPEST_GL_BUFFER_HH_

#include <GL/gl.h>
#include <cstddef>

#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
class GLBuffer
{
    size_t          m_Size;
    GLuint          m_Buffer;
    GLuint64        m_GPUAddress;
public:
    explicit GLBuffer(size_t size, VBType vb_type, size_t usage, const void* data);
     ~GLBuffer();
    
    void bindVertexBuffer(GLuint bind_slot, GLintptr offset, GLintptr stride);
    void bindIndexBuffer();
    
    GLuint64 getGPUAddress() const { return m_GPUAddress; }
    size_t getSize() const { return m_Size; }
};
}

#endif // _TEMPEST_GL_BUFFER_HH_
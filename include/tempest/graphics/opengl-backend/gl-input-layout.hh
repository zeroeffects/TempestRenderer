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

#ifndef _TEMPEST_GL_INPUT_LAYOUT_HH_
#define _TEMPEST_GL_INPUT_LAYOUT_HH_

#ifdef _WIN32
    #include <windows.h>
#endif

#include <GL/gl.h>
#include <stdlib.h>

#include "tempest/utils/patterns.hh"

namespace Tempest
{
struct GLVertexAttributeDescription
{
    int       Binding;
    GLenum    Type;
    int       Size;
    GLboolean Normalized;
    int       Offset;
};

struct VertexAttributeDescription;

class GLInputLayout
{
    PACKED_DATA(GLVertexAttributeDescription) m_Attributes;
public:    
    GLInputLayout(const GLInputLayout&)=delete;
    GLInputLayout& operator=(const GLInputLayout&)=delete;
    
    const GLVertexAttributeDescription* getAttribute(size_t idx) const { return m_Attributes.Values + idx; }
    size_t getAttributeCount() const { return m_Attributes.Count; }

private:
    GLInputLayout(size_t count, const VertexAttributeDescription* arr);
    ~GLInputLayout()=default;
};
}

#endif // _TEMPEST_GL_INPUT_LAYOUT_HH_
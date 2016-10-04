/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2014 Zdravko Velinov
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

#ifndef _TEMPEST_GL_STORAGE_HH_
#define _TEMPEST_GL_STORAGE_HH_

#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/graphics/opengl-backend/gl-library.hh"

namespace Tempest
{
struct TextureDescription;

class GLStorage
{
    uint32_t        m_Size;
    GLuint          m_Storage;
    GLBufferTarget  m_Target;
    GLbitfield      m_Access;
public:
    GLStorage(StorageMode storage_type, uint32_t size);
    ~GLStorage();

    uint32_t getSize() const { return m_Size; }

    void bindToTarget(GLBufferTarget target) { glBindBuffer(target, m_Storage); }

    void storeLinearBuffer(uint32_t offset, uint32_t size, const void* data);
    void storeTexture(uint32_t offset, const TextureDescription& tex_desc, const void* data);

    void extractLinearBuffer(uint32_t offset, uint32_t size, void* data);
    void extractTexture(uint32_t offset, const TextureDescription& tex_desc, void* data);
};
}

#endif // _TEMPEST_GL_STORAGE_HH_
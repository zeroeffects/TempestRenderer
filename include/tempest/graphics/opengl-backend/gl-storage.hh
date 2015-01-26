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
    uint32      m_Size;
    GLuint      m_Storage;
    uint8*      m_DataPtr;
public:
    GLStorage(StorageMode storage_type, uint32 size);
    ~GLStorage();

    uint32 getSize() const { return m_Size; }

    void bindToTarget(GLBufferTarget target) { glBindBuffer(target, m_Storage); }

    void storeLinearBuffer(uint32 offset, uint32 size, const void* data);
    void storeTexture(uint32 offset, const TextureDescription& tex_desc, const void* data);

    void extractLinearBuffer(uint32 offset, uint32 size, void* data);
    void extractTexture(uint32 offset, const TextureDescription& tex_desc, void* data);
};
}

#endif // _TEMPEST_GL_STORAGE_HH_
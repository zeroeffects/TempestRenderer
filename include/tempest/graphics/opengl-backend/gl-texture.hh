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

#ifndef _GL_TEXTURE_HH_
#define _GL_TEXTURE_HH_

#include "tempest/graphics/texture.hh"
#include "tempest/utils/types.hh"

#ifdef _WIN32
    #include <windows.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"

namespace Tempest
{
struct Vector4;
    
class GLTexture
{
    TextureDescription  m_Description;
    GLTextureTarget     m_Target;
    uint32              m_Flags;
    GLuint              m_Texture;
    mutable GLuint64    m_GPUHandle;
public:
    explicit GLTexture(const TextureDescription& desc, uint32 flags, const void* data);
     ~GLTexture();
    
    void setFilter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter, SamplerMode sampler_mode = SamplerMode::Default);
    void setWrapMode(WrapMode x_axis, WrapMode y_axis, WrapMode z_axis);
    void setMipLODBias(float lod_bias);
    void setMaxAnisotrophy(uint32 max_anisotropy);
    void setCompareFunction(ComparisonFunction compare_func);
    void setBorderColor(const Vector4& color);
    void setMinLOD(float min_lod);
    void setMaxLOD(float max_lod);
     
    //! \remarks Once you extract the handle state changes are going to be ignored.
    GLuint64 getHandle() const;
     
    const TextureDescription& getDescription() const { return m_Description; }
};
}

#endif // _GL_TEXTURE_HH_
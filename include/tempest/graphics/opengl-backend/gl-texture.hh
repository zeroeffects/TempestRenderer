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

#include "tempest/graphics/opengl-backend/gl-config.hh"
#include "tempest/graphics/texture.hh"
#include <cstdint>

#ifdef _WIN32
    #include <windows.h>
#endif

#include "tempest/graphics/opengl-backend/gl-library.hh"

namespace Tempest
{
union Vector4;
    
struct TextureInfo
{
    GLFormat InternalFormat;
    GLFormat Format;
    GLType   Type;
};

struct GLTextureBindInfo
{
    union
    {
        struct
        {
            GLTextureTarget target;
            uint32_t        handle;
        };
        GLuint64 handle64;
    };
};

#ifdef GL_R32F
 it_is_broken
#endif

inline TextureInfo TranslateTextureInfo(DataFormat fmt)
{
    switch(fmt)
    {
        //  case DataFormat::Unknown: break;
    case DataFormat::R32F: return TextureInfo{ GLFormat::GL_R32F, GLFormat::GL_RED, GLType::GL_FLOAT };
    case DataFormat::RG32F: return TextureInfo{ GLFormat::GL_RG32F, GLFormat::GL_RG, GLType::GL_FLOAT };
    case DataFormat::RGB32F: return TextureInfo{ GLFormat::GL_RGB32F, GLFormat::GL_RGB, GLType::GL_FLOAT };
    case DataFormat::RGBA32F: return TextureInfo{ GLFormat::GL_RGBA32F, GLFormat::GL_RGBA, GLType::GL_FLOAT };
    case DataFormat::R16F: return TextureInfo{ GLFormat::GL_R16F, GLFormat::GL_RED, GLType::GL_FLOAT };
    case DataFormat::RG16F: return TextureInfo{ GLFormat::GL_RG16F, GLFormat::GL_RG, GLType::GL_FLOAT };
        //  case DataFormat::RGB16F: break; 
    case DataFormat::RGBA16F: return TextureInfo{ GLFormat::GL_RGBA16F, GLFormat::GL_RGBA, GLType::GL_FLOAT };
        //  case DataFormat::R32: return TextureInfo{ GL_RED, GL_RED, GL_INT }; break;
        //  case DataFormat::RG32: return TextureInfo{ GL_RG, GL_RG, GL_INT }; break;
        //  case DataFormat::RGB32: return TextureInfo{ GL_RGB, GL_RGB, GL_INT }; break;
        //  case DataFormat::RGBA32: return TextureInfo{ GL_RGBA, GL_RGBA, GL_INT }; break;
    case DataFormat::R16SNorm: return TextureInfo{ GLFormat::GL_R16_SNORM, GLFormat::GL_RED, GLType::GL_SHORT };
    case DataFormat::RG16SNorm: return TextureInfo{ GLFormat::GL_RG16_SNORM, GLFormat::GL_RG, GLType::GL_SHORT };
        //  case DataFormat::RGB16SNorm: break;
    case DataFormat::RGBA16SNorm: return TextureInfo{ GLFormat::GL_RGBA16_SNORM, GLFormat::GL_RGBA, GLType::GL_SHORT };
    case DataFormat::R8SNorm: return TextureInfo{ GLFormat::GL_R8_SNORM, GLFormat::GL_RED, GLType::GL_BYTE };
    case DataFormat::RG8SNorm: return TextureInfo{ GLFormat::GL_RG8_SNORM, GLFormat::GL_RG, GLType::GL_BYTE };
        //  case DataFormat::RGB8SNorm: break;
    case DataFormat::RGBA8SNorm: return TextureInfo{ GLFormat::GL_RGBA8_SNORM, GLFormat::GL_RGBA, GLType::GL_BYTE };
        //  case DataFormat::R32: return TextureInfo{GL_RED, GL_RED, GL_UNSIGNED_INT }; break;
        //  case DataFormat::RG32: return TextureInfo{ GL_RG, GL_RG, GL_UNSIGNED_INT }; break;
        //  case DataFormat::RGB32: return TextureInfo{ GL_RGB, GL_RGB, GL_UNSIGNED_INT }; break;
        //  case DataFormat::RGBA32: return TextureInfo{ GL_RGBA, GL_RGBA, GL_UNSIGNED_INT }; break;
    case DataFormat::R16UNorm: return TextureInfo{ GLFormat::GL_R16, GLFormat::GL_RED, GLType::GL_UNSIGNED_SHORT };
    case DataFormat::RG16UNorm: return TextureInfo{ GLFormat::GL_RG16, GLFormat::GL_RG, GLType::GL_UNSIGNED_SHORT };
        //  case DataFormat::RGB16UNorm: return TextureInfo{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT }; break;
    case DataFormat::RGBA16UNorm: return TextureInfo{ GLFormat::GL_RGBA16, GLFormat::GL_RGBA, GLType::GL_UNSIGNED_SHORT };
    case DataFormat::R8UNorm: return TextureInfo{ GLFormat::GL_R8, GLFormat::GL_RED, GLType::GL_UNSIGNED_BYTE };
    case DataFormat::RG8UNorm: return TextureInfo{ GLFormat::GL_RG8, GLFormat::GL_RG, GLType::GL_UNSIGNED_BYTE };
        //  case DataFormat::RGB8UNorm: break;
    default: TGE_ASSERT(false, "Unsupported texture format."); // fall-through
    case DataFormat::RGBA8UNorm: return TextureInfo{ GLFormat::GL_RGBA8, GLFormat::GL_RGBA, GLType::GL_UNSIGNED_BYTE };
    case DataFormat::D16: return TextureInfo{ GLFormat::GL_DEPTH_COMPONENT16, GLFormat::GL_DEPTH_COMPONENT, GLType::GL_UNSIGNED_SHORT };
    case DataFormat::D24S8: return TextureInfo{ GLFormat::GL_DEPTH24_STENCIL8, GLFormat::GL_DEPTH_STENCIL, GLType::GL_UNSIGNED_INT_24_8 };
    case DataFormat::D32: return TextureInfo{ GLFormat::GL_DEPTH_COMPONENT32, GLFormat::GL_DEPTH_COMPONENT, GLType::GL_UNSIGNED_INT };
    case DataFormat::R10G10B10A2: return TextureInfo{ GLFormat::GL_RGBA, GLFormat::GL_RGBA, GLType::GL_INT_2_10_10_10_REV };
    case DataFormat::uR10G10B10A2: return TextureInfo{ GLFormat::GL_RGBA, GLFormat::GL_RGBA, GLType::GL_UNSIGNED_INT_10_10_10_2 };
    }
}

class GLTexture
{
    TextureDescription  m_Description;
    GLTextureTarget     m_Target;
    uint32_t            m_Flags;
    GLuint              m_Texture;
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    mutable GLuint64    m_GPUHandle;
#endif
public:
    explicit GLTexture(const TextureDescription& desc, uint32_t flags, const void* data);
     ~GLTexture();
    
    void setFilter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter, SamplerMode sampler_mode = SamplerMode::Default);
    void setWrapMode(WrapMode x_axis, WrapMode y_axis, WrapMode z_axis);
    void setMipLODBias(float lod_bias);
    void setMaxAnisotrophy(uint32_t max_anisotropy);
    void setCompareFunction(ComparisonFunction compare_func);
    void setBorderColor(const Vector4& color);
    void setMinLOD(float min_lod);
    void setMaxLOD(float max_lod);
     
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    //! \remarks Once you extract the handle state changes are going to be ignored.
    GLuint64 getGPUHandle() const;
#endif
    
    GLTextureTarget getTarget() const { return m_Target; }
    GLuint getCPUHandle() const { return m_Texture; }

    GLuint64 getHandle() const;

    const TextureDescription& getDescription() const { return m_Description; }
};

class GLRenderTarget: public GLTexture
{
public:
    GLRenderTarget(const TextureDescription& desc, uint32_t flags);
};

}

#endif // _GL_TEXTURE_HH_
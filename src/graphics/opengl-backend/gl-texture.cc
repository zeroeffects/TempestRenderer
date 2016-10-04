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

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/graphics/opengl-backend/gl-texture.hh"
#include "tempest/graphics/opengl-backend/gl-utils.hh"

namespace Tempest
{
GLTexture::GLTexture(const TextureDescription& desc, uint32_t flags, const void* data)
    :   m_Description(desc),
        m_Flags(flags)
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
        , m_GPUHandle(0)
#endif
{
    glGenTextures(1, &m_Texture);
    auto tex_info = TranslateTextureInfo(desc.Format);
    TGE_ASSERT(desc.Width && desc.Height && desc.Depth, "Texture should have all dimensions different than zero");
    if(desc.Depth > 1)
    {
        switch(desc.Tiling)
        {
        case TextureTiling::Array: m_Target = GLTextureTarget::GL_TEXTURE_2D_ARRAY; break;
        case TextureTiling::Volume: m_Target = GLTextureTarget::GL_TEXTURE_3D; break;
        case TextureTiling::Cube: m_Target = GLTextureTarget::GL_TEXTURE_CUBE_MAP_ARRAY; break;
        default: TGE_ASSERT(false, "Uexpected tiling mode"); break;
        }
        
        glBindTexture(m_Target, m_Texture);
        if(desc.Samples > 1)
        {
            TGE_ASSERT(desc.Tiling != TextureTiling::Cube, "Multisampling with cubemaps is unsupported");
            glTexImage3DMultisample(m_Target, desc.Samples, tex_info.InternalFormat, desc.Width, desc.Height, desc.Depth, GL_TRUE);
        }
        else
        {
            if(desc.Tiling == TextureTiling::Cube)
            {
                glTexImage3D(m_Target, 0, tex_info.InternalFormat, desc.Width, desc.Height, desc.Depth*6, 0, tex_info.Format, tex_info.Type, data);
            }
            else
            {
                glTexImage3D(m_Target, 0, tex_info.InternalFormat, desc.Width, desc.Height, desc.Depth, 0, tex_info.Format, tex_info.Type, data);
            }
        }
    }
    else if(desc.Height)
    {
        switch(desc.Tiling)
        {
        case TextureTiling::Array: m_Target = GLTextureTarget::GL_TEXTURE_1D_ARRAY; break;
        case TextureTiling::Flat: m_Target = GLTextureTarget::GL_TEXTURE_2D; break;
        case TextureTiling::Cube: m_Target = GLTextureTarget::GL_TEXTURE_CUBE_MAP; break;
        default: TGE_ASSERT(false, "Uexpected tiling mode"); break;
        }

        glBindTexture(m_Target, m_Texture);
        if(desc.Samples > 1)
        {
            TGE_ASSERT(desc.Tiling != TextureTiling::Cube, "Multisampling with cubemaps is unsupported");
            glTexImage2DMultisample(m_Target, desc.Samples, tex_info.InternalFormat, desc.Width, desc.Height, GL_TRUE);
            TGE_ASSERT(data == nullptr, "Uploading data on multisampled textures is unsupported");
        }
        else
        {
            if(desc.Tiling == TextureTiling::Cube)
            {
                const void *pos_x_map = nullptr,
                           *neg_x_map = nullptr,
                           *pos_y_map = nullptr,
                           *neg_y_map = nullptr,
                           *pos_z_map = nullptr,
                           *neg_z_map = nullptr;
                if(data)
                {
                    auto* data_u8 = reinterpret_cast<const uint8_t*>(data);
                    size_t depth_slice = desc.Width*desc.Height*DataFormatElementSize(desc.Format);
                    pos_x_map = data_u8;
                    neg_x_map = data_u8 + depth_slice;
                    pos_y_map = data_u8 + 2 * depth_slice;
                    neg_y_map = data_u8 + 3 * depth_slice;
                    pos_z_map = data_u8 + 4 * depth_slice;
                    neg_z_map = data_u8 + 5 * depth_slice;
                }

                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, pos_x_map);
                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, neg_x_map);
                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, pos_y_map);
                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, neg_y_map);
                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, pos_z_map);
                glTexImage2D(GLTextureTarget::GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, neg_z_map);
            }
            else
            {
                glTexImage2D(m_Target, 0, tex_info.InternalFormat, desc.Width, desc.Height, 0, tex_info.Format, tex_info.Type, data);
            }
        }
    }
    else
    {
        TGE_ASSERT(desc.Tiling == TextureTiling::Flat, "Unexpected tiling mode");
        m_Target = GLTextureTarget::GL_TEXTURE_1D;
        glBindTexture(m_Target, m_Texture);
        TGE_ASSERT(desc.Samples <= 1, "Multisampling is unsupported for 1D textures");
        glTexImage1D(m_Target, 0, tex_info.InternalFormat, desc.Width, 0, tex_info.Format, tex_info.Type, data);
    }

    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_SWIZZLE_R, GL_RED);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_SWIZZLE_B, GL_BLUE);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(m_Target, GLTextureParameter::GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    if(data && (flags & RESOURCE_GENERATE_MIPS))
    {
        glGenerateMipmap(GLTextureTarget::GL_TEXTURE_2D);
    }
    
    CheckOpenGL();
}

GLTexture::~GLTexture()
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(m_GPUHandle)
    {
        glMakeTextureHandleNonResidentARB(m_GPUHandle);
    }
#endif
    glDeleteTextures(1, &m_Texture);
}

void GLTexture::setFilter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter, SamplerMode sampler_mode)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    GLint gl_min_filter, gl_mag_filter;
    switch(mag_filter)
    {
    case FilterMode::Nearest: gl_mag_filter =  GL_NEAREST; break;
    case FilterMode::Linear: gl_mag_filter = GL_LINEAR; break;
    default: TGE_ASSERT(false, "Unsupported filter mode"); break;
    }
    
    switch((uint32_t)min_filter | ((uint32_t)mip_filter << 4))
    {
    case (uint32_t)FilterMode::Nearest | ((uint32_t)FilterMode::Nearest << 4): gl_min_filter = GL_NEAREST_MIPMAP_NEAREST; break;
    case (uint32_t)FilterMode::Nearest | ((uint32_t)FilterMode::Linear << 4): gl_min_filter = GL_NEAREST_MIPMAP_LINEAR; break;
    case (uint32_t)FilterMode::Linear  | ((uint32_t)FilterMode::Nearest << 4): gl_min_filter = GL_LINEAR_MIPMAP_NEAREST; break;
    case (uint32_t)FilterMode::Linear  | ((uint32_t)FilterMode::Linear << 4): gl_min_filter = GL_LINEAR_MIPMAP_LINEAR; break;
    }
    
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_MIN_FILTER, gl_min_filter);
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_MAG_FILTER, gl_mag_filter);
}

static GLint ConvertWrapMode(WrapMode mode)
{
    switch(mode)
    {
    default: TGE_ASSERT(false, "Unknown wrap mode");
    case WrapMode::Repeat: return GL_REPEAT;
    case WrapMode::Mirror: return GL_MIRRORED_REPEAT;
    case WrapMode::Clamp: return GL_CLAMP_TO_EDGE;
    case WrapMode::ClampBorder: return GL_CLAMP_TO_BORDER;
    case WrapMode::MirrorOnce: return GL_MIRROR_CLAMP_TO_EDGE;
    }
}

void GLTexture::setWrapMode(WrapMode x_axis, WrapMode y_axis, WrapMode z_axis)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_WRAP_S, ConvertWrapMode(x_axis));
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_WRAP_T, ConvertWrapMode(y_axis));
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_WRAP_R, ConvertWrapMode(z_axis));
}

void GLTexture::setMipLODBias(float lod_bias)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameterfEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_LOD_BIAS, lod_bias);
}

void GLTexture::setMaxAnisotrophy(uint32_t max_anisotropy)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_MAX_ANISOTROPY_EXT, max_anisotropy);
}

void GLTexture::setCompareFunction(ComparisonFunction compare_func)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    if(compare_func == ComparisonFunction::AlwaysPass)
    {
        glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_COMPARE_MODE, GL_NONE);
        glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_COMPARE_FUNC, static_cast<GLint>(GLComparisonFunction::GL_ALWAYS));
    }
    else
    {
        glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glTextureParameteriEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_COMPARE_FUNC, static_cast<GLint>(TranslateComparisonFunction(compare_func)));
    }
}

void GLTexture::setBorderColor(const Vector4& color)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameterfvEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_BORDER_COLOR, reinterpret_cast<const float*>(&color));
}

void GLTexture::setMinLOD(float min_lod)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameterfEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_MIN_LOD, min_lod);
}

void GLTexture::setMaxLOD(float max_lod)
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
#endif
    glTextureParameterfEXT(m_Texture, m_Target, GLTextureParameter::GL_TEXTURE_MAX_LOD, max_lod);
}

#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
GLuint64 GLTexture::getGPUHandle() const
{
    if(m_GPUHandle == 0)
    {
        m_GPUHandle = glGetTextureHandleARB(m_Texture);
        glMakeTextureHandleResidentARB(m_GPUHandle);
    }
    return m_GPUHandle;
}
#endif

GLuint64 GLTexture::getHandle() const
{
#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS))
    {
        if(m_GPUHandle == 0)
        {
            m_GPUHandle = glGetTextureHandleARB(m_Texture);
            glMakeTextureHandleResidentARB(m_GPUHandle);
        }
        return m_GPUHandle;
    }
    else
#endif
    {
        GLTextureBindInfo bind_info;
        bind_info.target = m_Target;
        bind_info.handle = m_Texture;
        return bind_info.handle64;
    }
}

// Render target
GLRenderTarget::GLRenderTarget(const TextureDescription& desc, uint32_t flags)
    :   GLTexture(desc, flags, nullptr)
{

}
}
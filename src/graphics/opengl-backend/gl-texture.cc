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
GLTexture::GLTexture(const TextureDescription& desc, uint32 flags, const void* data)
    :   m_Description(desc),
        m_Flags(flags),
        m_GPUHandle(0)
{
    struct TexInfo
    {
        GLint   internalFormat;
        GLenum  format,
                type;
    } tex_info;
    
    switch(desc.Format)
    {
//  case DataFormat::Unknown: break;
    case DataFormat::R32F: tex_info = TexInfo{ GL_R32F, GL_RED, GL_FLOAT }; break;
    case DataFormat::RG32F: tex_info = TexInfo{ GL_RG32F, GL_RG, GL_FLOAT }; break;
    case DataFormat::RGB32F: tex_info = TexInfo{ GL_RGB32F, GL_RGB, GL_FLOAT }; break;
    case DataFormat::RGBA32F: tex_info = TexInfo{ GL_RGBA32F, GL_RGBA, GL_FLOAT }; break;
    case DataFormat::R16F: tex_info = TexInfo{ GL_R16F, GL_RED, GL_FLOAT }; break;
    case DataFormat::RG16F: tex_info = TexInfo{ GL_RG16F, GL_RG, GL_FLOAT }; break;
//  case DataFormat::RGB16F: break; 
    case DataFormat::RGBA16F: tex_info = TexInfo{ GL_RGBA16F, GL_RGBA, GL_FLOAT }; break;
//  case DataFormat::R32: tex_info = TexInfo{ GL_RED, GL_RED, GL_INT }; break;
//  case DataFormat::RG32: tex_info = TexInfo{ GL_RG, GL_RG, GL_INT }; break;
//  case DataFormat::RGB32: tex_info = TexInfo{ GL_RGB, GL_RGB, GL_INT }; break;
//  case DataFormat::RGBA32: tex_info = TexInfo{ GL_RGBA, GL_RGBA, GL_INT }; break;
    case DataFormat::R16SNorm: tex_info = TexInfo{ GL_R16_SNORM, GL_RED, GL_SHORT }; break;
    case DataFormat::RG16SNorm: tex_info = TexInfo{ GL_RG16_SNORM, GL_RG, GL_SHORT }; break;
//  case DataFormat::RGB16SNorm: break;
    case DataFormat::RGBA16SNorm: tex_info = TexInfo{ GL_RGBA16_SNORM, GL_RGBA, GL_SHORT }; break;
    case DataFormat::R8SNorm: tex_info = TexInfo{ GL_R8_SNORM, GL_RED, GL_BYTE }; break;
    case DataFormat::RG8SNorm: tex_info = TexInfo{ GL_RG8_SNORM, GL_RG, GL_BYTE }; break;
//  case DataFormat::RGB8SNorm: break;
    case DataFormat::RGBA8SNorm: tex_info = TexInfo{ GL_RGBA8_SNORM, GL_RGBA, GL_BYTE }; break;
//  case DataFormat::R32: tex_info = TexInfo{GL_RED, GL_RED, GL_UNSIGNED_INT }; break;
//  case DataFormat::RG32: tex_info = TexInfo{ GL_RG, GL_RG, GL_UNSIGNED_INT }; break;
//  case DataFormat::RGB32: tex_info = TexInfo{ GL_RGB, GL_RGB, GL_UNSIGNED_INT }; break;
//  case DataFormat::RGBA32: tex_info = TexInfo{ GL_RGBA, GL_RGBA, GL_UNSIGNED_INT }; break;
    case DataFormat::R16UNorm: tex_info = TexInfo{ GL_R16, GL_RED, GL_UNSIGNED_SHORT }; break;
    case DataFormat::RG16UNorm: tex_info = TexInfo{ GL_RG16, GL_RG, GL_UNSIGNED_SHORT }; break;
//  case DataFormat::RGB16UNorm: tex_info = TexInfo{ GL_RGB, GL_RGB, GL_UNSIGNED_SHORT }; break;
    case DataFormat::RGBA16UNorm: tex_info = TexInfo{ GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT }; break;
    case DataFormat::R8UNorm: tex_info = TexInfo{ GL_R8, GL_RED, GL_UNSIGNED_BYTE }; break;
    case DataFormat::RG8UNorm: tex_info = TexInfo{ GL_RG8, GL_RG, GL_UNSIGNED_BYTE }; break;
//  case DataFormat::RGB8UNorm: break;
    case DataFormat::RGBA8UNorm: tex_info = TexInfo{ GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE }; break;
    case DataFormat::D16: tex_info = TexInfo{ GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT }; break;
    case DataFormat::D24S8: tex_info = TexInfo{ GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8 }; break;
    case DataFormat::D32: tex_info = TexInfo{ GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT }; break;
    case DataFormat::R10G10B10A2: tex_info = TexInfo{ GL_RGBA, GL_RGBA, GL_INT_2_10_10_10_REV }; break;
    case DataFormat::uR10G10B10A2: tex_info = TexInfo{ GL_RGBA, GL_RGBA, GL_UNSIGNED_INT_10_10_10_2 }; break;
    default: TGE_ASSERT(false, "Unsupported texture format."); break;
    }
    
    glGenTextures(1, &m_Texture);
    TGE_ASSERT(desc.Width && desc.Height && desc.Depth, "Texture should have all dimensions different than zero");
    if(desc.Tiling == TextureTiling::Cube)
    {
        TGE_ASSERT(desc.Depth == 6, "Unexpected number of faces for cube map. Should be 6.");
        TGE_ASSERT(false, "TODO: Unimplemented"); // TODO: cube mapping
        //m_Target = GL_TEXTURE_CUBE_MAP;
        //glBindTexture(m_Target, m_TexId);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data + width*height*tex_info.size);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data + 2*width*height*tex_info.size);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data + 3*width*height*tex_info.size);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data + 4*width*height*tex_info.size);
        //glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, tex_info.internalFormat, width, height, 0, tex_info.format, tex_info.type, data + 5*width*height*tex_info.size);
    }
    else if(desc.Depth > 1)
    {
        switch(desc.Tiling)
        {
        case TextureTiling::Array: m_Target = GL_TEXTURE_2D_ARRAY; break;
        case TextureTiling::Volume: m_Target = GL_TEXTURE_3D; break;
        default: TGE_ASSERT(false, "Uexpected tiling mode"); break;
        }
        
        glBindTexture(m_Target, m_Texture);
        if(desc.Samples > 1)
        {
            glTexImage3DMultisample(m_Target, desc.Samples, tex_info.internalFormat, desc.Width, desc.Height, desc.Depth, GL_TRUE);
        }
        else
        {
            glTexImage3D(m_Target, 0, tex_info.internalFormat, desc.Width, desc.Height, desc.Depth, 0, tex_info.format, tex_info.type, data);
        }
    }
    else if(desc.Height)
    {
        switch(desc.Tiling)
        {
        case TextureTiling::Array: m_Target = GL_TEXTURE_1D_ARRAY; break;
        case TextureTiling::Flat: m_Target = GL_TEXTURE_2D; break;
        default: TGE_ASSERT(false, "Uexpected tiling mode"); break;
        }

        glBindTexture(m_Target, m_Texture);
        if(desc.Samples > 1)
        {
            glTexImage2DMultisample(m_Target, desc.Samples, tex_info.internalFormat, desc.Width, desc.Height, GL_TRUE);
            TGE_ASSERT(data == nullptr, "Uploading data on multisampled textures is unsupported");
        }
        else
        {
            glTexImage2D(m_Target, 0, tex_info.internalFormat, desc.Width, desc.Height, 0, tex_info.format, tex_info.type, data);
        }
    }
    else
    {
        TGE_ASSERT(desc.Tiling == TextureTiling::Flat, "Unexpected tiling mode");
        m_Target = GL_TEXTURE_1D;
        glBindTexture(m_Target, m_Texture);
        TGE_ASSERT(desc.Samples <= 1, "Multisampling is unsupported for 1D textures");
        glTexImage1D(m_Target, 0, tex_info.internalFormat, desc.Width, 0, tex_info.format, tex_info.type, data);
    }

    glTexParameteri(m_Target, GL_TEXTURE_SWIZZLE_R, GL_RED);
    glTexParameteri(m_Target, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    glTexParameteri(m_Target, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
    glTexParameteri(m_Target, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
    glTexParameteri(m_Target, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(m_Target, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(m_Target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(m_Target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    if(data && (flags & RESOURCE_GENERATE_MIPS))
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    
    CheckOpenGL();
}

GLTexture::~GLTexture()
{
    if(m_GPUHandle)
    {
        glMakeTextureHandleNonResidentARB(m_GPUHandle);
    }
    glDeleteTextures(1, &m_Texture);
}

void GLTexture::setFilter(FilterMode min_filter, FilterMode mag_filter, FilterMode mip_filter, SamplerMode sampler_mode)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    GLenum gl_min_filter, gl_mag_filter;
    switch(mag_filter)
    {
    case FilterMode::Nearest: gl_mag_filter =  GL_NEAREST; break;
    case FilterMode::Linear: gl_mag_filter = GL_LINEAR; break;
    default: TGE_ASSERT(false, "Unsupported filter mode"); break;
    }
    
    switch((uint32)min_filter | ((uint32)mip_filter << 4))
    {
    case (uint32)FilterMode::Nearest | ((uint32)FilterMode::Nearest << 4): gl_min_filter = GL_NEAREST_MIPMAP_NEAREST; break;
    case (uint32)FilterMode::Nearest | ((uint32)FilterMode::Linear << 4): gl_min_filter = GL_NEAREST_MIPMAP_LINEAR; break;
    case (uint32)FilterMode::Linear  | ((uint32)FilterMode::Nearest << 4): gl_min_filter = GL_LINEAR_MIPMAP_NEAREST; break;
    case (uint32)FilterMode::Linear  | ((uint32)FilterMode::Linear << 4): gl_min_filter = GL_LINEAR_MIPMAP_LINEAR; break;
    }
    
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_MIN_FILTER, gl_min_filter);
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_MAG_FILTER, gl_mag_filter);
}

static GLenum ConvertWrapMode(WrapMode mode)
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
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_WRAP_S, ConvertWrapMode(x_axis));
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_WRAP_T, ConvertWrapMode(y_axis));
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_WRAP_R, ConvertWrapMode(z_axis));
}

void GLTexture::setMipLODBias(float lod_bias)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameterfEXT(m_Texture, m_Target, GL_TEXTURE_LOD_BIAS, lod_bias);
}

void GLTexture::setMaxAnisotrophy(uint32 max_anisotropy)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_MAX_ANISOTROPY_EXT, max_anisotropy);
}

void GLTexture::setCompareFunction(ComparisonFunction compare_func)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    if(compare_func == ComparisonFunction::AlwaysPass)
    {
        glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_COMPARE_MODE, GL_NONE);
        glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_COMPARE_FUNC, GL_ALWAYS);
    }
    else
    {
        glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        GLenum gl_cmp_func;
        switch(compare_func)
        {
        case ComparisonFunction::Never: gl_cmp_func = GL_NEVER; break;
        case ComparisonFunction::Less: gl_cmp_func = GL_LESS; break;
        case ComparisonFunction::Equal: gl_cmp_func = GL_EQUAL; break;
        case ComparisonFunction::LessEqual: gl_cmp_func = GL_LEQUAL; break;
        case ComparisonFunction::Greater: gl_cmp_func = GL_GREATER; break;
        case ComparisonFunction::NotEqual: gl_cmp_func = GL_NOTEQUAL; break;
        case ComparisonFunction::GreaterEqual: gl_cmp_func = GL_GEQUAL; break;
        case ComparisonFunction::AlwaysPass: gl_cmp_func = GL_ALWAYS; break;
        }
        glTextureParameteriEXT(m_Texture, m_Target, GL_TEXTURE_COMPARE_FUNC, gl_cmp_func);
    }
}

void GLTexture::setBorderColor(const Vector4& color)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameterfvEXT(m_Texture, m_Target, GL_TEXTURE_BORDER_COLOR, reinterpret_cast<const float*>(&color));
}

void GLTexture::setMinLOD(float min_lod)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameterfEXT(m_Texture, m_Target, GL_TEXTURE_MIN_LOD, min_lod);
}

void GLTexture::setMaxLOD(float max_lod)
{
    TGE_ASSERT(m_GPUHandle == 0, "Texture sampler state changes won't be done because this texture is already in use by resource table.");
    glTextureParameterfEXT(m_Texture, m_Target, GL_TEXTURE_MAX_LOD, max_lod);
}

GLuint64 GLTexture::getHandle() const
{
    if(m_GPUHandle == 0)
    {
        m_GPUHandle = glGetTextureHandleARB(m_Texture);
        glMakeTextureHandleResidentARB(m_GPUHandle);
    }
    return m_GPUHandle;
}
}
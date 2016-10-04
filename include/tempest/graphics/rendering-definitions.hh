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

#ifndef TEMPEST_RENDERING_DEFS_HH
#define TEMPEST_RENDERING_DEFS_HH

#include "tempest/graphics/opengl-backend/gl-library.hh"
#include "tempest/utils/assert.hh"
#include "tempest/compute/compute-macros.hh"
#include "tempest/math/vector4.hh"

#include <cstdint>
#include <string>

namespace Tempest
{
#ifndef MAX_VERTEX_BUFFERS
// Ok, you can have two. More is just going to ruin your performance
#   define MAX_VERTEX_BUFFERS 2
#endif

//! Enlists all types of render attachments. 
enum class RenderTargetBindPoint
{
    DepthStencilBuffer,
    DepthBuffer,
    StencilBuffer,
    ColorAttachment0,
    ColorAttachment1,
    ColorAttachment2,
    ColorAttachment3,
    LastAttachment = ColorAttachment3 //!< More than this is not worth the bandwidth.
};

enum class FilterMode
{
    Nearest,
    Linear
};

enum class SamplerMode
{
    Default,
    Minimum,
    Maximum,
    Comparison
};

enum class WrapMode
{
    Repeat,
    Mirror,
    Clamp,
    ClampBorder,
    MirrorOnce
};

enum class ComparisonFunction
{
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    AlwaysPass
};

enum ResourceFlags
{
    RESOURCE_STATIC_DRAW   = 0,
    RESOURCE_STATIC_READ   = 1,
    RESOURCE_STATIC_COPY   = 2,
    RESOURCE_STREAM_DRAW   = 3,
    RESOURCE_STREAM_READ   = 4,
    RESOURCE_STREAM_COPY   = 5,
    RESOURCE_DYNAMIC_DRAW  = 6,
    RESOURCE_DYNAMIC_READ  = 7,
    RESOURCE_DYNAMIC_COPY  = 8,
    RESOURCE_USAGE_MASK    = 0xF,
    RESOURCE_GENERATE_MIPS = 1 << 4
};

enum class ResourceBufferType
{
    ConstantBuffer,
    VertexBuffer,
    IndexBuffer
};

enum class DrawModes: uint16_t
{
    Unknown,
    PointList,
    LineStrip,
    LineList,
    TriangleStrip,
    TriangleList,
    LineStripAdjacency,
    LineListAdjacency,
    TriangleStripAdjacency,
    TriangleListAdjacency
};

enum class DataFormat
{
    Unknown,
    R32F,
    RG32F,
    RGB32F,
    RGBA32F,
    R16F,
    RG16F,
//  RGB16F,
    RGBA16F,
    R32,
    RG32,
    RGB32,
    RGBA32,
    R16,
    RG16,
//  RGB16,
    RGBA16,
    R8,
    RG8,
//  RGB8,
    RGBA8,
    uR32,
    uRG32,
    uRGB32,
    uRGBA32,
    uR16,
    uRG16,
//  uRGB16,
    uRGBA16,
    uR8,
    uRG8,
//  uRGB8,
    uRGBA8,
    
    R16SNorm,
    RG16SNorm,
//  RGB16SNorm,
    RGBA16SNorm,
    R8SNorm,
    RG8SNorm,
//  RGB8SNorm,
    RGBA8SNorm,
    R16UNorm,
    RG16UNorm,
//  RGB16UNorm,
    RGBA16UNorm,
    R8UNorm,
    RG8UNorm,
//  RGB8UNorm,
    RGBA8UNorm,
    BGRA8UNorm,

    D16,
    D24S8,
    D32,

    R10G10B10A2,
    uR10G10B10A2,
    Count
};

inline bool HasDepth(DataFormat fmt)
{
    switch(fmt)
    {
    case DataFormat::D16:
    case DataFormat::D24S8:
    case DataFormat::D32:
        return true;
    }
    return false;
}

inline bool HasStencil(DataFormat fmt)
{
    return fmt == DataFormat::D24S8;
}

DataFormat TranslateDataFormat(const std::string& str);

inline EXPORT_CUDA uint32_t DataFormatChannels(DataFormat format)
{
    switch(format)
    {
    case DataFormat::R32F:
    case DataFormat::R16F:
    case DataFormat::R32:
    case DataFormat::R16:
    case DataFormat::R8:
    case DataFormat::uR8:
    case DataFormat::uR16:
    case DataFormat::R16SNorm:
    case DataFormat::D16:
    case DataFormat::D32:
    case DataFormat::R8UNorm:
    case DataFormat::R16UNorm:
    case DataFormat::R8SNorm:
    case DataFormat::uR32:
        return 1;
    case DataFormat::RG32F:
    case DataFormat::RG16F:
    case DataFormat::RG32:
    case DataFormat::RG16:
    case DataFormat::RG8:
    case DataFormat::uRG32:
    case DataFormat::uRG16:
    case DataFormat::uRG8:
    case DataFormat::RG16UNorm:
    case DataFormat::RG8SNorm:
    case DataFormat::RG16SNorm:
    case DataFormat::RG8UNorm:
    case DataFormat::D24S8:
    //  RGB8,
        return 2;
    case DataFormat::RGB32F:
    //  RGB16F,
    case DataFormat::RGB32:
    //  RGB16,
    case DataFormat::uRGB32:
    //  uRGB16,
    //  uRGB8,
    //  RGB16SNorm,
    //  RGB8SNorm,
    //  RGB16UNorm,
    //  RGB8UNorm,
        return 3;
    case DataFormat::RGBA8:
    case DataFormat::RGBA32F:
    case DataFormat::RGBA16F:
    case DataFormat::RGBA32:
    case DataFormat::RGBA16:
    case DataFormat::uRGBA32:
    case DataFormat::uRGBA16:
    case DataFormat::uRGBA8:
    case DataFormat::RGBA16SNorm:
    case DataFormat::RGBA8SNorm:
    case DataFormat::RGBA16UNorm:
    case DataFormat::RGBA8UNorm:
    case DataFormat::R10G10B10A2:
    case DataFormat::uR10G10B10A2:
        return 4;
    default: TGE_ASSERT(false, "Unknown format");
    }
    return 0;
}

inline EXPORT_CUDA uint32_t DataFormatElementSize(DataFormat format)
{
    switch(format)
    {
    default: TGE_ASSERT(false, "Unknown format"); return 0;
    case DataFormat::R32F: return sizeof(float);
    case DataFormat::RG32F: return 2*sizeof(float);
    case DataFormat::RGB32F: return 3*sizeof(float);
    case DataFormat::RGBA32F: return 4*sizeof(float);
    case DataFormat::R16F: return sizeof(float)/2;
    case DataFormat::RG16F: return 2*sizeof(float)/2;
    //  RGB16F,
    case DataFormat::RGBA16F: return 4*sizeof(float)/2;
    case DataFormat::R32: return sizeof(int32_t);
    case DataFormat::RG32: return 2*sizeof(int32_t);
    case DataFormat::RGB32: return 3*sizeof(int32_t);
    case DataFormat::RGBA32: return 4*sizeof(int32_t);
    case DataFormat::R16: return sizeof(int16_t);
    case DataFormat::RG16: return 2*sizeof(int16_t);
    //  RGB16,
    case DataFormat::RGBA16: return 4*sizeof(int16_t);
    case DataFormat::R8: return sizeof(int8_t);
    case DataFormat::RG8: return 2*sizeof(int8_t);
    //  RGB8,
    case DataFormat::RGBA8: return 4*sizeof(int8_t);
    case DataFormat::uR32: return sizeof(uint32_t);
    case DataFormat::uRG32: return 2*sizeof(uint32_t);
    case DataFormat::uRGB32: return 3*sizeof(uint32_t);
    case DataFormat::uRGBA32: return 4*sizeof(uint32_t);
    case DataFormat::uR16: return sizeof(uint16_t);
    case DataFormat::uRG16: return 2*sizeof(uint16_t);
    //  uRGB16,
    case DataFormat::uRGBA16: return 4*sizeof(uint16_t);
    case DataFormat::uR8: return sizeof(uint8_t);
    case DataFormat::uRG8: return 2*sizeof(uint8_t);
    //  uRGB8,
    case DataFormat::uRGBA8: return 4*sizeof(uint8_t);
    case DataFormat::R16SNorm: return sizeof(int16_t);
    case DataFormat::RG16SNorm: return 2*sizeof(int16_t);
    //  RGB16SNorm,
    case DataFormat::RGBA16SNorm: return 4*sizeof(int16_t);
    case DataFormat::R8SNorm: return sizeof(int8_t);
    case DataFormat::RG8SNorm: return 2*sizeof(int8_t);
    //  RGB8SNorm,
    case DataFormat::RGBA8SNorm: return 4*sizeof(int8_t);
    case DataFormat::R16UNorm: return sizeof(uint16_t);
    case DataFormat::RG16UNorm: return 2*sizeof(uint16_t);
    //  RGB16UNorm,
    case DataFormat::RGBA16UNorm: return 4*sizeof(uint16_t);
    case DataFormat::R8UNorm: return sizeof(uint8_t);
    case DataFormat::RG8UNorm: return 2*sizeof(uint8_t);
    //  RGB8UNorm,
    case DataFormat::RGBA8UNorm: return 4*sizeof(uint8_t);
    case DataFormat::D16: return sizeof(uint16_t);
    case DataFormat::D24S8: return (24 + 8)/8;
    case DataFormat::D32: return sizeof(uint32_t);
    case DataFormat::R10G10B10A2: return (10 + 10 + 10 + 2)/8;
    case DataFormat::uR10G10B10A2:  return (10 + 10 + 10 + 2)/8;
    }
}

enum class UniformValueType: uint32_t
{
    Float,
    Vector2,
    Vector3,
    Vector4,
    Integer,
    IntegerVector2,
    IntegerVector3,
    IntegerVector4,
    UnsignedInteger,
    UnsignedIntegerVector2,
    UnsignedIntegerVector3,
    UnsignedIntegerVector4,
    Boolean,
    BooleanVector2,
    BooleanVector3,
    BooleanVector4,
    Matrix2,
    Matrix3,
    Matrix4,
    Matrix2x3,
    Matrix2x4,
    Matrix3x2,
    Matrix3x4,
    Matrix4x2,
    Matrix4x3,
    Texture,
    Struct,
};

enum class StorageMode: uint32_t
{
    BufferRead,
    BufferWrite,
    PixelPack,
    PixelUnpack
};

inline EXPORT_CUDA uint32_t UniformValueTypeSize(UniformValueType uniform_value, bool tex_ptr64 = true)
{
    switch(uniform_value)
    {
    case UniformValueType::Float: return sizeof(float);
    case UniformValueType::Vector2: return 2*sizeof(float);
    case UniformValueType::Vector3: return 3*sizeof(float);
    case UniformValueType::Vector4: return 4*sizeof(float);
    case UniformValueType::Integer: return sizeof(int32_t);
    case UniformValueType::IntegerVector2: return 2*sizeof(int32_t);
    case UniformValueType::IntegerVector3: return 3*sizeof(int32_t);
    case UniformValueType::IntegerVector4: return 4*sizeof(int32_t);
    case UniformValueType::UnsignedInteger: return sizeof(uint32_t);
    case UniformValueType::UnsignedIntegerVector2: return 2*sizeof(uint32_t);
    case UniformValueType::UnsignedIntegerVector3: return 3*sizeof(uint32_t);
    case UniformValueType::UnsignedIntegerVector4: return 4*sizeof(uint32_t);
    case UniformValueType::Boolean: return sizeof(int32_t);
    case UniformValueType::BooleanVector2: return 2*sizeof(int32_t);
    case UniformValueType::BooleanVector3: return 3*sizeof(int32_t);
    case UniformValueType::BooleanVector4: return 4*sizeof(int32_t);
    case UniformValueType::Matrix2: return 2*2*sizeof(float);
    case UniformValueType::Matrix3: return 3*3*sizeof(float);
    case UniformValueType::Matrix4: return 4*4*sizeof(float);
    case UniformValueType::Matrix2x3: return 2*3*sizeof(float);
    case UniformValueType::Matrix2x4: return 2*4*sizeof(float);
    case UniformValueType::Matrix3x2: return 3*2*sizeof(float);
    case UniformValueType::Matrix3x4: return 3*4*sizeof(float);
    case UniformValueType::Matrix4x2: return 4*2*sizeof(float);
    case UniformValueType::Matrix4x3: return 4*3*sizeof(float);
    case UniformValueType::Texture:
    {
    #ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
        if(IsGLCapabilitySupported(TEMPEST_GL_CAPS_TEXTURE_BINDLESS) && tex_ptr64)
        {
            return sizeof(uint64_t);
        }
        else
    #endif
        {
            return sizeof(uint32_t);
        }
    }
    case UniformValueType::Struct: return 1;
    default: TGE_ASSERT(false, "Unsupported uniform value type"); break;
    }
    return 0;
}

enum
{
    TEMPEST_SETTING_DISABLE_SSBO             = 1 << 0,
    TEMPEST_SETTING_DISABLE_MULTI_DRAW       = 1 << 1,
    TEMPEST_SETTING_DISABLE_TEXTURE_BINDLESS = 1 << 2,
};

struct DepthStencilClearValue
{
    float    Depth;
    uint32_t Stencil;
};

union ClearValue
{
    Vector4                Color;
    DepthStencilClearValue DepthStencil;
};

struct CommandBufferDescription
{
    uint32_t CommandCount;
    uint32_t ConstantsBufferSize;
};

struct IOCommandBufferDescription
{
    uint32_t CommandCount;
};

enum class InputRateType: uint32_t
{
    PerVertex,
    PerInstance
};

struct BufferBinding
{
    uint32_t      BindPoint;
    InputRateType InputRate;
    uint32_t      Stride;
};

enum ClearFlags
{
    CLEAR_COLOR_BIT = 1 << 0,
    CLEAR_DEPTH_BIT = 1 << 1
};

struct ViewportBox
{
    float X = 0.0f,
          Y = 0.0f,
          Width,
          Height,
          MinDepth = -1.0f,
          MaxDepth = 1.0f;
};

struct ScissorRect
{
    uint32_t X = 0,
             Y = 0,
             Width,
             Height;
};

struct DynamicState
{
    ViewportBox Viewport;
    ScissorRect Scissor;
};

struct BindPointLayout
{
    uint32_t TextureCount = 0;
};
}

#endif // TEMPEST_RENDERING_DEFS_HH
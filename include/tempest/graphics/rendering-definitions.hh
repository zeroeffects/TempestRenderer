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

#include "tempest/utils/types.hh"

namespace Tempest
{
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

enum class VBUsage
{
    StaticDraw,
    StaticRead,
    StaticCopy,
    StreamDraw,
    StreamRead,
    StreamCopy,
    DynamicDraw,
    DynamicRead,
    DynamicCopy
};

enum class VBType
{
    VertexBuffer,
    IndexBuffer
};

enum class DrawModes: uint16
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

    D16,
    D24S8,
    D32,

    R10G10B10A2,
    uR10G10B10A2
};

enum class UniformValueType
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
    Texture
};

struct VertexAttributeDescription
{
    int         BufferId;
    string      Name; // For annoying validation purposes.
    DataFormat  Format;
    int         Stride;
    int         Offset;
};
}

#endif // TEMPEST_RENDERING_DEFS_HH
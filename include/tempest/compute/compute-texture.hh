/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2016 Zdravko Velinov
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

#ifndef _TEMPEST_COMPUTE_TEXTURE_HH_
#define _TEMPEST_COMPUTE_TEXTURE_HH_

#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/graphics/texture.hh"

#include <driver_types.h>
#include <texture_types.h>

namespace Tempest
{
enum
{
    TEMPEST_CUDA_TEXTURE_GATHER = 1 << 0,
    TEMPEST_CUDA_TEXTURE_SRGB = 1 << 1,
};

class Texture;
class CubeMap;

inline cudaTextureFilterMode TranslateFilterMode(TextureSampling sampling_mode)
{
    switch(sampling_mode)
    {
    default: TGE_ASSERT(false, "Unsupported mode");
    case TextureSampling::Bilinear: return cudaFilterModeLinear;
    case TextureSampling::Nearest: return cudaFilterModePoint;
    }
}

inline cudaChannelFormatDesc DataFormatToCuda(DataFormat fmt)
{
    switch(fmt)
    {
    case DataFormat::R32F: return cudaChannelFormatDesc{ 32, 0, 0, 0, cudaChannelFormatKindFloat };
    case DataFormat::RG32F: return cudaChannelFormatDesc{ 32, 32, 0, 0, cudaChannelFormatKindFloat };
    case DataFormat::RGB32F: return cudaChannelFormatDesc{ 32, 32, 32, 0, cudaChannelFormatKindFloat };
    case DataFormat::RGBA32F: return cudaChannelFormatDesc{ 32, 32, 32, 32, cudaChannelFormatKindFloat };
    case DataFormat::R16F: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindFloat };
    case DataFormat::RG16F: return cudaChannelFormatDesc{ 16, 16, 0, 0, cudaChannelFormatKindFloat };
//  case DataFormat::RGB16F:
    case DataFormat::RGBA16F: return cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindFloat };
    case DataFormat::R32: return cudaChannelFormatDesc{ 32, 0, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RG32: return cudaChannelFormatDesc{ 32, 32, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RGB32: return cudaChannelFormatDesc{ 32, 32, 32, 0, cudaChannelFormatKindSigned };
    case DataFormat::RGBA32: return cudaChannelFormatDesc{ 32, 32, 32, 32, cudaChannelFormatKindSigned };
    case DataFormat::R16: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RG16: return cudaChannelFormatDesc{ 16, 16, 0, 0, cudaChannelFormatKindSigned };
//  case DataFormat::RGB16:
    case DataFormat::RGBA16: return cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindSigned };
    case DataFormat::R8: return cudaChannelFormatDesc{ 8, 0, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RG8: return cudaChannelFormatDesc{ 8, 8, 0, 0, cudaChannelFormatKindSigned };
//  case DataFormat::RGB8:
    case DataFormat::RGBA8: return cudaChannelFormatDesc{ 8, 8, 8, 8, cudaChannelFormatKindSigned };
    case DataFormat::uR32: return cudaChannelFormatDesc{ 32, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::uRG32: return cudaChannelFormatDesc{ 32, 32, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::uRGB32: return cudaChannelFormatDesc{ 32, 32, 32, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::uRGBA32: return cudaChannelFormatDesc{ 32, 32, 32, 32, cudaChannelFormatKindUnsigned };
    case DataFormat::uR16: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::uRG16: return cudaChannelFormatDesc{ 16, 16, 0, 0, cudaChannelFormatKindUnsigned };
//  case DataFormat::uRGB16:
    case DataFormat::uRGBA16: return cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindUnsigned };
    case DataFormat::uR8: return cudaChannelFormatDesc{ 8, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::uRG8: return cudaChannelFormatDesc{ 8, 8, 0, 0, cudaChannelFormatKindUnsigned };
//  case DataFormat::uRGB8:
    case DataFormat::uRGBA8: return cudaChannelFormatDesc{ 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
    case DataFormat::R16SNorm: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RG16SNorm: return cudaChannelFormatDesc{ 16, 16, 0, 0, cudaChannelFormatKindSigned };
//  case DataFormat::RGB16SNorm:
    case DataFormat::RGBA16SNorm: return cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindSigned };
    case DataFormat::R8SNorm: return cudaChannelFormatDesc{ 8, 0, 0, 0, cudaChannelFormatKindSigned };
    case DataFormat::RG8SNorm: return cudaChannelFormatDesc{ 8, 8, 0, 0, cudaChannelFormatKindSigned };
//  case DataFormat::RGB8SNorm:
    case DataFormat::RGBA8SNorm: return cudaChannelFormatDesc{ 8, 8, 8, 8, cudaChannelFormatKindSigned };
    case DataFormat::R16UNorm: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::RG16UNorm: return cudaChannelFormatDesc{ 16, 16, 0, 0, cudaChannelFormatKindUnsigned };
//  case DataFormat::RGB16UNorm:
    case DataFormat::RGBA16UNorm: return cudaChannelFormatDesc{ 16, 16, 16, 16, cudaChannelFormatKindUnsigned };
    case DataFormat::R8UNorm: return cudaChannelFormatDesc{ 8, 0, 0, 0, cudaChannelFormatKindUnsigned };
    case DataFormat::RG8UNorm: return cudaChannelFormatDesc{ 8, 8, 0, 0, cudaChannelFormatKindUnsigned };
//  case DataFormat::RGB8UNorm:
    case DataFormat::RGBA8UNorm: return cudaChannelFormatDesc{ 8, 8, 8, 8, cudaChannelFormatKindUnsigned };
    case DataFormat::D16: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindFloat };
    case DataFormat::D24S8: return cudaChannelFormatDesc{ 24, 8, 0, 0, cudaChannelFormatKindNone }; // TODO: No idea whether this is actually supported
    case DataFormat::D32: return cudaChannelFormatDesc{ 16, 0, 0, 0, cudaChannelFormatKindFloat };
    case DataFormat::R10G10B10A2: return cudaChannelFormatDesc{ 10, 10, 10, 2, cudaChannelFormatKindSigned };
    case DataFormat::uR10G10B10A2: return cudaChannelFormatDesc{ 10, 10, 10, 2, cudaChannelFormatKindUnsigned };
    default: TGE_ASSERT(false, "Unsupported format");
    }
    return {};
}

inline cudaResourceViewFormat DataFormatToResourceViewFormat(DataFormat fmt)
{
    switch(fmt)
    {
    case DataFormat::R32F: return cudaResViewFormatFloat1;
    case DataFormat::RG32F: return cudaResViewFormatFloat2;
    //case DataFormat::RGB32F: return ;
    case DataFormat::RGBA32F: return cudaResViewFormatFloat4;
    case DataFormat::R16F: return cudaResViewFormatHalf1;
    case DataFormat::RG16F: return cudaResViewFormatHalf2;
//  case DataFormat::RGB16F:
    case DataFormat::RGBA16F: return cudaResViewFormatHalf4;
    case DataFormat::R32: return cudaResViewFormatSignedInt1;
    case DataFormat::RG32: return cudaResViewFormatSignedInt2;
//  case DataFormat::RGB32:
    case DataFormat::RGBA32: return cudaResViewFormatSignedInt4;
    case DataFormat::R16: return cudaResViewFormatSignedShort1;
    case DataFormat::RG16: return cudaResViewFormatSignedShort2;
//  case DataFormat::RGB16:
    case DataFormat::RGBA16: return cudaResViewFormatSignedShort4;
    case DataFormat::R8: return cudaResViewFormatSignedChar1;
    case DataFormat::RG8: return cudaResViewFormatSignedChar2;
//  case DataFormat::RGB8:
    case DataFormat::RGBA8: return cudaResViewFormatSignedChar4;
    case DataFormat::uR32: return cudaResViewFormatUnsignedShort1;
    case DataFormat::uRG32: return cudaResViewFormatUnsignedShort2;
//  case DataFormat::uRGB32:
    case DataFormat::uRGBA32: return cudaResViewFormatUnsignedShort4;
    case DataFormat::uR16: return cudaResViewFormatSignedShort1;
    case DataFormat::uRG16: return cudaResViewFormatUnsignedShort2;
//  case DataFormat::uRGB16:
    case DataFormat::uRGBA16: return cudaResViewFormatUnsignedShort4;
    case DataFormat::uR8: return cudaResViewFormatUnsignedChar1;
    case DataFormat::uRG8: return cudaResViewFormatUnsignedChar2;
//  case DataFormat::uRGB8:
    case DataFormat::uRGBA8: return cudaResViewFormatUnsignedChar4;
    case DataFormat::R16SNorm: return cudaResViewFormatSignedShort1;
    case DataFormat::RG16SNorm: return cudaResViewFormatSignedShort2;
//  case DataFormat::RGB16SNorm:
    case DataFormat::RGBA16SNorm: return cudaResViewFormatSignedShort4;
    case DataFormat::R8SNorm: return cudaResViewFormatSignedChar1;
    case DataFormat::RG8SNorm: return cudaResViewFormatSignedChar2;
//  case DataFormat::RGB8SNorm:
    case DataFormat::RGBA8SNorm: return cudaResViewFormatSignedChar4;
    case DataFormat::R16UNorm: return cudaResViewFormatUnsignedShort1;
    case DataFormat::RG16UNorm: return cudaResViewFormatUnsignedShort2;
//  case DataFormat::RGB16UNorm:
    case DataFormat::RGBA16UNorm: return cudaResViewFormatUnsignedShort4;
    case DataFormat::R8UNorm: return cudaResViewFormatUnsignedChar1;
    case DataFormat::RG8UNorm: return cudaResViewFormatUnsignedChar2;
//  case DataFormat::RGB8UNorm:
    case DataFormat::RGBA8UNorm: return cudaResViewFormatUnsignedChar4;
    //case DataFormat::D16: 
    //case DataFormat::D24S8:
    //case DataFormat::D32:
    //case DataFormat::R10G10B10A2:
    //case DataFormat::uR10G10B10A2: 
    default: TGE_ASSERT(false, "Unsupported format");
    }
    return cudaResViewFormatNone;
}

inline cudaTextureReadMode TranslateReadMode(DataFormat fmt)
{
    switch(fmt)
    {
    case DataFormat::R32F:
    case DataFormat::RG32F:
    case DataFormat::RGB32F:
    case DataFormat::RGBA32F:
    case DataFormat::R16F:
    case DataFormat::RG16F:
//  case DataFormat::RGB16F:
    case DataFormat::RGBA16F:
    case DataFormat::R32:
    case DataFormat::RG32:
    case DataFormat::RGB32:
    case DataFormat::RGBA32:
    case DataFormat::R16:
    case DataFormat::RG16:
//  case DataFormat::RGB16:
    case DataFormat::RGBA16:
    case DataFormat::R8:
    case DataFormat::RG8:
//  case DataFormat::RGB8:
    case DataFormat::RGBA8:
    case DataFormat::uR32:
    case DataFormat::uRG32:
    case DataFormat::uRGB32:
    case DataFormat::uRGBA32:
    case DataFormat::uR16:
    case DataFormat::uRG16:
//  case DataFormat::uRGB16:
    case DataFormat::uRGBA16:
    case DataFormat::uR8:
    case DataFormat::uRG8:
//  case DataFormat::uRGB8:
    case DataFormat::uRGBA8: return cudaReadModeElementType;
    case DataFormat::R16SNorm:
    case DataFormat::RG16SNorm:
//  case DataFormat::RGB16SNorm:
    case DataFormat::RGBA16SNorm:
    case DataFormat::R8SNorm:
    case DataFormat::RG8SNorm:
//  case DataFormat::RGB8SNorm:
    case DataFormat::RGBA8SNorm:
    case DataFormat::R16UNorm:
    case DataFormat::RG16UNorm:
//  case DataFormat::RGB16UNorm:
    case DataFormat::RGBA16UNorm:
    case DataFormat::R8UNorm:
    case DataFormat::RG8UNorm:
//  case DataFormat::RGB8UNorm:
    case DataFormat::RGBA8UNorm: return cudaReadModeNormalizedFloat;
    default: TGE_ASSERT(false, "Unsupported format");
    }
    return cudaReadModeElementType;
}

cudaTextureObject_t CreateCudaTexture(const Texture* tex, uint32_t flags = TEMPEST_CUDA_TEXTURE_SRGB);
cudaTextureObject_t CreateCudaTexture(const CubeMap* tex, uint32_t flags = TEMPEST_CUDA_TEXTURE_SRGB);
void CudaTextureDeleter(cudaTextureObject_t cuda_tex);
}

#endif // _TEMPEST_COMPUTE_TEXTURE_HH_
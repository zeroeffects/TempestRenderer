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

#ifndef _TEMPEST_TEXTURE_HH_
#define _TEMPEST_TEXTURE_HH_

#include <cstdint>
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/spectrum.hh"

#include <memory>
#include <algorithm>

#ifndef __CUDACC__
typedef unsigned long long cudaSurfaceObject_t;
struct cudaArray;
typedef cudaArray* cudaArray_t;
#else
#   include <surface_types.h>
#endif

struct cudaGraphicsResource;

namespace Tempest
{
enum class TextureTiling
{
    Flat,
    Cube,
    Volume,
    Array
};

enum class TextureSampling
{
    Bilinear,
    Nearest
};

struct TextureDescription
{
    uint16_t        Width;
    uint16_t        Height;
    uint16_t        Depth = 1;
    uint16_t        Samples = 1;
    DataFormat      Format;
    TextureSampling Sampling = TextureSampling::Bilinear;
    TextureTiling   Tiling = TextureTiling::Flat;
};

enum class SamplerAddressMode 
{
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
    MirrorClampToEdge
};

struct SamplerDescription
{
    TextureSampling         MagFilter = TextureSampling::Bilinear,
                            MinFilter = TextureSampling::Bilinear,
                            MipmapMode = TextureSampling::Bilinear;
    SamplerAddressMode      AddressModeU = SamplerAddressMode::ClampToEdge,
                            AddressModeV = SamplerAddressMode::ClampToEdge,
                            AddressModeW = SamplerAddressMode::ClampToEdge;
    float                   MipLodBias = 0.0f;
    bool                    EnableAnisotropy = false;
    float                   MaxAnisotropy = 0;
    bool                    CompareEnable = false;
    ComparisonFunction      CompareOperation = ComparisonFunction::Never;
    float                   MinLod = 0.0f;
    float                   MaxLod = 0.0f;
};

struct CUDATextureResource
{
    cudaGraphicsResource* Resource;
    TextureDescription    Description;

    operator bool() const { return Resource != nullptr; }
};

struct CUDASurfaceResource
{
    cudaArray_t           Array;
    cudaSurfaceObject_t   Surface;
    CUDATextureResource   Texture;

    operator bool() const { return Surface != 0; }
};

inline bool operator==(const TextureDescription& lhs, const TextureDescription& rhs)
{
    return lhs.Width == rhs.Width &&
           lhs.Height == rhs.Height &&
           lhs.Depth == rhs.Depth &&
           lhs.Samples == rhs.Samples &&
           lhs.Format == rhs.Format &&
           lhs.Tiling == rhs.Tiling;
}

inline bool operator!=(const TextureDescription& lhs, const TextureDescription& rhs)
{
    return lhs.Width != rhs.Width     ||
           lhs.Height != rhs.Height   ||
           lhs.Depth != rhs.Depth     ||
           lhs.Samples != rhs.Samples ||
           lhs.Format != rhs.Format   ||
           lhs.Tiling != rhs.Tiling;
}

struct SpectrumExtractor
{
    typedef Spectrum ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RGBA8UNorm:
        {
            return SRGBToSpectrum(ToVector3(*reinterpret_cast<const uint32_t*>(data)));
        }
        case DataFormat::RGB32F:
        {
            auto vec_data = reinterpret_cast<const Vector3*>(data);
            return RGBToSpectrum(*vec_data);
        }
        case DataFormat::RGBA32F:
        {
            return RGBToSpectrum(*reinterpret_cast<const Vector3*>(data));
        }
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return {};
    }
};

struct RGExtractor
{
    typedef Vector2 ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RG8SNorm:
        {
            auto* rg = reinterpret_cast<const int8_t*>(data);
            return Vector2{ rg[0]/127.0f, rg[1]/127.0f };
        }
        case DataFormat::RG8UNorm:
        {
            auto* rg = reinterpret_cast<const uint8_t*>(data);
            return Vector2{ rg[0]/255.0f, rg[1]/255.0f };
        }
        case DataFormat::RG16SNorm:
        {
            auto* rg = reinterpret_cast<const int16_t*>(data);
            return Vector2{ rg[0]/32767.0f, rg[1]/32767.0f };
        }
        case DataFormat::RG16UNorm:
        {
            auto* rg = reinterpret_cast<const uint16_t*>(data);
            return Vector2{ rg[0]/65535.0f, rg[1]/65535.0f };
        }
        case DataFormat::RG32F:
        case DataFormat::RGB32F:
        case DataFormat::RGBA32F:
        {
            return *reinterpret_cast<const Vector2*>(data);
        }
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return {};
    }
};

struct RGBExtractor
{
    typedef Vector3 ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RGBA8SNorm:
        {
            auto* rgb = reinterpret_cast<const int8_t*>(data);
            return Vector3{ rgb[0]/127.0f, rgb[1]/127.0f, rgb[2]/127.0f };
        }
        case DataFormat::RGBA8UNorm:
        {
            auto* rgb = reinterpret_cast<const uint8_t*>(data);
            return Vector3{ rgb[0]/255.0f, rgb[1]/255.0f, rgb[2]/255.0f };
        }
        case DataFormat::RGBA16SNorm:
        {
            auto* rgb = reinterpret_cast<const int16_t*>(data);
            return Vector3{ rgb[0]/32767.0f, rgb[1]/32767.0f, rgb[2]/32767.0f };
        }
        case DataFormat::RGBA16UNorm:
        {
            auto* rgb = reinterpret_cast<const int16_t*>(data);
            return Vector3{ rgb[0]/65535.0f, rgb[1]/65535.0f, rgb[2]/65535.0f };
        }
        case DataFormat::RGB32F:
        {
            return *reinterpret_cast<const Vector3*>(data);
        }
        case DataFormat::RGBA32F:
        {
            return *reinterpret_cast<const Vector3*>(data);
        }
        case DataFormat::R8SNorm:
        {
            float r = *reinterpret_cast<const int8_t*>(data)/127.0f;
            return ToVector3(r);
        }
        case DataFormat::R8UNorm:
        {
            float r = *reinterpret_cast<const uint8_t*>(data)/255.0f;
            return ToVector3(r);
        }
        case DataFormat::R16SNorm:
        {
            float r = *reinterpret_cast<const int8_t*>(data)/32767.0f;
            return ToVector3(r);
        }
        case DataFormat::R16UNorm:
        {
            float r = *reinterpret_cast<const uint8_t*>(data)/65535.0f;
            return ToVector3(r);
        }
        case DataFormat::R32F:
        {
            return ToVector3(*reinterpret_cast<const float*>(data));
        }
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return {};
    }
};

struct RGBAExtractor
{
    typedef Vector4 ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RGBA8SNorm:
        {
            auto* rgba = reinterpret_cast<const int8_t*>(data);
            return Vector4{ rgba[0] / 127.0f, rgba[1] / 127.0f, rgba[2] / 127.0f, rgba[3] / 127.0f };
        }
        case DataFormat::RGBA8UNorm:
        {
            auto* rgba = reinterpret_cast<const uint8_t*>(data);
            return Vector4{ rgba[0] / 255.0f, rgba[1] / 255.0f, rgba[2] / 255.0f, rgba[3] / 255.0f };
        }
        case DataFormat::RGBA16SNorm:
        {
            auto* rgba = reinterpret_cast<const int16_t*>(data);
            return Vector4{ rgba[0] / 32767.0f, rgba[1] / 32767.0f, rgba[2] / 32767.0f, rgba[3] / 32767.0f };
        }
        case DataFormat::RGBA16UNorm:
        {
            auto* rgba = reinterpret_cast<const int16_t*>(data);
            return Vector4{ rgba[0] / 65535.0f, rgba[1] / 65535.0f, rgba[2] / 65535.0f, rgba[3] / 65535.0f };
        }
        case DataFormat::RGBA32F:
        {
            return *reinterpret_cast<const Vector4*>(data);
        }
        case DataFormat::R8SNorm:
        {
            float r = *reinterpret_cast<const int8_t*>(data) / 127.0f;
            return ToVector4(r);
        }
        case DataFormat::R8UNorm:
        {
            float r = *reinterpret_cast<const uint8_t*>(data) / 255.0f;
            return ToVector4(r);
        }
        case DataFormat::R16SNorm:
        {
            float r = *reinterpret_cast<const int8_t*>(data) / 32767.0f;
            return ToVector4(r);
        }
        case DataFormat::R16UNorm:
        {
            float r = *reinterpret_cast<const uint8_t*>(data) / 65535.0f;
            return ToVector4(r);
        }
        case DataFormat::R32F:
        {
            return ToVector4(*reinterpret_cast<const float*>(data));
        }
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return{};
    }
};

// No, it is not the most potent of them all
struct RedExtractor
{
    typedef float ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RGBA8UNorm: return (1.0f/255.0f)*rgbaR(*reinterpret_cast<const uint32_t*>(data));
        case DataFormat::RGBA32F: return reinterpret_cast<const Vector4*>(data)->x;
        case DataFormat::R16UNorm: return (*reinterpret_cast<const uint16_t*>(data))/65535.0f;
        case DataFormat::R16SNorm: return (*reinterpret_cast<const int16_t*>(data))/32767.0f;
        case DataFormat::R8UNorm: return (*reinterpret_cast<const uint8_t*>(data))/255.0f;
        case DataFormat::R8SNorm: return (*reinterpret_cast<const int8_t*>(data))/127.0f;
        case DataFormat::R32F: return *reinterpret_cast<const float*>(data);
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return 0.0f;
    }
};

struct AlphaExtractor
{
    typedef float ExtractType;

    inline EXPORT_CUDA static ExtractType extract(DataFormat fmt, const void* data)
    {
        switch(fmt)
        {
        case DataFormat::RGBA8UNorm: return (1.0f / 255.0f)*rgbaA(*reinterpret_cast<const uint32_t*>(data));
        case DataFormat::RGBA8SNorm: return (1.0f / 127.0f)*reinterpret_cast<const uint8_t*>(data)[3];
        case DataFormat::RGBA32F: return reinterpret_cast<const Vector4*>(data)->w;
        case DataFormat::RGBA16UNorm: return reinterpret_cast<const uint16_t*>(data)[3] / 65535.0f;
        case DataFormat::RGBA16SNorm: return reinterpret_cast<const int16_t*>(data)[3] / 32767.0f;
		case DataFormat::RGB32F: break;
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return 1.0f;
    }
};

class Texture
{
    TextureDescription         m_Header;
    std::unique_ptr<uint8_t[]> m_Data;
public:
    explicit Texture(const TextureDescription& desc, uint8_t* data)
        :   m_Header(desc),
            m_Data(data) {}
     ~Texture()=default;
    
    Texture(Texture&& tex)
        :   m_Header(tex.m_Header),
            m_Data(std::move(tex.m_Data)) {}

    Texture& operator=(Texture&& tex)
    {
        m_Header = tex.m_Header;
        m_Data = std::move(tex.m_Data);
		return *this;
    }

    explicit Texture(const Texture& tex);
    Texture& operator=(const Texture& tex);

    const TextureDescription& getHeader() const { return m_Header; }

    void setSamplingMode(TextureSampling mode) { m_Header.Sampling = mode; }

    void realloc(const TextureDescription& desc, uint8_t* data)
    {
		m_Header = desc;
        m_Data = std::unique_ptr<uint8_t[]>(data);
    }

    void realloc(const TextureDescription& desc)
    {
		m_Header = desc;
        m_Data = std::unique_ptr<uint8_t[]>(new uint8_t[m_Header.Width*m_Header.Height*DataFormatElementSize(m_Header.Format)]);
    }

    Vector4 gather(const Vector2& tc, int comp) const
    {
        auto fmt = m_Header.Format;
        auto data = m_Data.get();
        auto el_size = DataFormatElementSize(fmt);

        int64_t width = m_Header.Width;
        int64_t height = m_Header.Height;

		Vector2 tc_unorm{ width*tc.x - 0.5f, height*tc.y - 0.5f };

        size_t x0 = static_cast<size_t>(Modulo(FastFloorToInt64(tc_unorm.x), width));
        size_t y0 = static_cast<size_t>(Modulo(FastFloorToInt64(tc_unorm.y), height));
        size_t x1 = (x0 + 1) % width;
        size_t y1 = (y0 + 1) % height;

        auto comp_w = data + el_size*(y0*width + x0);
        auto comp_z = data + el_size*(y0*width + x1);
        auto comp_x = data + el_size*(y1*width + x0);
        auto comp_y = data + el_size*(y1*width + x1);

        TGE_ASSERT(comp < (int)DataFormatChannels(fmt), "invalid component");

        switch(fmt)
        {
        case DataFormat::R8SNorm:
        case DataFormat::RG8SNorm:
        case DataFormat::RGBA8SNorm:
        {
            return Vector4{ comp_x[comp] / 127.0f, comp_y[comp] / 127.0f, comp_z[comp]  / 127.0f, comp_w[comp] / 127.0f };
        }
        case DataFormat::R8UNorm:
        case DataFormat::RG8UNorm:
        case DataFormat::RGBA8UNorm:
        {
            TGE_ASSERT(comp < 4, "invalid component");
            return Vector4{ comp_x[comp] / 255.0f, comp_y[comp] / 255.0f, comp_z[comp] / 255.0f, comp_w[comp] / 255.0f };
        }
        case DataFormat::R16SNorm:
        case DataFormat::RG16SNorm:
        case DataFormat::RGBA16SNorm:
        {
            return Vector4{ reinterpret_cast<const int16_t*>(comp_x)[comp] / 32767.0f,
                            reinterpret_cast<const int16_t*>(comp_y)[comp] / 32767.0f,
                            reinterpret_cast<const int16_t*>(comp_z)[comp] / 32767.0f,
                            reinterpret_cast<const int16_t*>(comp_w)[comp] / 32767.0f };
        }
        case DataFormat::R16UNorm:
        case DataFormat::RG16UNorm:
        case DataFormat::RGBA16UNorm:
        {
            return Vector4{ reinterpret_cast<const int16_t*>(comp_x)[comp] / 65535.0f,
                            reinterpret_cast<const int16_t*>(comp_y)[comp] / 65535.0f,
                            reinterpret_cast<const int16_t*>(comp_z)[comp] / 65535.0f,
                            reinterpret_cast<const int16_t*>(comp_w)[comp] / 65535.0f };
        }
        case DataFormat::R32F:
        case DataFormat::RG32F:
        case DataFormat::RGBA32F:
        {
            return Vector4{ reinterpret_cast<const float*>(comp_x)[comp],
                            reinterpret_cast<const float*>(comp_y)[comp],
                            reinterpret_cast<const float*>(comp_z)[comp],
                            reinterpret_cast<const float*>(comp_w)[comp] };
        }
        default: TGE_ASSERT(false, "Unsupported format");
        }
        return{};
    }

    template<class TExtractor>
    typename TExtractor::ExtractType sample(const Vector2& tc) const
    {
        TGE_ASSERT(m_Header.Tiling == TextureTiling::Flat, "Invalid tiling format");
        switch(m_Header.Sampling)
        {
        default: TGE_ASSERT(false, "Unsupported sampling mode");
        case TextureSampling::Bilinear:
        {
            auto fmt = m_Header.Format;
            auto el_size = DataFormatElementSize(fmt);

            int64_t width = m_Header.Width;
            int64_t height = m_Header.Height;

			Vector2 tc_unorm{ width*tc.x - 0.5f, height*tc.y - 0.5f };

            auto x_trunc = FastFloorToInt64(tc_unorm.x),
                 y_trunc = FastFloorToInt64(tc_unorm.y);

            size_t x0 = static_cast<size_t>(Modulo(x_trunc, width));
            size_t y0 = static_cast<size_t>(Modulo(y_trunc, height));
            size_t x1 = (x0 + 1) % width;
            size_t y1 = (y0 + 1) % height;

            auto c00 = TExtractor::extract(fmt, m_Data.get() + el_size*(y0*width + x0));
            auto c01 = TExtractor::extract(fmt, m_Data.get() + el_size*(y0*width + x1));
            auto c10 = TExtractor::extract(fmt, m_Data.get() + el_size*(y1*width + x0));
            auto c11 = TExtractor::extract(fmt, m_Data.get() + el_size*(y1*width + x1));

            float fx1 = tc_unorm.x - (float)x_trunc,
                  fx0 = 1.0f - fx1,
                  fy1 = tc_unorm.y - (float)y_trunc,
                  fy0 = 1.0f - fy1;

            return (fx0 * c00 + fx1 * c01) * fy0 +
                   (fx0 * c10 + fx1 * c11) * fy1;
        }
        case TextureSampling::Nearest:
        {
            auto fmt = m_Header.Format;
            auto el_size = DataFormatElementSize(fmt);

            int64_t width = m_Header.Width;
            int64_t height = m_Header.Height;

            Vector2 tc_unorm{ width*tc.x, height*tc.y };

            auto x_trunc = FastFloorToInt64(tc_unorm.x),
                 y_trunc = FastFloorToInt64(tc_unorm.y);

            size_t x0 = static_cast<size_t>(Modulo(x_trunc, width));
            size_t y0 = static_cast<size_t>(Modulo(y_trunc, height));

            return TExtractor::extract(fmt, m_Data.get() + el_size*(y0*width + x0));
        }
        }
    }

    Spectrum sampleSpectrum(const Vector2& tc) const
    {
        return sample<SpectrumExtractor>(tc);
    }

    float sampleRed(const Vector2& tc) const
    {
        return sample<RedExtractor>(tc);
    }

    Vector2 sampleRG(const Vector2& tc) const
    {
        return sample<RGExtractor>(tc);
    }

    Vector3 sampleRGB(const Vector2& tc) const
    {
        return sample<RGBExtractor>(tc);
    }

	Vector4 sampleRGBA(const Vector2& tc) const
    {
        return sample<RGBAExtractor>(tc);
    }

    float fetchRed(uint32_t x, uint32_t y) const
    {
        TGE_ASSERT(x < m_Header.Width && y < m_Header.Height, "out of bounds access");
        auto fmt = m_Header.Format;
        auto el_size = DataFormatElementSize(fmt);
        return RedExtractor::extract(fmt, m_Data.get() + el_size*(y*m_Header.Width + x));
    }

    float fetchAlpha(uint32_t x, uint32_t y) const
    {
        TGE_ASSERT(x < m_Header.Width && y < m_Header.Height, "out of bounds access");
        auto fmt = m_Header.Format;
        auto el_size = DataFormatElementSize(fmt);
        return AlphaExtractor::extract(fmt, m_Data.get() + el_size*(y*m_Header.Width + x));
    }

    Vector2 fetchRG(uint32_t x, uint32_t y) const
    {
        TGE_ASSERT(x < m_Header.Width && y < m_Header.Height, "out of bounds access");
        auto fmt = m_Header.Format;
        auto el_size = DataFormatElementSize(fmt);
        return RGExtractor::extract(fmt, m_Data.get() + el_size*(y*m_Header.Width + x));
    }

    Vector3 fetchRGB(uint32_t x, uint32_t y) const
    {
        TGE_ASSERT(x < m_Header.Width && y < m_Header.Height, "out of bounds access");
        auto fmt = m_Header.Format;
        auto el_size = DataFormatElementSize(fmt);
        return RGBExtractor::extract(fmt, m_Data.get() + el_size*(y*m_Header.Width + x));
    }

    Vector4 fetchRGBA(uint32_t x, uint32_t y) const
    {
        TGE_ASSERT(x < m_Header.Width && y < m_Header.Height, "out of bounds access");
        auto fmt = m_Header.Format;
        auto el_size = DataFormatElementSize(fmt);
        return RGBAExtractor::extract(fmt, m_Data.get() + el_size*(y*m_Header.Width + x));
    }

    template<class T>
    void writeValue(T val, uint32_t x_bytes, uint32_t y)
    {
        auto dt_size = DataFormatElementSize(m_Header.Format);
        uint32_t pitch = m_Header.Width*dt_size;
        TGE_ASSERT(sizeof(T) == dt_size || sizeof(T) == dt_size/DataFormatChannels(m_Header.Format), "Invalid data size");
        TGE_ASSERT(x_bytes + sizeof(T) <= pitch, "Out of x bounds write");
        TGE_ASSERT(y < m_Header.Height, "Out y bounds write");
        *reinterpret_cast<T*>(m_Data.get() + y*pitch + x_bytes) = val;
    }

    void convertToRGBA();
    void convertToLuminance();
    void convertToUNorm8();

    // Flipping should be done within image loaders. Don't use this function to fix bad loader after it is called.
    // NOTE: Also, it changes the data inplace - you might mess up with other functions if you use it across the pipeline.
    void flipY();

    const uint8_t* getData() const { return m_Data.get(); }
    uint8_t* getData() { return m_Data.get(); }

    uint8_t* release() { return m_Data.release(); }
};

typedef std::unique_ptr<Tempest::Texture> TexturePtr;
}

#endif // _TEMPEST_TEXTURE_HH_

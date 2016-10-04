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

#include "tempest/graphics/texture.hh"
#include "tempest/graphics/rendering-definitions.hh"

#include <cstring>

namespace Tempest
{
void Texture::convertToRGBA()
{
    auto fmt = m_Header.Format;
    switch(fmt)
    {
    case DataFormat::R8:
    case DataFormat::R16:
    case DataFormat::R32:
    case DataFormat::uR8:
    case DataFormat::uR16:
    case DataFormat::uR32:
    case DataFormat::R8UNorm:
    case DataFormat::R16UNorm:
    case DataFormat::R8SNorm:
    case DataFormat::R16SNorm:
    {
        size_t bpp = DataFormatElementSize(fmt);
        size_t area = m_Header.Height*m_Header.Width;
        size_t size = 4*area*bpp;
        std::unique_ptr<uint8_t[]> new_data(new uint8_t[size]);
        auto* in_data = m_Data.get();
        for(auto* out_data = new_data.get(), *out_data_end = out_data + size; out_data != out_data_end;)
        {
            memcpy(out_data, in_data, bpp); out_data += bpp;
            memcpy(out_data, in_data, bpp); out_data += bpp;
            memcpy(out_data, in_data, bpp); out_data += bpp;
            memset(out_data, 0xFF, bpp); out_data += bpp;
            in_data += bpp;
        }
        std::swap(m_Data, new_data);
        switch(fmt)
        {
        case DataFormat::R8: m_Header.Format = DataFormat::RGBA8;  break;
        case DataFormat::R16: m_Header.Format = DataFormat::RGBA16; break;
        case DataFormat::R32: m_Header.Format = DataFormat::RGBA32; break;
        case DataFormat::uR8: m_Header.Format = DataFormat::uRGBA8; break;
        case DataFormat::uR16: m_Header.Format = DataFormat::uRGBA16; break;
        case DataFormat::uR32: m_Header.Format = DataFormat::uRGBA32; break;
        case DataFormat::R8UNorm: m_Header.Format = DataFormat::RGBA8UNorm; break;
        case DataFormat::R16UNorm: m_Header.Format = DataFormat::RGBA16UNorm; break;
        case DataFormat::R8SNorm: m_Header.Format = DataFormat::RGBA8SNorm; break;
        case DataFormat::R16SNorm: m_Header.Format = DataFormat::RGBA16SNorm; break;
        }
    } break;
    case DataFormat::uRGBA8:
    case DataFormat::uRGBA16:
    case DataFormat::uRGBA32:
    case DataFormat::RGBA8:
    case DataFormat::RGBA16:
    case DataFormat::RGBA32:
    case DataFormat::RGBA16F:
    case DataFormat::RGBA32F:
    case DataFormat::RGBA8UNorm:
    case DataFormat::RGBA16UNorm:
    case DataFormat::RGBA8SNorm:
    case DataFormat::RGBA16SNorm:
        break;
    case DataFormat::RGB32F:
    {
        size_t area = m_Header.Height*m_Header.Width;
        auto new_data_ptr = new Tempest::Vector4[area];
        std::unique_ptr<uint8_t[]> new_data(reinterpret_cast<uint8_t*>(new_data_ptr));

        auto old_data = reinterpret_cast<Tempest::Vector3*>(m_Data.get());

        for(size_t idx = 0; idx < area; ++idx)
        {
            auto& new_elem = new_data_ptr[idx];
            auto& old_elem = old_data[idx];

            new_elem = { old_elem.x, old_elem.y, old_elem.z, 1.0f };
        }
        m_Header.Format = Tempest::DataFormat::RGBA32F;
        std::swap(m_Data, new_data);
    } break;
    default:
        TGE_ASSERT(false, "Unsupported format");
    }
}

void Texture::convertToUNorm8()
{
    // TODO: optimize
    auto fmt = m_Header.Format;
    auto channel_count = DataFormatChannels(fmt);
    auto channel_size = DataFormatElementSize(fmt)/channel_count;

    size_t area = m_Header.Height*m_Header.Width;

    switch(channel_count)
    {
    case 1:
    {
        std::unique_ptr<uint8_t[]> new_data(new uint8_t[area]);
        for(uint32_t y = 0; y < m_Header.Height; ++y)
            for(uint32_t x = 0; x < m_Header.Width; ++x)
            {
                new_data[y*m_Header.Width + x] = (uint8_t)Clampf(fetchRed(x, y)*255.0f, 0.0f, 1.0f);
            }

        m_Header.Format = DataFormat::R8UNorm;
        std::swap(m_Data, new_data);
    } break;
    case 2:
    {
        std::unique_ptr<uint16_t[]> new_data(new uint16_t[area]);
        for(uint32_t y = 0; y < m_Header.Height; ++y)
            for(uint32_t x = 0; x < m_Header.Width; ++x)
            {
                auto rg = fetchRG(x, y);

                new_data[y*m_Header.Width + x] = (uint16_t)Clampf(rg.x*255.0f, 0.0f, 1.0f) | ((uint16_t)Clampf(rg.x*255.0f, 0.0f, 1.0f) << 8);
            }

        m_Header.Format = DataFormat::RG8UNorm;
        m_Data = decltype(m_Data)(reinterpret_cast<uint8_t*>(new_data.release()));
    } break;
    case 3:
    case 4:
    {
        std::unique_ptr<uint32_t[]> new_data(new uint32_t[area]);
        for(uint32_t y = 0; y < m_Header.Height; ++y)
            for(uint32_t x = 0; x < m_Header.Width; ++x)
            {
                new_data[y*m_Header.Width + x] = ToColor(fetchRGBA(x, y));
            }

        m_Header.Format = DataFormat::RGBA8UNorm;
        m_Data = decltype(m_Data)(reinterpret_cast<uint8_t*>(new_data.release()));
    } break;
    }
}

void Texture::convertToLuminance()
{
    auto fmt = m_Header.Format;
    switch(fmt)
    {
    case DataFormat::R8:
    case DataFormat::R16:
    case DataFormat::R32:
    case DataFormat::uR8:
    case DataFormat::uR16:
    case DataFormat::uR32:
    case DataFormat::R8UNorm:
    case DataFormat::R16UNorm:
    case DataFormat::R8SNorm:
    case DataFormat::R16SNorm:
        break;
    case DataFormat::uRGBA8:
    case DataFormat::uRGBA16:
    case DataFormat::uRGBA32:
    case DataFormat::RGBA8:
    case DataFormat::RGBA16:
    case DataFormat::RGBA32:
    case DataFormat::RGBA8UNorm:
    case DataFormat::RGBA16UNorm:
    case DataFormat::RGBA8SNorm:
    case DataFormat::RGBA16SNorm:
    {
        size_t area = m_Header.Height*m_Header.Width;
        auto new_data_ptr = new float[area];
        std::unique_ptr<uint8_t[]> new_data(reinterpret_cast<uint8_t*>(new_data_ptr));

        auto old_data = reinterpret_cast<Tempest::Vector3*>(m_Data.get());

        for(size_t y = 0, yend = m_Header.Height; y < yend; ++y)
            for(size_t x = 0, xend = m_Header.Width; x < xend; ++x)
            {
                auto luminance = RGBToLuminance(ConvertSRGBToLinear(fetchRGB(x, y)));
                new_data_ptr[y*xend + x] = luminance;
            }

        m_Header.Format = Tempest::DataFormat::R32F;
        std::swap(m_Data, new_data);
    } break;
    case DataFormat::RGBA16F:
    case DataFormat::RGB32F:
    case DataFormat::RGBA32F:
    {
        size_t area = m_Header.Height*m_Header.Width;
        auto new_data_ptr = new float[area];
        std::unique_ptr<uint8_t[]> new_data(reinterpret_cast<uint8_t*>(new_data_ptr));

        auto old_data = reinterpret_cast<Tempest::Vector3*>(m_Data.get());

        for(size_t y = 0, yend = m_Header.Height; y < yend; ++y)
            for(size_t x = 0, xend = m_Header.Width; x < xend; ++x)
            {
                auto luminance = RGBToLuminance(fetchRGB(x, y));
                new_data_ptr[y*xend + x] = luminance;
            }

        m_Header.Format = Tempest::DataFormat::R32F;
        std::swap(m_Data, new_data);
    } break;
    default:
        TGE_ASSERT(false, "Unsupported format");
    }
}

Texture::Texture(const Texture& tex)
    :   m_Header(tex.m_Header)
{
    auto tex_size = m_Header.Width*m_Header.Height*DataFormatElementSize(m_Header.Format);
    if(!m_Data)
    {
        m_Data = std::unique_ptr<uint8_t[]>(new uint8_t[tex_size]);
    }
    memcpy(m_Data.get(), tex.m_Data.get(), tex_size);
}

Texture& Texture::operator=(const Texture& tex)
{
    auto tex_size = m_Header.Width*m_Header.Height*DataFormatElementSize(m_Header.Format);
    if(!m_Data)
    {
        m_Data = std::unique_ptr<uint8_t[]>(new uint8_t[tex_size]);
    }
    m_Header = tex.m_Header;
    memcpy(m_Data.get(), tex.m_Data.get(), tex_size);
    return *this;
}

void Texture::flipY()
{
    auto data_ptr = m_Data.get();
    auto data_size = Tempest::DataFormatElementSize(m_Header.Format);
    for(uint32_t y = 0, height = m_Header.Height, yend = height/2; y < yend; ++y)
    {
        for(uint32_t x = 0, width = m_Header.Width; x < width; ++x)
        {
            auto* src = data_ptr + (y*width + x)*data_size;
            auto* dst = data_ptr + ((height - 1 - y)*width + x)*data_size;
            std::swap_ranges(src, src + data_size, dst);
        }
    }
}
}
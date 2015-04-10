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
        std::unique_ptr<uint8[]> new_data(new uint8[size]);
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
    default:
        Log(LogLevel::Error, "Unsupported format");
    }
}
}
/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#include "tempest/graphics/cube-map.hh"
#include "tempest/math/functions.hh"

#include <algorithm>
#include <cstring>

namespace Tempest
{
CubeMap::CubeMap(Texture** textures)
{
    m_Description.Format = DataFormat::Unknown;
    for(size_t face_idx = 0; face_idx < 6; ++face_idx)
    {
        auto& hdr = textures[face_idx]->getHeader();
        if(m_Description.Format == DataFormat::Unknown)
        {
            m_Description = hdr;
            if(hdr.Width != hdr.Height) // We don't support non-square cube maps
                return;
        }
        else if(m_Description != hdr)
        {
            return;
        }
    }

    size_t data_elem_size = DataFormatElementSize(m_Description.Format);
    TGE_ASSERT(data_elem_size == sizeof(uint32_t) && m_Description.Width > 16, "Invalid format");

    size_t face_size = m_Description.Width*m_Description.Height*data_elem_size;
    m_Data = std::unique_ptr<uint8_t[]>(new uint8_t[6*face_size]);
    size_t offset = 0;
    // TODO: Use Morton tiling pattern to improve data locality
    for(size_t face_idx = 0; face_idx < 6; ++face_idx)
    {
        auto face_data = m_Data.get() + offset;
        auto tex_face_data = textures[face_idx]->getData();
        auto pitch = m_Description.Width*data_elem_size;
        for(size_t y = 0, y_end = m_Description.Height; y < y_end; ++y)
        {
            memcpy(reinterpret_cast<uint8_t*>(face_data) + y*pitch, tex_face_data + (y_end - 1 - y)*pitch, pitch);
        }

        offset += face_size;
    }
}

template<class TExtractor>
typename TExtractor::ExtractType SampleCubeMapTexel(const TextureDescription& hdr, const uint8_t* data, const Vector3& dir)
{
    auto fmt = hdr.Format;
    size_t width = hdr.Width,
          height = hdr.Height,
          face_offset;

    auto el_size = DataFormatElementSize(fmt);
    
    auto tc = CartesianToCubeMapCoordinates(dir, &face_offset);

    face_offset *= width*height;

    float sf = width * tc.x - 0.5f;
    float tf = height * tc.y - 0.5f;
    int sf0 = Clamp((int)sf, 0, (int)width - 1);
    int tf0 = Clamp((int)tf, 0, (int)height - 1);
    int sf1 = std::min(sf0 + 1, (int)width - 1);
    int tf1 = std::min(tf0 + 1, (int)height - 1);

    auto c00 = TExtractor::extract(fmt, data + (face_offset + sf0 + tf0*width)*el_size),
         c01 = TExtractor::extract(fmt, data + (face_offset + sf1 + tf0*width)*el_size),
         c10 = TExtractor::extract(fmt, data + (face_offset + sf0 + tf1*width)*el_size),
         c11 = TExtractor::extract(fmt, data + (face_offset + sf1 + tf1*width)*el_size);

    float fx1 = sf - sf0,
          fx0 = 1.0f - fx1,
          fy1 = tf - tf0,
          fy0 = 1.0f - fy1;

    return (((fx0 * c00 + fx1 * c01) * fy0 +
            (fx0 * c10 + fx1 * c11) * fy1));
}
    
Spectrum CubeMap::sampleSpectrum(const Vector3& dir)
{
    return SampleCubeMapTexel<SpectrumExtractor>(m_Description, m_Data.get(), dir);
}

Vector3 CubeMap::sampleRGB(const Vector3& dir)
{
    return SampleCubeMapTexel<RGBExtractor>(m_Description, m_Data.get(), dir);
}

Vector4 CubeMap::sampleRGBA(const Vector3& dir)
{
    return SampleCubeMapTexel<RGBAExtractor>(m_Description, m_Data.get(), dir);
}

Vector3 CubeMap::fetchFaceRGB(uint32_t face, uint32_t x, uint32_t y)
{
    auto fmt = m_Description.Format;
    size_t width = m_Description.Width,
          height = m_Description.Height,
          face_offset = face*width*height;

    auto el_size = DataFormatElementSize(fmt);

    return RGBExtractor::extract(fmt, m_Data.get() + (face_offset + x + y*width)*el_size);
}
}

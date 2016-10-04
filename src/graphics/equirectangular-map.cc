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

#include "tempest/graphics/equirectangular-map.hh"
#include "tempest/graphics/cube-map.hh"

namespace Tempest
{
CubeMap* ConvertEquirectangularMapToCubeMap(const TextureDescription& cube_tex_desc, const EquirectangularMap& eqrect_map)
{
    switch(cube_tex_desc.Format)
    {
    case DataFormat::RGBA8UNorm:
    {
        size_t plane_size = (size_t)cube_tex_desc.Width*cube_tex_desc.Height;
        uint32_t* data = new uint32_t[6*plane_size];

        auto cube_map = new CubeMap(cube_tex_desc, reinterpret_cast<uint8_t*>(data));

        for(size_t face = 0; face < 6; ++face)
        {
            auto* data_plane = data + face*plane_size;
            for(size_t y = 0, height = cube_tex_desc.Height; y < height; ++y)
                for(size_t x = 0, width = cube_tex_desc.Width; x < width; ++x)
                {
                    Vector2 tc{ (x + 0.5f)/width, (y + 0.5f)/height };
                    auto dir = CubeMapToCartesianCoordinates(tc, face);
                    data_plane[y*width + x] = ToColor(ConvertLinearToSRGB(eqrect_map.sampleRGB(dir)));
                }
        }
        return cube_map;
    }
    case DataFormat::RGBA32F:
    {
        size_t plane_size = (size_t)cube_tex_desc.Width*cube_tex_desc.Height;
        Vector4* data = new Vector4[6*plane_size];

        auto cube_map = new CubeMap(cube_tex_desc, reinterpret_cast<uint8_t*>(data));

        for(size_t face = 0; face < 6; ++face)
        {
            auto* data_plane = data + face*plane_size;
            for(size_t y = 0, height = cube_tex_desc.Height; y < height; ++y)
                for(size_t x = 0, width = cube_tex_desc.Width; x < width; ++x)
                {
                    Vector2 tc{ (x + 0.5f)/width, (y + 0.5f)/height };
                    auto dir = Tempest::CubeMapToCartesianCoordinates(tc, face);
                    auto rgb = eqrect_map.sampleRGB(dir);
                    data_plane[y*width + x] = (std::isfinite(rgb.x) || std::isfinite(rgb.y) || std::isfinite(rgb.z)) ? Tempest::Vector4{ rgb.x, rgb.y, rgb.z, 1.0f } : Tempest::Vector4{};
                }
        }
        return cube_map;
    }
    }

    Log(LogLevel::Error, "Unsupported format");
    return nullptr;
}
}
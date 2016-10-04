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

#include "tempest/utils/parse-command-line.hh"
#include "tempest/graphics/equirectangular-map.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/cube-map.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/spherical-harmonics.hh"

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("env-map-tool", true);
    parser.createOption('o', "output", "Specify output file for the processed data", true, "output.exr");
    parser.createOption('e', "extract", "Specify extraction mode (cubemap-unfold, SH-unfold)", true, "cubemap-unfold");
    parser.createOption('s', "SH-order", "Specify order of the spherical harmonics", true, "5");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    if(parser.getUnassociatedCount() != 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify exactly one input image");
        return EXIT_FAILURE;
    }

    auto texame = parser.getUnassociatedArgument(0);
    Tempest::TexturePtr texture(Tempest::LoadImage(Tempest::Path(texame)));
    if(!texture)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load texture: ", texame);
        return EXIT_FAILURE;
    }

    Tempest::EquirectangularMap eqrect_map(texture.get());

    auto& hdr = texture->getHeader();
    auto cubemap_size = std::max(Tempest::NextPowerOf2(hdr.Width / 4), Tempest::NextPowerOf2(hdr.Height / 4)); // Reduce aliasing, but don't do it too large.

   

    auto extract_mode_str = parser.extractString("extract");
    if(extract_mode_str == "cubemap-unfold")
    {
        Tempest::TextureDescription cubemap_tex_desc;
        cubemap_tex_desc.Width = cubemap_size;
        cubemap_tex_desc.Height = cubemap_size;
        cubemap_tex_desc.Format = Tempest::DataFormat::RGBA32F;

        std::unique_ptr<Tempest::CubeMap> cubemap(Tempest::ConvertEquirectangularMapToCubeMap(cubemap_tex_desc, eqrect_map));

        Tempest::TextureDescription unfold_cube_desc;
        unfold_cube_desc.Width = cubemap_size*3;
        unfold_cube_desc.Height = cubemap_size*4;
        unfold_cube_desc.Format = Tempest::DataFormat::RGB32F;

        std::unique_ptr<Tempest::Vector3[]> data(new Tempest::Vector3[unfold_cube_desc.Width*unfold_cube_desc.Height]);

        struct FaceInfo
        {
            uint32_t StartX,
                     StartY;
            int32_t  M00, M01,
                     M10, M11;
        } face_data[] =
        {
            { cubemap_size,     3*cubemap_size, -1, 0, 0, -1 },
            { cubemap_size,     cubemap_size,   1, 0, 0, 1 },
            { cubemap_size,     0,              0, -1, 1, 0  },
            { cubemap_size,     2*cubemap_size, 0, 1, -1, 0 },
            { 2*cubemap_size,   cubemap_size,   1, 0, 0, 1 },
            { 0,                cubemap_size,   1, 0, 0, 1 },
        };
    
        for(uint32_t face_idx = 0; face_idx < 6; ++face_idx)
        {
            auto& face = face_data[face_idx];
            for(uint32_t y = 0; y < cubemap_size; ++y)
            {
                for(uint32_t x = 0; x < cubemap_size; ++x)
                {
                    int32_t x_t = face.M00*x + face.M01*y;
                    int32_t y_t = face.M10*x + face.M11*y;

                    if(x_t < 0)
                    {
                        x_t = cubemap_size - 1 + x_t;
                    }
                    if(y_t < 0)
                    {
                        y_t = cubemap_size - 1 + y_t;
                    }

                    data[(unfold_cube_desc.Height - 1 - (y + face.StartY))*unfold_cube_desc.Width + x + face.StartX] = cubemap->fetchFaceRGB(face_idx, x_t, y_t);
                }
            }
        }

        auto output_file = parser.extractString("output");
        status = Tempest::SaveImage(unfold_cube_desc, data.get(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to save output file: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else if(extract_mode_str == "SH-unfold")
    {
        Tempest::TextureDescription unfold_cube_desc;
        unfold_cube_desc.Width = cubemap_size*3;
        unfold_cube_desc.Height = cubemap_size*4;
        unfold_cube_desc.Format = Tempest::DataFormat::RGB32F;

        std::unique_ptr<Tempest::Vector3[]> data(new Tempest::Vector3[unfold_cube_desc.Width*unfold_cube_desc.Height]);

        uint32_t order = parser.extract<uint32_t>("SH-order");
        if(order == 0)
        {
            Tempest::Log(Tempest::LogLevel::Error, "You must specify SH order that is greater than 0");
            return EXIT_FAILURE;
        }

        std::unique_ptr<Tempest::Vector3[]> coeffs(Tempest::MonteCarloSphericalHarmonicsIntegrator(order, 1024,
                                        [&eqrect_map](const Tempest::Vector3& dir)
                                        {
                                            return eqrect_map.sampleRGB(dir);
                                        }));
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported extract operation: ", extract_mode_str);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
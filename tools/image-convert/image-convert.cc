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
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"

#include <cstdlib>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("image-convert", true);
    parser.createOption('o', "output", "Specify output image filename", true, "output.png");
    parser.createOption('z', "drop-z", "Drop Z component", false);

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    if(parser.getUnassociatedCount() != 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify one input image\n"
                                               "USAGE:\n"
                                               "\timage-convert [ <options> ] <input>");
        return EXIT_FAILURE;
    }

    Tempest::Path input_file(parser.getUnassociatedArgument(0));
    std::unique_ptr<Tempest::Texture> input_texture(Tempest::LoadImage(input_file));
    if(!input_texture)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load input texture: ", input_file);
        return EXIT_FAILURE;
    }

    auto& input_hdr = input_texture->getHeader();

    Tempest::Path output_file(parser.extractString("output"));
    auto ext = output_file.extension();

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = input_hdr.Width;
    tex_desc.Height = input_hdr.Height;
    auto tex_area = tex_desc.Width*tex_desc.Height;
    auto channel_count = Tempest::DataFormatChannels(input_hdr.Format);

    status = true;
    if(ext == "png" || ext == "tga")
    {
        switch(channel_count)
        {
        case 1:
        {
            tex_desc.Format = Tempest::DataFormat::R8UNorm;

            uint8_t* texture_data = new uint8_t[tex_area];
            Tempest::Texture result_tex(tex_desc, texture_data);
            for(uint32_t y = 0; y < tex_desc.Height; ++y)
                for(uint32_t x = 0; x < tex_desc.Width; ++x)
                {
                    texture_data[y*tex_desc.Width + x] = static_cast<uint8_t>(255.0f*input_texture->fetchRed(x, y) + 0.5f);
                }
            status = Tempest::SaveImage(tex_desc, texture_data, output_file);
        } break;
        case 2:
        {
            tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

            uint32_t* texture_data = new uint32_t[tex_area];
            Tempest::Texture result_tex(tex_desc, reinterpret_cast<uint8_t*>(texture_data));
            for(uint32_t y = 0; y < tex_desc.Height; ++y)
                for(uint32_t x = 0; x < tex_desc.Width; ++x)
                {
                    auto rg = input_texture->fetchRG(x, y);
                    
                    texture_data[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::Vector3{ rg.x, rg.y, 0.0f });
                }
            status = Tempest::SaveImage(tex_desc, texture_data, output_file);
        } break;
        case 3:
        {
            tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

            if(parser.isSet("drop-z"))
            {
                uint32_t* texture_data = new uint32_t[tex_area];
                Tempest::Texture result_tex(tex_desc, reinterpret_cast<uint8_t*>(texture_data));
                for(uint32_t y = 0; y < tex_desc.Height; ++y)
                    for(uint32_t x = 0; x < tex_desc.Width; ++x)
                    {
                        auto rgb = input_texture->fetchRGB(x, y);
                    
                        rgb.x /= rgb.z;
                        rgb.y /= rgb.z;
                        rgb.z = 0.0f;

                        texture_data[y*tex_desc.Width + x] = Tempest::ToColor(rgb);
                    }
                status = Tempest::SaveImage(tex_desc, texture_data, output_file);
            }
            else
            {
                uint32_t* texture_data = new uint32_t[tex_area];
                Tempest::Texture result_tex(tex_desc, reinterpret_cast<uint8_t*>(texture_data));
                for(uint32_t y = 0; y < tex_desc.Height; ++y)
                    for(uint32_t x = 0; x < tex_desc.Width; ++x)
                    {
                        auto rgb = input_texture->fetchRGB(x, y);
                    
                        texture_data[y*tex_desc.Width + x] =  Tempest::ToColor(rgb);
                    }
                status = Tempest::SaveImage(tex_desc, texture_data, output_file);
            }
        } break;
        case 4:
        {
            tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

            uint32_t* texture_data = new uint32_t[tex_area];
            Tempest::Texture result_tex(tex_desc, reinterpret_cast<uint8_t*>(texture_data));
            for(uint32_t y = 0; y < tex_desc.Height; ++y)
                for(uint32_t x = 0; x < tex_desc.Width; ++x)
                {
                    auto rgb = input_texture->fetchRGBA(x, y);
                    
                    texture_data[y*tex_desc.Width + x] = Tempest::ToColor(rgb);
                }
            status = Tempest::SaveImage(tex_desc, texture_data, output_file);
        } break;
        }
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported output file format: ", output_file);
        return EXIT_FAILURE;
    }

    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to save output file: ", output_file);
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}
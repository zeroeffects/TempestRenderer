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
#include "tempest/image/eps-draw.hh"
#include "tempest/image/image.hh"
#include "tempest/math/shapes.hh"

#include <cstdlib>
#include <algorithm>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("image-combine", true);
    parser.createOption('o', "output", "Specify output combined file (default: combined.png)", true, "combined.png");
    parser.createOption('w', "width", "Specify width of the final image", true, "0");
    parser.createOption('h', "height", "Specify height of the final image", true, "0");
    parser.createOption('s', "scale-to-range", "Scale image to the range of the smallest and highest value", false);
    parser.createOption('b', "background", "Specify background image", true);
    
    auto status =  parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto count = parser.getUnassociatedCount();
    if(count % 3)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid argument count:\n"
                                               "USAGE:\n"
                                               "\timage-combine <image-file> <x-axis> <y-axis>\n");
        return EXIT_FAILURE;
    }

    struct ImageData
    {
        float                               X,
                                            Y;
        std::unique_ptr<Tempest::Texture>   Texture;
    };

    size_t image_count = count/3;

    std::unique_ptr<ImageData[]> images(new ImageData[image_count]);
    
    uint32_t final_width = parser.extract<uint32_t>("width"),
             final_height = parser.extract<uint32_t>("height");

    bool compute_x = (final_width == 0),
         compute_y = (final_height == 0);

    std::unique_ptr<Tempest::Texture> background;
    if(parser.isSet("background"))
    {
        Tempest::Path background_path(parser.extractString("background"));
        background = decltype(background)(Tempest::LoadImage(background_path));
        if(!background)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to open background image");
            return EXIT_FAILURE;
        }

        auto& hdr = background->getHeader();
        final_width = hdr.Width;
        final_height = hdr.Height;
    }

    bool scale_range = parser.isSet("scale-to-range");

    for(size_t image_idx = 0, param_idx = 0; image_idx < image_count; ++image_idx)
    {
        auto& image = images[image_idx];
        auto name = parser.getUnassociatedArgument(param_idx++);
        image.X = parser.extractUnassociatedArgument<float>(param_idx++);
        image.Y = parser.extractUnassociatedArgument<float>(param_idx++);
        image.Texture = decltype(image.Texture)(Tempest::LoadImage(Tempest::Path(name)));
        image.Texture->convertToRGBA();

        if(scale_range)
        {
            auto& hdr = image.Texture->getHeader();
            switch(hdr.Format)
            {
            case Tempest::DataFormat::RGBA32F:
            {
                auto data_ptr = reinterpret_cast<Tempest::Vector4*>(image.Texture->getData());
                float highest_value = Maxf(Maxf(data_ptr->x, data_ptr->y), data_ptr->z);
                for(uint32_t pixel_idx = 1, pixel_idx_end = (uint32_t)hdr.Width*hdr.Height; pixel_idx < pixel_idx_end; ++pixel_idx)
                {
                    auto& vec = data_ptr[pixel_idx];
                    highest_value = Maxf(Maxf(vec.x, vec.y), Maxf(vec.z, highest_value));
                }

                for(uint32_t pixel_idx = 0, pixel_idx_end = (uint32_t)hdr.Width*hdr.Height; pixel_idx < pixel_idx_end; ++pixel_idx)
                {
                    auto& vec = data_ptr[pixel_idx];
                    vec.x /= highest_value;
                    vec.y /= highest_value;
                    vec.z /= highest_value;
                }
            } break;
            }
        }

        image.Texture->convertToUNorm8();

        if(!image.Texture)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to load texture: ", name);
            return EXIT_FAILURE;
        }

        auto& hdr = image.Texture->getHeader();
        if(compute_x)
        {
            final_width = std::max(final_width, (uint32_t)(image.X + hdr.Width));
        }
        if(compute_y)
        {
            final_height = std::max(final_height, (uint32_t)(image.Y + hdr.Height));
        }
    }

    Tempest::EPSImageInfo eps_info;
    eps_info.Width = final_width;
    eps_info.Height = final_height;

    Tempest::EPSDraw eps_draw(eps_info);

    if(background)
    {
        eps_draw.drawImage(*background, 0, 0);
    }

    for(size_t image_idx = 0; image_idx < image_count; ++image_idx)
    {
        auto& image = images[image_idx];
        auto& hdr = image.Texture->getHeader();

        if(count > 1)
        {
            float border = 10.0f;
            Tempest::Rect2 rect{ { image.X + hdr.Width*0.5f, image.Y + hdr.Height*0.5f }, 0, hdr.Width*0.5f + border, hdr.Height*0.5f + border };
            eps_draw.drawRect(rect);
        }

        eps_draw.drawImage(*image.Texture, image.X, image.Y);
    }

    auto output_file = parser.extractString("output");
    status = eps_draw.saveImage(output_file.c_str());
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to save to output file: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
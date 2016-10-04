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
#include "tempest/image/image-utils.hh"
#include "tempest/image/eps-draw.hh"
#include "tempest/math/shapes.hh"

#include <cstdlib>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("image-analyze", true);
    parser.createOption('o', "output", "Specify output image filename", true, "output.exr");
    parser.createOption('h', "buckets", "Specify number of histogram buckets", true, "256");
    parser.createOption('m', "minimum", "Specify minimum of the histogram", true);
    parser.createOption('M', "maximum", "Specify maximum of the histogram", true);
    parser.createOption('b', "background", "Specify background image", true);
    parser.createOption('a', "arrows", "Specify interest arrows location", true);
    parser.createOption('d', "arrow-directions", "Specify interest arrows directions", true);
    parser.createOption('s', "arrow-size", "Specify arrow size in pixels", true, "10");
    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    size_t count = parser.getUnassociatedCount();
    if(count < 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify one input image\n"
                                               "USAGE:\n"
                                               "\timage-analyze [ <options> ] <input>");
        return EXIT_FAILURE;
    }

    auto filename = parser.getUnassociatedArgument(0);

    uint32_t image_width = 0, image_height = 0;

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
        image_width = hdr.Width;
        image_height = hdr.Height;
    }


    bool min_specified = parser.isSet("minimum"),
         max_specified = parser.isSet("maximum");

    float min_value = std::numeric_limits<float>::max(),
          max_value = -std::numeric_limits<float>::max();

    struct ImageData
    {
        float                               X,
                                            Y;
        std::unique_ptr<Tempest::Texture>   Texture;
    };

    size_t image_count = (count + 2) / 3;

    std::unique_ptr<ImageData[]> images(new ImageData[image_count]);

    for(size_t i = 0, param_idx = 0; i < image_count; ++i)
    {
        auto input_file = parser.getUnassociatedArgument(param_idx++);
        Tempest::Path input_file_path(input_file);

        auto& image = images[i];

        image.Texture = decltype(image.Texture)(Tempest::LoadImage(input_file_path));
        if(!image.Texture)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load input texture: ", input_file);
            return EXIT_FAILURE;
        }

        image.Texture->convertToLuminance();

        auto& hdr = image.Texture->getHeader();
        if(!min_specified || !max_specified)
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    float value = image.Texture->fetchRed(x, y);
                    if(max_value < value)
                        max_value = value;
                    if(value < min_value)
                        min_value = value;
                }
        }

        if(count > 1)
        {
            image.X = parser.extractUnassociatedArgument<float>(param_idx++);
            image.Y = parser.extractUnassociatedArgument<float>(param_idx++);
        }
        else
        {
            image.X = image.Y = 0;
        }

        image_width = std::max(image_width, static_cast<uint32_t>(Tempest::FastCeil(image.X + hdr.Width)));
        image_height = std::max(image_height, static_cast<uint32_t>(Tempest::FastCeil(image.Y + hdr.Height)));
    }

    if(min_specified)
        min_value = parser.extract<float>("minimum");
    if(max_specified)
        max_value = parser.extract<float>("maximum");

    if(min_value > max_value)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Minimum should be smaller than maximum");
        return EXIT_FAILURE;
    }

    size_t buckets = parser.extract<size_t>("buckets");
    if(buckets == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify more than zero buckets");
        return EXIT_FAILURE;
    }

    float coef = (buckets - 1)/(max_value - min_value);

    std::unique_ptr<size_t[]> histogram(new size_t[buckets]);
    
    Tempest::EPSImageInfo info;
    info.Width = image_width;
    info.Height = image_height;

    auto output_file = parser.extractString("output");
    Tempest::Path output_path(output_file);
    if(output_path.extension() != "eps")
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported format for this mode: ", output_file);
        return EXIT_FAILURE;
    }

    Tempest::EPSDraw eps_draw(info);

    if(background)
    {
        eps_draw.drawImage(*background, 0, 0);
    }

    for(size_t i = 0, param_idx = 0; i < image_count; ++i)
    {
        std::fill(histogram.get(), histogram.get() + buckets, 0);

        auto& image = images[i];
        auto& hdr = image.Texture->getHeader();

        for(uint32_t y = 0, yend = hdr.Height; y < yend; ++y)
        {
            for(uint32_t x = 0, xend = hdr.Width; x < xend; ++x)
            {
                float value = image.Texture->fetchRed(x, y);
                value -= min_value;
                size_t idx = Tempest::Clamp((unsigned long long)Tempest::FastFloorToInt64(value*coef + 0.5f), 0ULL, buckets - 1ULL);
                ++histogram[idx];
            }
        }

        size_t max_size = 0;
        for(size_t idx = 0; idx < buckets; ++idx)
        {
            size_t value = histogram[idx];
            if(max_size < value)
                max_size = value;
        }

        size_t tex_area = hdr.Width*hdr.Height;
        std::unique_ptr<Tempest::Vector2[]> paths(new Tempest::Vector2[2*buckets + 2]);

        for(size_t image_idx = 0; image_idx < image_count; ++image_idx)
        {
            paths[0] = Tempest::Vector2{ (float)hdr.Width*0/buckets + image.X, image.Y };
            paths[1] = Tempest::Vector2{ (float)hdr.Width*1/buckets + image.X, image.Y };

            for(size_t bucket_idx = 0; bucket_idx < buckets; ++bucket_idx)
            {
                paths[2*bucket_idx + 2] = Tempest::Vector2{ (float)hdr.Width*bucket_idx/buckets + image.X, (float)hdr.Height*histogram[bucket_idx]/max_size + image.Y };
                paths[2*bucket_idx + 3] = Tempest::Vector2{ (float)hdr.Width*(bucket_idx + 1)/buckets + image.X, (float)hdr.Height*histogram[bucket_idx]/max_size + image.Y };
            }

            const float border = 5;
            Tempest::Rect2 rect{ { border + image.X + hdr.Width*0.5f, border + image.Y + hdr.Height*0.5f }, 0, hdr.Width*0.5f + border, hdr.Height*0.5f + border };
            eps_draw.drawRect(rect);
            eps_draw.drawPolygon(paths.get(), 2*buckets, 0, 0.0f, 0);
        }
    }

    if(parser.isSet("arrows"))
    {
        auto arrows_str = parser.extractString("arrows");
        std::vector<Tempest::Vector2> points;
        auto status = Tempest::ParseCommaSeparatedVectors(arrows_str.c_str(), &points);
        if(!status)
            return EXIT_FAILURE;

        auto arrow_dirs_str = parser.extractString("arrow-directions");
        std::vector<Tempest::Vector2> dirs;
        status = Tempest::ParseCommaSeparatedVectors(arrow_dirs_str.c_str(), &dirs);
        if(!status)
            return EXIT_FAILURE;

        if(dirs.size() != points.size())
        {
            Tempest::Log(Tempest::LogLevel::Error, "You must specify the same number arrow directions as arrow locations:\n"
                                                   "NOTE:\n"
                                                   "\tarrows: ", arrows_str, "\n"
                                                   "\tarrow-directions: ", arrow_dirs_str, "\n");
            return EXIT_FAILURE;
        }

        float arrow_size = parser.extract<float>("arrow-size");
        if(arrow_size <= 0.0f)
        {
            Tempest::Log(Tempest::LogLevel::Error, "You must speciy arrow size greater than 0\n",
                                                   "NOTE: arrow-size: ", arrow_size);
            return EXIT_FAILURE;
        }

        Tempest::Vector2 arrow_verts[] =
        {
            {  arrow_size*0.125f,  arrow_size*0.25f }, { arrow_size*0.5f, 0.0f },
            {  arrow_size*0.125f, -arrow_size*0.25f },
            { -arrow_size*0.5f,  0.0f }, { arrow_size*0.5f, 0.0f },
        };

        for(size_t arrow_idx = 0, arrow_end = points.size(); arrow_idx < arrow_end; ++arrow_idx)
        {
            auto translation = points[arrow_idx] + Tempest::Vector2{ 0.0f, 0.0f };
            auto& dir = dirs[arrow_idx];

            Tempest::Matrix2 rot_matrix({ dir.x,  dir.y },
                                        { dir.y, -dir.x });

            Tempest::Vector2 this_arrow_verts[TGE_FIXED_ARRAY_SIZE(arrow_verts)];

            for(uint32_t vert_idx = 0, vert_end = TGE_FIXED_ARRAY_SIZE(arrow_verts); vert_idx < vert_end; ++vert_idx)
            {
                auto v = arrow_verts[vert_idx];
                v = rot_matrix.transform(v);
                v += translation;

                this_arrow_verts[vert_idx] = v;
            }

            {
            eps_draw.drawPath(this_arrow_verts, 3, false, arrow_size*0.125f, 0);
            }

            {
            Tempest::Vector2& v0 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 2];
            Tempest::Vector2& v1 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 1];
            eps_draw.drawLine(v0.x, v0.y, v1.x, v1.y, arrow_size*0.125f, 0);
            }

            for(uint32_t vert_idx = 0, vert_end = TGE_FIXED_ARRAY_SIZE(arrow_verts); vert_idx < vert_end; ++vert_idx)
            {
                this_arrow_verts[vert_idx] += Tempest::Vector2{ -arrow_size*0.03125f, arrow_size*0.03125f };
            }

            {
            eps_draw.drawPath(this_arrow_verts, 3, false, arrow_size*0.125f, 0xFFFFFF);
            }

            {
            Tempest::Vector2& v0 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 2];
            Tempest::Vector2& v1 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 1];
            eps_draw.drawLine(v0.x, v0.y, v1.x, v1.y, arrow_size*0.125f, 0xFFFFFF);
            }
        }
    }

    status = eps_draw.saveImage(output_file.c_str());
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to save output file: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

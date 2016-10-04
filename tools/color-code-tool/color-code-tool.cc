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
#include "tempest/image/image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/eps-draw.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/memory.hh"

#include <cstdlib>

int main(int argc, char* argv[])
{
    const float border = 5;
    const float ruler_x_size = 10;
    const float text_offset = ruler_x_size/2 + 5;
    int font_size = 42;
    float text_size = 60;

    Tempest::CommandLineOptsParser parser("color-code-tool", true);
    parser.createOption('o', "output", "Specify output EPS file", true);
    parser.createOption('b', "background", "Specify background image", true);
    parser.createOption('r', "ruler-samples", "Specify number of samples on the ruler", true, "5");
    parser.createOption('u', "unit", "Specify unit of measurements", true, "dB");
    parser.createOption('i', "invert", "Invert color coding scale", false);
    parser.createOption('m', "minimum", "Specify minimum value", true);
    parser.createOption('M', "maximum", "Specify maximum value", true);
    parser.createOption('a', "arrows", "Specify interest arrows location", true);
    parser.createOption('d', "arrow-directions", "Specify interest arrows directions", true);
    parser.createOption('s', "arrow-size", "Specify arrow size in pixels", true, "10");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto count = parser.getUnassociatedCount();
    if(count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify at least one input file");
        return EXIT_FAILURE;
    }

    if(count % 3 && count != 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid argument count:\n"
                                               "USAGE:\n"
                                               "\timage-combine [ <options> ] <image-file> <x-axis> <y-axis> ...\n");
        return EXIT_FAILURE;
    }

    auto unit = parser.extractString("unit");

    text_size += unit.size()*20.0f;

    struct ImageData
    {
        float                               X,
                                            Y;
        std::unique_ptr<Tempest::Texture>   Texture;
    };

    size_t image_count = (count + 2) / 3;

    std::unique_ptr<ImageData[]> images(new ImageData[image_count]);

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

    float min_value = std::numeric_limits<float>::max(), max_value = 0.0f;

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

        auto& hdr = image.Texture->getHeader();
        for(uint32_t y = 0; y < hdr.Height; ++y)
            for(uint32_t x = 0; x < hdr.Width; ++x)
            {
                float value = image.Texture->fetchRed(x, y);
                if(max_value < value)
                    max_value = value;
                if(value < min_value)
                    min_value = value;
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

    if(parser.isSet("minimum"))
    {
        min_value = Maxf(min_value, parser.extract<float>("minimum"));
    }

    if(parser.isSet("maximum"))
    {
        max_value = Minf(max_value, parser.extract<float>("maximum"));
    }

    float order = powf(10.0f, ceilf(log10f(max_value)));
    min_value = FastFloor(20*min_value/order)*order/20;
    max_value = Tempest::FastCeil(20*max_value/order)*order/20;

    Tempest::EPSImageInfo image_info;
    image_info.Width = Tempest::AlignAddress((uint32_t)(image_width + 3*border + ruler_x_size + text_offset + text_size), 4u);
    image_info.Height = Tempest::AlignAddress((uint32_t)(image_height + 2*border), 4u);

    Tempest::TextureDescription scale_tex_desc;
    scale_tex_desc.Width = 1;
    scale_tex_desc.Height = image_height;
    scale_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    auto scale_tex_data = new uint32_t[scale_tex_desc.Width*scale_tex_desc.Height];
    Tempest::Texture scale_tex(scale_tex_desc, reinterpret_cast<uint8_t*>(scale_tex_data));

    bool invert_color_code = parser.isSet("invert");
    
    if(invert_color_code)
    {
        for(uint32_t h = 0; h < scale_tex_desc.Height; ++h)
        {
            scale_tex_data[h] = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB((float)h/(image_height - 1)));
        }
    }
    else
    {
        for(uint32_t h = 0; h < scale_tex_desc.Height; ++h)
        {
            scale_tex_data[h] = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB(1.0f - (float)h/(image_height - 1)));
        }
    }

    Tempest::EPSDraw eps_draw(image_info);

    float x_start_ruler = image_width + 2*border + ruler_x_size,
          x_end_ruler = x_start_ruler + ruler_x_size/2;

    if(background)
    {
        eps_draw.drawImage(*background, border, border);
    }

    for(size_t i = 0; i < image_count; ++i)
    {
        auto& image = images[i];

        float range = max_value - min_value;

        auto& hdr = image.Texture->getHeader();

        Tempest::TextureDescription tex_desc;
        tex_desc.Width = hdr.Width;
        tex_desc.Height = hdr.Height;
        tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

        uint32_t* data = new uint32_t[tex_desc.Width*tex_desc.Height];

        Tempest::Texture result_image(tex_desc, reinterpret_cast<uint8_t*>(data));

        if(invert_color_code)
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    float ratio = (image.Texture->fetchRed(x, y) - min_value)/range;
                    data[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB(ratio));
                }
        }
        else
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    float ratio = (image.Texture->fetchRed(x, y) - min_value)/range;
                    data[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB(1.0f - ratio));
                }
        }

        if(count > 1)
        {
            Tempest::Rect2 rect{ { border + image.X + hdr.Width*0.5f, border + image.Y + hdr.Height*0.5f }, 0, hdr.Width*0.5f + border, hdr.Height*0.5f + border };
            eps_draw.drawRect(rect);
        }

        eps_draw.drawImage(result_image, border + image.X, border + image.Y);
    }
        
    eps_draw.drawImage(scale_tex, image_width + 2*border, border, ruler_x_size);

    Tempest::Vector2 border_ruler[] =
    {
        { image_width + 2*border, border },
        { image_width + 2*border + ruler_x_size, border },
        { image_width + 2*border + ruler_x_size, image_height + border },
        { image_width + 2*border, image_height + border },
    };

    eps_draw.drawPath(border_ruler, TGE_FIXED_ARRAY_SIZE(border_ruler), true, 1);
    
    Tempest::Vector2 border_image[] =
    {
        { border, border },
        { image_width + border, border },
        { image_width + border, image_height + border },
        { border, image_height + border }
    };
    
    eps_draw.drawPath(border_image, TGE_FIXED_ARRAY_SIZE(border_image), true, 1);

    for(size_t sample_idx = 0, sample_count = parser.extract<size_t>("ruler-samples"); sample_idx < sample_count; ++sample_idx)
    {
        std::stringstream ss;
        float ruler_value = min_value + (max_value - min_value)*sample_idx/(sample_count - 1);
        ss << ruler_value << " " << unit;
        auto min_str = ss.str();

        float mark_height = border + image_height*sample_idx/(sample_count - 1);
        float text_height = mark_height;
        if(sample_idx == sample_count - 1)
            text_height -= font_size*0.75f;
        else if(sample_idx != 0)
            text_height -= font_size*0.75f*0.5f;

        eps_draw.drawLine(x_start_ruler, mark_height, x_end_ruler, mark_height, 1);
        eps_draw.drawText(min_str.c_str(), font_size, image_width + 2*border + ruler_x_size + text_offset, text_height);
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
            auto translation = points[arrow_idx] + Tempest::Vector2{ border, border };
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

    std::string output_file;
    if(parser.isSet("output"))
    {
        output_file = parser.extractString("output");
    }
    else
    {
        auto input_file = parser.getUnassociatedArgument(0);
        Tempest::Path input_file_path(input_file);
        auto filename_wo_ext = input_file_path.filenameWOExt();
        auto out_dir = input_file_path.directoryPath();
        if(!out_dir.empty())
            out_dir += "/";

        output_file = out_dir + filename_wo_ext + ".eps";
    }
    
    status = eps_draw.saveImage(output_file.c_str());
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to save to output file: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#include "tempest/utils/parse-command-line.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/image/eps-draw.hh"
#include "tempest/math/shapes.hh"
#include "tempest/graphics/sampling-wrapper.hh"

enum class VectorType
{
    Tangent,
    Binormal,
    Normal,
};

std::unique_ptr<Tempest::Texture> ConvertBasisToVectorTexture(const Tempest::Texture* input_texture, VectorType vec_type)
{
    auto& hdr = input_texture->getHeader();

    Tempest::TextureDescription out_tex_desc;
    out_tex_desc.Width = hdr.Width;
    out_tex_desc.Height = hdr.Height;
    out_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    uint32_t* out_data(new uint32_t[out_tex_desc.Width*out_tex_desc.Height]);
    std::unique_ptr<Tempest::Texture> output_texture(new Tempest::Texture(out_tex_desc, reinterpret_cast<uint8_t*>(out_data)));

    switch(vec_type)
    {
    case VectorType::Normal:
    {
        for(uint32_t y = 0; y < out_tex_desc.Height; ++y)
            for(uint32_t x = 0; x < out_tex_desc.Width; ++x)
            {
                Tempest::Quaternion quat;
                quat.V4 = input_texture->fetchRGBA(x, y);
                auto normal = Tempest::ToNormal(quat)*0.5f + 0.5f;
                out_data[y*hdr.Width + x] = Tempest::ToColor(normal);
            }
    } break;
    case VectorType::Tangent:
    {
        for(uint32_t y = 0; y < out_tex_desc.Height; ++y)
            for(uint32_t x = 0; x < out_tex_desc.Width; ++x)
            {
                Tempest::Quaternion quat;
                quat.V4 = input_texture->fetchRGBA(x, y);
                auto normal = Tempest::ToTangent(quat)*0.5f + 0.5f;
                out_data[y*hdr.Width + x] = Tempest::ToColor(normal);
            }
    } break;
    case VectorType::Binormal:
    {
        for(uint32_t y = 0; y < out_tex_desc.Height; ++y)
            for(uint32_t x = 0; x < out_tex_desc.Width; ++x)
            {
                Tempest::Quaternion quat;
                quat.V4 = input_texture->fetchRGBA(x, y);
                auto normal = Tempest::ToBinormal(quat)*0.5f + 0.5f;
                out_data[y*hdr.Width + x] = Tempest::ToColor(normal);
            }
    } break;
    }

    return std::move(output_texture);
}

int main(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("quaternion-texture-tool", true);
    parser.createOption('e', "extract", "Extract information from encoded quaternion image (normal, tangent, binormal)", true, "normal");
    parser.createOption('f', "render-flow", "Computes the tangent flow for this component. Note that this outputs eps files.", false);
    parser.createOption('o', "output", "Specify output file (<name>_<component>.png)", true);
    parser.createOption('b', "background", "Specify background image", true);
    parser.createOption('m', "mix-texture", "Specify texture to mix with base textures", true);
    parser.createOption('M', "mix-ratio", "Specify mixing ratio between mix texture and base textures", true, "0.5");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "quaternion-texture-tool: error: input file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tquaternion-texture-tool [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }

    if(unassoc_count % 3 && unassoc_count != 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid argument count:\n"
                                               "USAGE:\n"
                                               "\tquaternion-texture-tool [ <options> ] <image-file> <x-axis> <y-axis> ...\n");
        return EXIT_FAILURE;
    }

    struct ImageData
    {
        float                               X,
                                            Y;
        std::unique_ptr<Tempest::Texture>   Texture;
    };

    size_t image_count = (unassoc_count + 2) / 3;

    std::unique_ptr<ImageData[]> images(new ImageData[image_count]);
    
    uint32_t image_width = 0,
             image_height = 0;

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
        if(unassoc_count > 1)
        {
            image.X = parser.extractUnassociatedArgument<float>(param_idx++);
            image.Y = parser.extractUnassociatedArgument<float>(param_idx++);
        }
        else
        {
            image.X = image.Y = 0;
        }

        uint32_t chan_count = Tempest::DataFormatChannels(hdr.Format);
        if(4 != chan_count)
        {
            Tempest::Log(Tempest::LogLevel::Error, "invalid format. Input texture (", input_file, ") has ", chan_count, " channels per pixel. While quaternion require 4.");
            return EXIT_FAILURE;
        }

        image_width = std::max(image_width, static_cast<uint32_t>(Tempest::FastCeil(image.X + hdr.Width)));
        image_height = std::max(image_height, static_cast<uint32_t>(Tempest::FastCeil(image.Y + hdr.Height)));
    }

    if(parser.isSet("mix-texture"))
    {
        auto mix_tex_name = parser.extractString("mix-texture");
        auto t_mix = parser.extract<float>("mix-ratio");
        if(0.0f > t_mix || t_mix > 1.0f)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Invalid mix ratio (allowed range is 0.0 ... 1.0): ", t_mix);
            return EXIT_FAILURE;
        }

        std::unique_ptr<Tempest::Texture> mix_tex(Tempest::LoadImage(Tempest::Path(mix_tex_name)));
        if(!mix_tex)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load mix texture: ", mix_tex_name);
            return EXIT_FAILURE;
        }

        auto& mix_hdr = mix_tex->getHeader();

        for(size_t image_idx = 0; image_idx < image_count; ++image_idx)
        {
            auto& image = images[image_idx];
            auto& hdr = image.Texture->getHeader();
            auto image_data_ptr = image.Texture.get();

            if(hdr.Width != mix_hdr.Width ||
               hdr.Height != mix_hdr.Height)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Mismatching format between textures: ", mix_tex_name, "; ", parser.getUnassociatedArgument(image_idx*3));
                return EXIT_FAILURE;
            }

            switch(hdr.Format)
            {
            case Tempest::DataFormat::RGBA32F:
            {
                for(uint32_t y = 0, yend = hdr.Height; y < yend; ++y)
                    for(uint32_t x = 0, xend = hdr.Width; x < xend; ++x)
                    {
                        Tempest::Quaternion sample0, sample1;
                        sample0.V4 = Tempest::FetchRGBA(image_data_ptr, x, y);
                        sample1.V4 = Tempest::FetchRGBA(mix_tex.get(), x, y);

                        auto sample = Tempest::Slerp(sample0, sample1, t_mix);

                    #ifndef NDEBUG
                        auto v0 = Tempest::ToTangent(sample), 
                             v1 = Tempest::ToTangent(sample0),
                             v2 = Tempest::ToTangent(sample1);

                        TGE_ASSERT(Tempest::Dot(v0, v1) >= 0.0f ||
                                   Tempest::Dot(v0, v2) >= 0.0f, "Rubbish interpolation");
                    #endif

                        Tempest::Surface2DWrite(sample, image_data_ptr, x*sizeof(Tempest::Vector4), y);
                    }
            } break;
            default:
                Tempest::Log(Tempest::LogLevel::Error, "Unsupported basis format");
                return EXIT_FAILURE;
            }
        }
    }


    auto operation = parser.extractString("extract");

    VectorType vec_type;

    if(operation == "normal")
    {
        vec_type = VectorType::Normal;
    }
    else if(operation == "tangent")
    {
        vec_type = VectorType::Tangent;
    }
    else if(operation == "binormal")
    {
        vec_type = VectorType::Binormal;
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "unsupported quaternion extract operation: ", operation);
        return EXIT_FAILURE;
    }

    auto enable_flow = parser.isSet("render-flow");

    bool enable_eps_image = image_count > 1 || background || enable_flow;

    std::string extension = enable_eps_image ? ".eps" : ".png";

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

        output_file = out_dir + filename_wo_ext + "_" + operation + extension;
    }

    if(enable_eps_image)
    {
        Tempest::EPSImageInfo eps_info;
        eps_info.Width = image_width;
        eps_info.Height = image_height;

        Tempest::EPSDraw eps_draw(eps_info);

        if(background)
        {
            eps_draw.drawImage(*background);
        }

        for(size_t image_idx = 0; image_idx < image_count; ++image_idx)
        {
            auto& image = images[image_idx];

            auto* input_texture = image.Texture.get();

            auto output_texture = ConvertBasisToVectorTexture(input_texture, vec_type);

            auto& out_tex_desc = output_texture->getHeader();

            float border = 5.0f;
            Tempest::Rect2 rect{ { image.X + out_tex_desc.Width*0.5f, image.Y + out_tex_desc.Height*0.5f }, 0, out_tex_desc.Width*0.5f + border, out_tex_desc.Height*0.5f + border };
            eps_draw.drawRect(rect);

            eps_draw.drawImage(*output_texture, image.X, image.Y);

            

            if(enable_flow)
            {
                auto ext = Tempest::Path(output_file).extension();
                if(ext != "eps")
                {
                    Tempest::Log(Tempest::LogLevel::Error, "invalid output file format selected. EPS is only supported for outputting tangent flows");
                    return EXIT_FAILURE;
                }

                const int skip_pixels = 48;
                const int arrow_size = 32;

                Tempest::Vector2 arrow_verts[] =
                {
                    {  arrow_size*0.125f,  arrow_size*0.25f }, { arrow_size*0.5f, 0.0f },
                    {  arrow_size*0.125f, -arrow_size*0.25f },
                    { -arrow_size*0.5f,  0.0f }, { arrow_size*0.5f, 0.0f },
                };


                for(uint32_t y = (out_tex_desc.Height/2 - skip_pixels/2) % skip_pixels; y < out_tex_desc.Height; y += skip_pixels)
                {
                    for(uint32_t x = (out_tex_desc.Width/2 - skip_pixels/2) % skip_pixels; x < out_tex_desc.Width; x += skip_pixels)
                    {
                        Tempest::Vector2 translation{ static_cast<float>(x) + image.X, static_cast<float>(y) + image.Y };

                        Tempest::Quaternion quat;
                        quat.V4 = input_texture->fetchRGBA(x, y);

                        Tempest::Vector3 dir;
                        switch(vec_type)
                        {
                        case VectorType::Tangent:
                        {
                            dir = Tempest::ToTangent(quat);
                        } break;
                        case VectorType::Binormal:
                        {
                            dir = Tempest::ToBinormal(quat);
                        } break;
                        case VectorType::Normal:
                        {
                            dir = Tempest::ToNormal(quat);
                        } break;
                        }

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
                        eps_draw.drawPath(this_arrow_verts, 3, false, 2.0f, 0);
                        }

                        {
                        Tempest::Vector2& v0 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 2];
                        Tempest::Vector2& v1 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 1];
                        eps_draw.drawLine(v0.x, v0.y, v1.x, v1.y, 2.0f, 0);
                        }

                        for(uint32_t vert_idx = 0, vert_end = TGE_FIXED_ARRAY_SIZE(arrow_verts); vert_idx < vert_end; ++vert_idx)
                        {
                            this_arrow_verts[vert_idx] += Tempest::Vector2{ -0.5f, 0.5f };
                        }

                        {
                        eps_draw.drawPath(this_arrow_verts, 3, false, 2.0f, 0xFFFFFF);
                        }

                        {
                        Tempest::Vector2& v0 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 2];
                        Tempest::Vector2& v1 = this_arrow_verts[TGE_FIXED_ARRAY_SIZE(this_arrow_verts) - 1];
                        eps_draw.drawLine(v0.x, v0.y, v1.x, v1.y, 2.0f, 0xFFFFFF);
                        }
                    }
                }
            }
        }

        auto status = eps_draw.saveImage(output_file.c_str());
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to save output file: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else
    {
        auto output_texture = ConvertBasisToVectorTexture(images[0].Texture.get(), vec_type);

        status = Tempest::SaveImage(output_texture->getHeader(), output_texture->getData(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to save output file: ", output_file);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
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

#include "tempest/image/btf.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/hdr.hh"
#include "tempest/utils/threads.hh"
#include "tempest/utils/video-encode.hh"

const int64_t FPS = 30;

enum class ConvertType
{
    HDR,
    LDR,
    SRGB,
    Linear
};

bool ExtractMaxLight(const Tempest::BTF* btf, const std::string& extract, uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end, uint32_t view_idx,
                     const Tempest::TextureDescription& tex_desc, Tempest::Vector3* tex_data)
{
    if(extract == "max-light")
    {
        for(uint32_t btf_y = y_start; btf_y < y_end; ++btf_y)
        {
            for(uint32_t btf_x = x_start; btf_x < x_end; ++btf_x)
            {
                float max_lum = 0.0f;
                Tempest::Spectrum max_spec{};
                for(uint32_t btf_light_idx = 0; btf_light_idx < btf->LightCount; ++btf_light_idx)
                {
                    auto spec = Tempest::BTFFetchSpectrum(btf, btf_light_idx, view_idx, btf_x, btf_y);
                    float lum = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec));
                    if(max_lum < lum)
                    {
                        max_spec = spec;
                        max_lum = lum;
                    }
                }
                tex_data[(btf_y - y_start)*tex_desc.Width + btf_x - x_start] = Tempest::SpectrumToRGB(max_spec);
            }
        }
    }
    else if(extract == "max-light-vector")
    {
        for(uint32_t btf_y = y_start; btf_y < y_end; ++btf_y)
        {
            for(uint32_t btf_x = x_start; btf_x < x_end; ++btf_x)
            {
                float max_lum = 0.0f;
                Tempest::Vector3 max_dir{};
                for(uint32_t btf_light_idx = 0; btf_light_idx < btf->LightCount; ++btf_light_idx)
                {
                    auto spec = Tempest::BTFFetchSpectrum(btf, btf_light_idx, view_idx, btf_x, btf_y);
                    float lum = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec));
                    if(max_lum < lum)
                    {
                        max_dir = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[btf_light_idx]);
                        max_lum = lum;
                    }
                }
                tex_data[(btf_y - y_start)*tex_desc.Width + btf_x - x_start] = Tempest::SpectrumToRGB(max_dir)*0.5f + 0.5f;
            }
        }
    }
    else
        return false;

    return true;
}

int main(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("btf-slice", true);
    parser.createOption('l', "light-index", "Specify light index used to create light-view slice", true);
    parser.createOption('v', "view-index", "Specify view index used to create light-view slice", true);
    parser.createOption('L', "light-direction", "Specify light direction used to create interpolated light-view slice (example: 0:0:1)", true);
    parser.createOption('V', "view-direction", "Specify view direction used to create interpolated light-view slice (example: 0:0:1)", true);
    parser.createOption('f', "format", "Specify format (rgba8, luminance8)", true, "rgba8");
    parser.createOption('o', "output", "Specify output file (default: \"btf-slice.tga\")", true, "btf-slice.tga");
	parser.createOption('X', "x-start", "Specify starting texture coordinate on X-axis", true, "0");
	parser.createOption('Y', "y-start", "Specify starting texture coordinate on Y-axis", true, "0");
	parser.createOption('W', "width", "Specify width of the sample image", true);
	parser.createOption('H', "height", "Specify height of the sample image", true);
    parser.createOption('w', "light-video", "Specify list of points to traverse while recording a video", true);
    parser.createOption('I', "light-video-image", "Specify that you want video to be saved as images instead", false);
    parser.createOption('E', "export-light", "Export all light directions", false);
    parser.createOption('a', "angular-velocity", "Specify angular velocity used for traversing the points", true, "1");
    parser.createOption('e', "extract", "Specify special extract mode (max-light, max-light-vector)", true, "none");
    parser.createOption('c', "convert", "Conversion options (hdr, srgb, ldr, linear)", true, "hdr");

	if(!parser.parse(argc, argv))
	{
		return EXIT_FAILURE;
	}

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-slice: error: input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-slice [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-slice: error: too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-slice [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }

    auto input_filename = parser.getUnassociatedArgument(0);
	Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(input_filename)));
    if(!btf)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load btf file: ", input_filename);
        return EXIT_FAILURE;
    }

    uint32_t btf_width = btf->Width,
             btf_height = btf->Height,
             btf_light_count = btf->LightCount,
             btf_view_count = btf->LightCount;

    bool light_dir_set = parser.isSet("light-direction"),
         view_dir_set = parser.isSet("view-direction"),
         light_idx_set = parser.isSet("light-index"),
         view_idx_set = parser.isSet("view-index");

	auto x_start = parser.extract<uint32_t>("x-start"),
         y_start = parser.extract<uint32_t>("y-start");

    if(x_start >= btf->Width)
    {
        Tempest::Log(Tempest::LogLevel::Error, "out of bounds X starting value specified: ", x_start);
        return EXIT_FAILURE;
    }

    if(y_start >= btf->Height)
    {
        Tempest::Log(Tempest::LogLevel::Error, "out of bounds Y starting value specified: ", y_start);
        return EXIT_FAILURE;
    }

    auto x_end = btf->Width,
         y_end = btf->Height;

	if(parser.isSet("width"))
    {
        x_end = x_start + parser.extract<uint32_t>("width");
    }

    if(parser.isSet("height"))
    {
        y_end = y_start + parser.extract<uint32_t>("height");
    }

    if(x_end > btf->Width || y_end > btf->Height)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Specified texture area is out of bounds: (", x_start, ", ", y_start, ") -- (", x_end, ", ", y_end, ")");
        return EXIT_FAILURE;
    }

	Tempest::TextureDescription tex_desc;
    tex_desc.Width = x_end - x_start;
    tex_desc.Height = y_end - y_start;
    tex_desc.Format = Tempest::DataFormat::RGB32F;
    Tempest::Vector3* tex_data(new Tempest::Vector3[tex_desc.Width*tex_desc.Height]);
    Tempest::Texture tex(tex_desc, reinterpret_cast<uint8_t*>(tex_data));

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    if(parser.isSet("light-video"))
    {
        std::vector<Tempest::Vector3> light_dirs;
        auto light_dirs_str = parser.extractString("light-video");
        auto status = Tempest::ParseCommaSeparatedVectors(light_dirs_str.c_str(), &light_dirs); 
        if(!status)
        {
            return EXIT_FAILURE;
        }

        Tempest::VideoInfo video_info;
        video_info.FileName = parser.extractString("output");
        video_info.Width = tex_desc.Width;
        video_info.Height = tex_desc.Height;
        video_info.FPS = 30;
        video_info.Bitrate = 50000;

        bool light_video_image = parser.isSet("light-video-image");

        std::unique_ptr<Tempest::VPXVideoEncoder> video_enc;
        if(!light_video_image)
        {
            video_enc = std::unique_ptr<Tempest::VPXVideoEncoder>(new Tempest::VPXVideoEncoder());
            status = video_enc->openStream(video_info);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Failed to open video encoding stream: ", video_info.FileName);
                return EXIT_FAILURE;
            }
        }

        if(light_dirs.size() <= 1)
        {
            Tempest::Log(Tempest::LogLevel::Error, "You must specify at least 2 points to make light animation");
            return EXIT_FAILURE;
        }

        float angular_velocity = parser.extract<float>("angular-velocity");

        float lower_bound_angle = 0.0f;
        float upper_bound_angle = acosf(Tempest::Dot(light_dirs[0], light_dirs[1]));
        size_t idx = 0;

        Tempest::Vector3 view_dir;
            
        if(view_dir_set)
        {
            auto view_str = parser.extractString("view-direction");
            status = Tempest::ParseDirection(view_str.c_str(), &view_dir);
            if(!status)
            {
                return EXIT_FAILURE;
            }
        }
        else
        {
            uint32_t view_idx = 0;
            if(view_idx_set)
            {
                view_idx = parser.extract<uint32_t>("view-index");
            }

            if(view_idx >= btf_view_count)
            {
                Tempest::Log(Tempest::LogLevel::Error, "out of bounds light or view index: ", view_idx);
                return EXIT_FAILURE;
            }

            view_dir = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);
        }

        auto convert_options = parser.extractString("convert");
        ConvertType convert_type;
        if(convert_options == "srgb")
        {
            convert_type = ConvertType::SRGB;
        }
        else if(convert_options == "hdr")
        {
            convert_type = ConvertType::HDR;
        }
        else if(convert_options == "ldr")
        {
            convert_type = ConvertType::LDR;
        }
        else
        {
            Tempest::Log(Tempest::LogLevel::Error, "unsupported conversion options: ", convert_options);
            return EXIT_FAILURE;
        }

        for(int64_t cur_time = 0;; cur_time += 1000000ULL/FPS)
        {
            float cur_angle = cur_time*1e-6f*angular_velocity;
            if(cur_angle > upper_bound_angle)
            {
                if(++idx == light_dirs.size() - 1)
                    break;
                lower_bound_angle = upper_bound_angle;
                upper_bound_angle += acosf(Tempest::Dot(light_dirs[idx], light_dirs[idx + 1]));
                continue;
            }

            float span = upper_bound_angle - lower_bound_angle;
            float t = (cur_angle - lower_bound_angle)/span;

            auto& lower_vec = light_dirs[idx];
            auto& upper_vec = light_dirs[idx + 1];

            auto rot_quat = Tempest::Slerp(Tempest::FastRotationBetweenVectorQuaternion(lower_vec, upper_vec), t);

            auto light_dir = Tempest::Transform(rot_quat, lower_vec);
            
            uint32_t light_prim_id, view_prim_id;
            Tempest::Vector3 light_barycentric, view_barycentric;
            status = Tempest::BTFFetchLightViewDirection(btf.get(), light_dir, view_dir, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "failed to intersect BTF geometry. TODO!!! Implement grazing angle BTF");
                return EXIT_FAILURE;
            }

            auto btf_ptr = btf.get();
            auto process = Tempest::CreateParallelForLoop2D(x_end - x_start, y_end - y_start, 64,
                                [tex_data, x_start, y_start, &tex_desc, btf_ptr, light_prim_id, light_barycentric, view_prim_id, view_barycentric](uint32_t worker_id, uint32_t x, uint32_t y)
                                {
                                    auto spec = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, light_prim_id, light_barycentric, view_prim_id, view_barycentric, x_start + x, y_start + y);
                                    tex_data[y*tex_desc.Width + x] = Tempest::SpectrumToRGB(spec);
                                });
            pool.enqueueTask(&process);
            pool.waitAndHelp(id, &process);

            if(video_enc)
            {
                Tempest::TexturePtr out_tex;

                switch(convert_type)
                {
                case ConvertType::SRGB:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRSRGB(id, pool, &tex, 64));
                } break;
                case ConvertType::HDR:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToSRGB(id, pool, &tex, 64));
                } break;
                case ConvertType::LDR:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRRGB(id, pool, &tex, 64));
                } break;
                }

                video_enc->submitFrame(*out_tex);
            }
            else
            {
                std::stringstream ss;
                ss << "-light-" << light_dir.x << "-" << light_dir.y << "-" << light_dir.z;
                auto name = ss.str();
                std::replace(std::begin(name), std::end(name), '.', '_');
                name = video_info.FileName + name + ".exr";

                status = Tempest::SaveImage(tex_desc, tex_data, Tempest::Path(name));
                if(!status)
                {
                    Tempest::Log(Tempest::LogLevel::Error, "failed to save image: ", name);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

    if(parser.isSet("export-light"))
    {
        uint32_t final_view_idx = 0;
        if(view_dir_set)
        {
            Tempest::Vector3 view_dir;
            auto view_str = parser.extractString("view-direction");
            auto status = Tempest::ParseDirection(view_str.c_str(), &view_dir);
            if(!status)
            {
                return EXIT_FAILURE;
            }

            final_view_idx = BTFNearestAngle(btf.get(), view_dir);
        }
        else
        {
            if(view_idx_set)
            {
                final_view_idx = parser.extract<uint32_t>("view-index");
            }

            if(final_view_idx >= btf_view_count)
            {
                Tempest::Log(Tempest::LogLevel::Error, "out of bounds light or view index: ", final_view_idx);
                return EXIT_FAILURE;
            }
        }

        auto convert_options = parser.extractString("convert");
        ConvertType convert_type;
        if(convert_options == "srgb")
        {
            convert_type = ConvertType::SRGB;
        }
        else if(convert_options == "hdr")
        {
            convert_type = ConvertType::HDR;
        }
        else if(convert_options == "ldr")
        {
            convert_type = ConvertType::LDR;
        }
        else if(convert_options == "linear")
        {
            convert_type = ConvertType::Linear;
        }
        else
        {
            Tempest::Log(Tempest::LogLevel::Error, "unsupported conversion options: ", convert_options);
            return EXIT_FAILURE;
        }

        auto output = parser.extractString("output");

        Tempest::Log(Tempest::LogLevel::Debug, "view direction: ", Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[final_view_idx]));

        for(uint32_t light_idx = 0; light_idx < btf->LightCount; ++light_idx)
        {
            auto light_dir = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
            
            auto btf_ptr = btf.get();
            auto process = Tempest::CreateParallelForLoop2D(x_end - x_start, y_end - y_start, 64,
                                [tex_data, x_start, y_start, &tex_desc, btf_ptr, light_idx, final_view_idx](uint32_t worker_id, uint32_t x, uint32_t y)
                                {
                                    auto spec = Tempest::BTFFetchSpectrum(btf_ptr, light_idx, final_view_idx, x_start + x, y_start + y);
                                    tex_data[y*tex_desc.Width + x] = Tempest::SpectrumToRGB(spec);
                                });
            pool.enqueueTask(&process);
            pool.waitAndHelp(id, &process);

            std::stringstream ss;
            ss << "-light-" << light_dir.x << "-" << light_dir.y << "-" << light_dir.z;
            auto name = ss.str();
            std::replace(std::begin(name), std::end(name), '.', '_');

            if(convert_type == ConvertType::Linear)
            {
                name = output + name + ".exr";

                auto status = Tempest::SaveImage(tex_desc, tex_data, Tempest::Path(name));
                if(!status)
                {
                    Tempest::Log(Tempest::LogLevel::Error, "failed to save image: ", name);
                    return EXIT_FAILURE;
                }
            }
            else
            {
                Tempest::TexturePtr out_tex;

                switch(convert_type)
                {
                case ConvertType::SRGB:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRSRGB(id, pool, &tex, 64));
                } break;
                case ConvertType::HDR:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToSRGB(id, pool, &tex, 64));
                } break;
                case ConvertType::LDR:
                {
                    out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRRGB(id, pool, &tex, 64));
                } break;
                }

                name = output + name + ".png";

                auto status = Tempest::SaveImage(out_tex->getHeader(), out_tex->getData(), Tempest::Path(name));
                if(!status)
                {
                    Tempest::Log(Tempest::LogLevel::Error, "failed to save image: ", name);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

    auto extract = parser.extractString("extract");

    if(light_dir_set || view_dir_set)
    {
        if(!view_dir_set)
        {
            Tempest::Log(Tempest::LogLevel::Error, "you must specify view(-V) direction");
            return EXIT_FAILURE;
        }

        if(light_idx_set || view_idx_set)
        {
            Tempest::Log(Tempest::LogLevel::Error, "mixed light direction and indexing mode is unsupported");
            return EXIT_FAILURE;
        }

        auto view_dir_str = parser.extractString("view-direction");

        Tempest::Vector3 view_dir;

        auto status = ParseDirection(view_dir_str.c_str(), &view_dir);
        if(!status)
        {
            return EXIT_FAILURE;
        }
        Tempest::NormalizeSelf(&view_dir);

        if(extract == "none")
        {
            if(!light_dir_set)
            {
                Tempest::Log(Tempest::LogLevel::Error, "you must specify light(-L) direction");
                return EXIT_FAILURE;
            }

            Tempest::Vector3 light_dir;
            auto light_dir_str = parser.extractString("light-direction");

             auto status = ParseDirection(light_dir_str.c_str(), &light_dir);
            if(!status)
            {
                return EXIT_FAILURE;
            }
            Tempest::NormalizeSelf(&light_dir);


            uint32_t light_prim_id, view_prim_id;
            Tempest::Vector3 light_barycentric, view_barycentric;
            status = Tempest::BTFFetchLightViewDirection(btf.get(), light_dir, view_dir, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "failed to intersect BTF geometry. TODO!!! Implement grazing angle BTF");
                return EXIT_FAILURE;
            }

            for(uint32_t btf_y = y_start; btf_y < y_end; ++btf_y)
            {
                for(uint32_t btf_x = x_start; btf_x < x_end; ++btf_x)
                {
                    auto spec = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf.get(), light_prim_id, light_barycentric, view_prim_id, view_barycentric, btf_x, btf_y);
                    tex_data[(btf_y - y_start)*tex_desc.Width + btf_x - x_start] = Tempest::SpectrumToRGB(spec);
                }
            }
        }
        else
        {
            auto nearest_idx = Tempest::BTFNearestAngle(btf.get(), view_dir);

            status = ExtractMaxLight(btf.get(), extract, x_start, y_start, x_end, y_end, nearest_idx, tex_desc, tex_data);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Unsupported extraction mode: ", extract);
                return EXIT_FAILURE;
            }
        }
    }
    else
    {
        uint32_t light_idx = 0, view_idx = 0;
		if(light_idx_set)
			light_idx = parser.extract<uint32_t>("light-index");
		if(view_idx_set)
            view_idx = parser.extract<uint32_t>("view-index");

        if(light_idx >= btf_light_count ||
           view_idx >= btf_view_count)
        {
            Tempest::Log(Tempest::LogLevel::Error, "out of bounds light or view index: ", light_idx, ", ", view_idx);
            return EXIT_FAILURE;
        }

        if(extract == "none")
        {
            for(uint32_t btf_y = y_start; btf_y < y_end; ++btf_y)
            {
                for(uint32_t btf_x = x_start; btf_x < x_end; ++btf_x)
                {
                    auto spec = Tempest::BTFFetchSpectrum(btf.get(), light_idx, view_idx, btf_x, btf_y);
                    tex_data[(btf_y - y_start)*tex_desc.Width + btf_x - x_start] = Tempest::SpectrumToRGB(spec);
                }
            }
        }
        else
        {
            auto status = ExtractMaxLight(btf.get(), extract, x_start, y_start, x_end, y_end, view_idx, tex_desc, tex_data);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Unsupported extraction mode: ", extract);
                return EXIT_FAILURE;
            }
        }
    }

    std::unique_ptr<Tempest::Texture> out_tex;

    auto format_str = parser.extractString("format");
    auto output = parser.extractString("output");
    auto convert_options = parser.extractString("convert");
    if(format_str == "rgba8")
    {    
        if(convert_options == "srgb")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRSRGB(id, pool, &tex, 64));
        }
        else if(convert_options == "hdr")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToSRGB(id, pool, &tex, 64));
        }
        else if(convert_options == "ldr")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRRGB(id, pool, &tex, 64));
        }
        else if(convert_options == "linear")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertToLinearRGB(id, pool, &tex, 64));
        }
        else
        {
            Tempest::Log(Tempest::LogLevel::Error, "unsupported conversion options: ", convert_options);
            return EXIT_FAILURE;
        }

        if(!out_tex)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to convert slice to BTF");
            return EXIT_FAILURE;
        }

        auto status = Tempest::SaveImage(out_tex->getHeader(), out_tex->getData(), Tempest::Path(output));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed save image to file: ", output);
            return EXIT_FAILURE;
        }
    }
    else if(format_str == "luminance8")
    {
        if(convert_options == "srgb")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLDRLuminance8(id, pool, &tex, 64));
        }
        else if(convert_options == "hdr")
        {
            out_tex = decltype(out_tex)(Tempest::ParallelConvertHDRToLuminance8(id, pool, &tex, 64));
        }
        else
        {
            Tempest::Log(Tempest::LogLevel::Error, "unsupported conversion options: ", convert_options);
            return EXIT_FAILURE;
        }

        if(!out_tex)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to convert slice to BTF");
            return EXIT_FAILURE;
        }
        
        auto status = Tempest::SaveImage(out_tex->getHeader(), out_tex->getData(), Tempest::Path(output));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed save image to file: ", output);
            return EXIT_FAILURE;
        }
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "unsupported output format: ", format_str);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
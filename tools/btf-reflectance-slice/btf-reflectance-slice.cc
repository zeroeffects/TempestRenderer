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

#include "tempest/image/btf.hh"
#include "tempest/utils/threads.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/math/spherical-harmonics.hh"
#include "tempest/math/simple-basis.hh"

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("btf-reflectance-slice", true);
    parser.createOption('o', "output", "Specify output file", true, "output.exr");
    parser.createOption('X', "x-start", "Specify starting texture coordinate on X-axis", true, "0");
    parser.createOption('Y', "y-start", "Specify starting texture coordinate on Y-axis", true, "0");
    parser.createOption('W', "width", "Specify width of the sample image", true);
    parser.createOption('H', "height", "Specify height of the sample image", true);
    parser.createOption('e', "extract", "Specify extract mode (SH-slice, three-basis)", true, "SH-slice");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-reflectance-slice: error: input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-reflectance-slice [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-reflectance-slice: error: too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-reflectance-slice [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }

    auto input_filename = parser.getUnassociatedArgument(0);
	Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(input_filename)));

    uint32_t btf_width = btf->Width,
             btf_height = btf->Height,
             btf_light_count = btf->LightCount,
             btf_view_count = btf->LightCount;

    uint32_t view_dir_set = Tempest::BTFNearestAngle(btf.get(), { 0.0f, 0.0f, 1.0f });

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

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    auto btf_ptr = btf.get();

    auto extract = parser.extractString("extract");
    if(extract == "SH-slice")
    {
        Tempest::TextureDescription tex_desc;
        tex_desc.Width = x_end - x_start;
        tex_desc.Height = y_end - y_start;
        tex_desc.Format = Tempest::DataFormat::R32F;

        std::unique_ptr<float[]> tex_data(new float[btf_width*btf_height]);
        auto tex_data_ptr = tex_data.get();

        auto process_sh = Tempest::CreateParallelForLoop2D(x_end - x_start, y_end - y_start, 64,
                            [tex_data_ptr, x_start, y_start, view_dir_set, &tex_desc, btf_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
                            {
                                auto coef = Tempest::SphericalHarmonicCoefficient(0, y);
                                float total = 0.0f;

                                for(uint32_t light_idx = 0, light_idx_end = btf_ptr->LightCount; light_idx < light_idx_end; ++light_idx)
                                {
                                    auto spec = Tempest::BTFFetchSpectrum(btf_ptr, light_idx, view_dir_set, x_start + x, y_start);
                                    total += Tempest::RGBToLuminance(spec)*coef*Tempest::SphericalHarmonicEvaluate(0, y, Tempest::ParabolicToCartesianCoordinates(btf_ptr->LightsParabolic[light_idx]));
                                }

                                total *= 2.0f*Tempest::MathPi/btf_ptr->LightCount; // Sort of bad monte carlo integration

                                tex_data_ptr[y*tex_desc.Width + x] = total;
                            });
        pool.enqueueTask(&process_sh);
        pool.waitAndHelp(id, &process_sh);

        auto output_file = parser.extractString("output");
        status = Tempest::SaveImage(tex_desc, tex_data.get(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to output image data: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else if(extract == "three-basis")
    {
        Tempest::TextureDescription tex_desc;
        tex_desc.Width = x_end - x_start;
        tex_desc.Height = y_end - y_start;
        tex_desc.Format = Tempest::DataFormat::RGB32F;

        std::unique_ptr<Tempest::Vector3[]> tex_data(new Tempest::Vector3[btf_width*btf_height]);
        auto tex_data_ptr = tex_data.get();

        auto thread_count = pool.getThreadCount();
        auto max_value = TGE_TYPED_ALLOCA(float, thread_count);
        std::fill(max_value, max_value + thread_count, 0.0f);

        auto radiance_integrator = Tempest::CreateParallelForLoop2D(x_end - x_start, y_end - y_start, 64,
                            [max_value, tex_data_ptr, x_start, y_start, view_dir_set, &tex_desc, btf_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
                            {
                                float total = 0.0f;

                                for(uint32_t light_idx = 0, light_idx_end = btf_ptr->LightCount; light_idx < light_idx_end; ++light_idx)
                                {
                                    auto spec = Tempest::BTFFetchSpectrum(btf_ptr, light_idx, view_dir_set, x_start + x, y_start + y);
                                    total += Tempest::RGBToLuminance(spec);
                                }

                                total *= 2.0f*Tempest::MathPi/btf_ptr->LightCount; // Sort of bad monte carlo integration

                                max_value[worker_id] = std::max(max_value[worker_id], total);
                            });
        pool.enqueueTask(&radiance_integrator);
        pool.waitAndHelp(id, &radiance_integrator);

        float total_max_value = max_value[0];
        for(uint32_t thread_idx = 0; thread_idx < thread_count; ++thread_idx)
        {
            total_max_value = std::max(total_max_value, max_value[thread_idx]);
        }

        auto process_refl = Tempest::CreateParallelForLoop2D(x_end - x_start, y_end - y_start, 64,
                            [tex_data_ptr, x_start, y_start, total_max_value, view_dir_set, &tex_desc, btf_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
                            {
                                Tempest::Vector3 total = {};

                                for(uint32_t light_idx = 0, light_idx_end = btf_ptr->LightCount; light_idx < light_idx_end; ++light_idx)
                                {
                                    auto dir = Tempest::ParabolicToCartesianCoordinates(btf_ptr->LightsParabolic[light_idx]);
                                    auto spec = Tempest::BTFFetchSpectrum(btf_ptr, light_idx, view_dir_set, x_start + x, y_start + y);
                                    total += Tempest::RGBToLuminance(spec)*Tempest::ThreeBasis(dir);
                                }

                                total *= 2.0f*Tempest::MathPi/btf_ptr->LightCount; // Sort of bad monte carlo integration

                                tex_data_ptr[y*tex_desc.Width + x] = total/total_max_value;
                            });
        pool.enqueueTask(&process_refl);
        pool.waitAndHelp(id, &process_refl);

        auto output_file = parser.extractString("output");
        status = Tempest::SaveImage(tex_desc, tex_data.get(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to output image data: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported extraction mode: ", extract);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
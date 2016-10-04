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
#include "tempest/image/btf.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/utils/timer.hh"

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("btf-to-ambient-occlusion", true);
    parser.createOption('o', "output", "Specify output texture name (default: ao.png)", true, "ao.png");
    parser.createOption('s', "sample-count", "Specify sample count for integration step", true, "128");
    parser.createOption('X', "x-start", "Specify start location on the X-axis", true, "0");
    parser.createOption('Y', "y-start", "Specify start location on the Y-axis", true, "0");
    parser.createOption('W', "width", "Specify sample width", true);
    parser.createOption('H', "height", "Specify sample height", true);
    parser.createOption('g', "global-ao", "Enable global AO approximation, i.e. isotropic shadows", false);

    auto status = parser.parse(argc, argv);

    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-to-ambient-occlusion: error: input BTF file is not specified \n\n"
                                               "USAGE:\n"
                                               "\tbtf-to-ambient-occlusion <input-file>");
        return EXIT_FAILURE;
    }

    if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-to-ambient-occlusion: error: too many input BTF files are specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-to-ambient-occlusion <input-file>");
        return EXIT_FAILURE;
    }

    Tempest::Path input_file(parser.getUnassociatedArgument(0));
    Tempest::BTFPtr btf(Tempest::LoadBTF(input_file));
    if(!btf)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load BTF: ", input_file);
        return EXIT_FAILURE;
    }

    auto sample_count = parser.extract<uint32_t>("sample-count");
    if(sample_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid sample count: ", parser.extractString("sample-count"));
        return EXIT_FAILURE;
    }



    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    auto tex_area = (uint32_t)btf->Width*btf->Height;
    std::unique_ptr<float> luminance_slice(new float[tex_area]);
    auto lv_lum_slice_ptr = luminance_slice.get();

    Tempest::TimeQuery timer;

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
    tex_desc.Format = Tempest::DataFormat::R8UNorm;

    auto result_data_ptr = new uint8_t[tex_desc.Width*tex_desc.Height];
    Tempest::Texture result_texture(tex_desc, result_data_ptr);

    bool global_ao = parser.isSet("global-ao");

    float diffuse_approx_lum = 0.0f;

    auto btf_ptr = btf.get();

    if(global_ao)
    {
        for(uint32_t y = y_start; y < y_end; ++y)
            for(uint32_t x = x_start; x < x_end; ++x)
            {
                auto diffuse_approx_color = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, x, y);
                diffuse_approx_lum = Maxf(diffuse_approx_lum, Tempest::RGBToLuminance(Tempest::SpectrumToRGB(diffuse_approx_color)));
            }
    }

    for(uint32_t y = y_start; y < y_end; ++y)
    {
        auto start_time = timer.time();

        for(uint32_t x = x_start; x < x_end; ++x)
        {
            // NOTE: The bad assumption is that it is somewhat diffuse material and we are only approximating visibility
            if(!global_ao)
            {
                auto diffuse_approx_color = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, x, y);
                diffuse_approx_lum = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(diffuse_approx_color));
            }
        #ifndef NDEBUG
            auto start_extract = timer.time();
        #endif

            Tempest::BTFParallelExtractLuminanceSlice(btf_ptr, id, pool, x, y, &lv_lum_slice_ptr);

        #ifndef NDEBUG
            auto elapsed_extract = timer.time() - start_extract;

            Tempest::Log(Tempest::LogLevel::Info, "Completed extract of luminance(", x, ", ", y, ") in ", elapsed_extract, "us");

            auto start_integrate = timer.time();
        #endif

            // TODO: triangle area integrator
            auto ao_integral = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere(id, pool, sample_count*sample_count, 32,
                                        [btf_ptr, lv_lum_slice_ptr](const Tempest::Vector3& dir2)
                                        {
                                            Tempest::Vector3 dir1{ 0.0f, 0.0f, 1.0f };
                                            auto vis0 = Tempest::BTFSampleLuminanceSlice(btf_ptr, dir1, dir2, lv_lum_slice_ptr);
                                            auto vis1 = Tempest::BTFSampleLuminanceSlice(btf_ptr, dir2, dir1, lv_lum_slice_ptr);

                                            return 0.5f*(vis0 + vis1);
                                        })/(2.0f*Tempest::MathPi);

        #ifndef NDEBUG
            auto elapsed_integrate = timer.time() - start_integrate;

            Tempest::Log(Tempest::LogLevel::Info, "Completed integration on sample(", x, ", ", y, ") in ", elapsed_integrate, "us");
        #endif
            TGE_ASSERT(ao_integral < diffuse_approx_lum, "invalid ao approximation");

            result_data_ptr[(y - y_start)*tex_desc.Width + (x - x_start)] = static_cast<uint8_t>(Tempest::Clampf(ao_integral*255.0f/diffuse_approx_lum, 0.0f, 255.0f) + 0.5f);
        }

        auto elapsed_time = timer.time() - start_time;

        Tempest::Log(Tempest::LogLevel::Info, "Completed line ", y, " in ", elapsed_time*1e-6f, "s");
    }

    Tempest::Path output_file(parser.extractString("output"));
    status = Tempest::SaveImage(tex_desc, result_data_ptr, output_file);
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to save image: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
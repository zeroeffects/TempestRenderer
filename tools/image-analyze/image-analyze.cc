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

#include <cstdlib>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("image-analyze", true);
    parser.createOption('e', "extract", "Specify extract mode (luminance, spectrum)", true, "spectrum");
    parser.createOption('d', "decibel", "Output power spectrum in decibels per sample", false);
    parser.createOption('o', "output", "Specify output image filename", true, "output.exr");
    parser.createOption('c', "center", "Center spectrum", false);
    parser.createOption('g', "gaussian-prefilter", "Apply gaussian pre-filter on data", false);
    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    if(parser.getUnassociatedCount() != 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify one input image\n"
                                               "USAGE:\n"
                                               "\timage-analyze [ <options> ] <input>");
        return EXIT_FAILURE;
    }

    auto filename = parser.getUnassociatedArgument(0);

    Tempest::TexturePtr input_texture(Tempest::LoadImage(Tempest::Path(filename)));
    if(!input_texture)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load input image file: ", filename);
        return EXIT_FAILURE;
    }

    input_texture->convertToLuminance();

    if(parser.isSet("gaussian-prefilter"))
    {
        input_texture = decltype(input_texture)(Tempest::GaussianBlurWrap(*input_texture));
    }

    auto output_file = parser.extractString("output");
    
    auto extract = parser.extractString("extract");
    if(extract == "luminance")
    {
        status = Tempest::SaveImage(input_texture->getHeader(), input_texture->getData(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to save output file: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else if(extract == "spectrum")
    {
        Tempest::TexturePtr power_spec_texture(Tempest::PowerSpectrumTexture(*input_texture));
        TGE_ASSERT(power_spec_texture, "Invalid input texture");

        auto& hdr = power_spec_texture->getHeader();
        if(parser.isSet("center"))
        {
            auto data = reinterpret_cast<uint8_t*>(power_spec_texture->getData());
            size_t elem_size = Tempest::DataFormatElementSize(hdr.Format);

            for(size_t y = 0; y < hdr.Height/2; ++y)
            {
                for(size_t x = 0; x < hdr.Width/2; ++x)
                {
                    for(size_t byte = 0; byte < elem_size; ++byte)
                    {
                        std::swap(data[(y*hdr.Width + x)*elem_size + byte],
                                  data[((y + hdr.Height/2)*hdr.Width + x + hdr.Width/2)*elem_size + byte]);
                    }
                }
            }

            for(size_t y = hdr.Height/2; y < hdr.Height; ++y)
            {
                for(size_t x = 0; x < hdr.Width/2; ++x)
                {
                    for(size_t byte = 0; byte < elem_size; ++byte)
                    {
                        std::swap(data[(y*hdr.Width + x)*elem_size + byte],
                                  data[((y - hdr.Height/2)*hdr.Width + x + hdr.Width/2)*elem_size + byte]);
                    }
                }
            }
        }

        if(parser.isSet("decibel"))
        {
            switch(hdr.Format)
            {
            case Tempest::DataFormat::R32F:
            {
                auto* vec_data = reinterpret_cast<float*>(power_spec_texture->getData());
                for(size_t idx = 0, idx_end = hdr.Width*hdr.Height; idx < idx_end; ++idx)
                {
                    float value = vec_data[idx];
                    vec_data[idx] = value != 0.0f ? 10.0f*log10f(value) : -INFINITY;
                }
            } break;
            default:
                Tempest::Log(Tempest::LogLevel::Error, "Printing decibels in this format is unsupported");
                return EXIT_FAILURE;
            }
        }

        status = Tempest::SaveImage(power_spec_texture->getHeader(), power_spec_texture->getData(), Tempest::Path(output_file));
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to save output file: ", output_file);
            return EXIT_FAILURE;
        }
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported extract mode: ", extract);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
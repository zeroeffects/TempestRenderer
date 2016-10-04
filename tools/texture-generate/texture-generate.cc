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
#include "tempest/math/gabor-noise.hh"
#include "tempest/image/image.hh"

#include <cstdlib>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("texture-generate", false);
    parser.createOption('w', "width", "Specify width of generated texture", true, "256");
    parser.createOption('h', "height", "Specify height of generated texture", true, "256");
    parser.createOption('o', "output", "Specify output file", true, "output.png");
    parser.createOption('m', "magnitude", "Specify magnitude of noise", true, "1.0");
    parser.createOption('s', "bandwidth", "Specify spread of noise pattern", true, "0.05");
    parser.createOption('p', "phase", "Specify phase of noise pattern in radians", true, "0.7853981");
    parser.createOption('i', "impulses", "Specify number of noise impulses to sum up", true, "64");
    parser.createOption('f', "frequency", "Specify frequency of the noise pattern", true, "0.0625");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        EXIT_FAILURE;
    }

    float magnitude = parser.extract<float>("magnitude");
    float bandwidth = parser.extract<float>("bandwidth");
    float phase = parser.extract<float>("phase");
    float frequency = parser.extract<float>("frequency");
    unsigned impulses = parser.extract<unsigned>("impulses");

    if(impulses == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Specify greater than zero number of impulses");
        return EXIT_FAILURE;
    }

    if(magnitude == 0.0f)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Specify greater than zero magnitude");
        return EXIT_FAILURE;
    }

    float sin_omega, cos_omega;
    Tempest::FastSinCos(phase, &sin_omega, &cos_omega);

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = parser.extract<uint16_t>("width");
    tex_desc.Height = parser.extract<uint16_t>("height");
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    if(tex_desc.Width == 0 || tex_desc.Height == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Invalid texture dimensions (", tex_desc.Width, ", ", tex_desc.Height, ")");
        return EXIT_FAILURE;
    }

    uint32_t* data = new uint32_t[tex_desc.Width*tex_desc.Height];
    Tempest::Texture tex(tex_desc, reinterpret_cast<uint8_t*>(data));

    float scale = 1.0f/(3.0f*sqrtf(Tempest::GaborVariance(magnitude, bandwidth, frequency, impulses)));
    for(uint32_t y = 0; y < tex_desc.Height; ++y)
    {
        for(uint32_t x = 0; x < tex_desc.Width; ++x)
        {
            float xf = x + 0.5f - tex_desc.Width*0.5f;
            float yf = y + 0.5f - tex_desc.Height*0.5f;

            float intensity = 0.5f + 0.5f*Tempest::GaborNoise(magnitude, bandwidth, frequency, cos_omega, sin_omega, impulses, xf, yf)*scale;

            data[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::Vector3{ intensity, intensity, intensity}); 
        }
    }

    auto output_file = parser.extractString("output");
    status = Tempest::SaveImage(tex_desc, tex.getData(), Tempest::Path(output_file));
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to save texture: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
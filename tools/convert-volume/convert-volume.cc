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

#include "tempest/math/numerical-methods.hh"
#include "tempest/utils/system.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/volume/volume.hh"

#include <iomanip>

const uint32_t IntegratorSamples = 1024;

int TempestMain(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("convert-volume", true);
    
    parser.createOption('o', "output", "The name of the file to which the processed volume is going to be written.", true);
    parser.createOption('s', "stddev", "The standard deviation used in Zhao's model", true);
    if(!parser.parse(argc, argv))
    {
        return EXIT_FAILURE;
    }

    if(parser.getUnassociatedCount() != 1)
    {
        Tempest::GenerateError("Expecting single input file");
        return EXIT_FAILURE;
    }

    auto input_file = parser.getUnassociatedArgument(0);
    auto stddev = parser.extract<float>("stddev");
    auto output_file = parser.extract<std::string>("output");

    Tempest::Path input_file_path(input_file);

    auto sggx_stddev = Tempest::ConvertZhaoMicroFlakeToSGGX(stddev);

    auto ext = input_file_path.extension();
    if(ext.empty() && !Tempest::System::Exists(input_file))
    {
        std::unique_ptr<Tempest::VolumeRoot> volume(Tempest::ParseVolumeHierarchy(input_file));
 
        const size_t angle_offset = volume->Dimensions.X*volume->Dimensions.Y*volume->Dimensions.Z*sizeof(uint8_t);

        auto output_file_desc = output_file + "_description.vol";

        std::fstream output_root_fs(output_file_desc.c_str(), std::ios::binary | std::ios::out);
        if(!output_root_fs)
        {
            Tempest::GenerateError("Failed to open output file");
            return EXIT_FAILURE;
        }

        output_root_fs.write(reinterpret_cast<char*>(&volume->MinCorner), sizeof(volume->MinCorner));
        output_root_fs.write(reinterpret_cast<char*>(&volume->MaxCorner), sizeof(volume->MaxCorner));
        output_root_fs.write(reinterpret_cast<char*>(&volume->Dimensions), sizeof(volume->Dimensions));

        auto vol_range = volume->MaxCorner - volume->MinCorner;

        for(int32_t vol_idx = 0, vol_idx_end = volume->Dimensions.X*volume->Dimensions.Y*volume->Dimensions.Z; vol_idx < vol_idx_end; ++vol_idx)
        {
            auto& vol = volume->Volumes[vol_idx];

            if(vol.Data)
            {
                int32_t x = vol_idx % volume->Dimensions.X;
                int32_t y = (vol_idx / volume->Dimensions.X) % volume->Dimensions.Y;
                int32_t z = vol_idx / (volume->Dimensions.X*volume->Dimensions.Y);

                output_root_fs.write(reinterpret_cast<char*>(&vol.Dimensions), sizeof(vol.Dimensions));

                std::stringstream ss; // yeah, i know
                ss << output_file << "_" << std::setw(3) << std::setfill('0') << x << "_"
                                         << std::setw(3) << std::setfill('0') << y << "_"
                                         << std::setw(3) << std::setfill('0') << z;

                const char* axis_names[3] =
                {
                    "-xaxis.vol",
                    "-yaxis.vol",
                    "-zaxis.vol"
                };

                std::fstream output_vol_axis_fs[TGE_FIXED_ARRAY_SIZE(axis_names)];

                for(size_t axis_idx = 0; axis_idx < TGE_FIXED_ARRAY_SIZE(axis_names); ++axis_idx)
                {
                    auto vol_name_x = ss.str() + axis_names[axis_idx];
                    std::fstream output_vol_fs(vol_name_x.c_str(), std::ios::binary | std::ios::out);

                    int32_t channels = 3;
                    auto fmt = Tempest::GridDataType::Float32;
                    output_vol_fs.write(reinterpret_cast<char*>(&fmt), sizeof(fmt));
                    output_vol_fs.write(reinterpret_cast<char*>(&vol.Dimensions), sizeof(vol.Dimensions));
                    output_vol_fs.write(reinterpret_cast<char*>(&channels), sizeof(channels));

                    Tempest::Vector3 aabb_min = vol_range * Tempest::Vector3{ (float)x/volume->Dimensions.X, (float)y/volume->Dimensions.Y, (float)z/volume->Dimensions.Z };
                    Tempest::Vector3 aabb_max = vol_range * Tempest::Vector3{ (float)(x + 1)/volume->Dimensions.X, (float)(y + 1)/volume->Dimensions.Y, (float)(z + 1)/volume->Dimensions.Z };

                    output_vol_fs.write(reinterpret_cast<char*>(&aabb_min), sizeof(aabb_min));
                    output_vol_fs.write(reinterpret_cast<char*>(&aabb_max), sizeof(aabb_max));

                    float axis_scale = Array(sggx_stddev)[axis_idx];

                    for(int32_t cell_idx = 0, cell_idx_end = vol.Dimensions.X*vol.Dimensions.Y*vol.Dimensions.Z; cell_idx < cell_idx_end; ++cell_idx)
                    {
                        uint8_t* density = (uint8_t*)vol.Data + cell_idx;
                        uint8_t* quant = (uint8_t*)vol.Data + angle_offset + 2*sizeof(uint8_t)*cell_idx;
		                auto theta = quant[0], phi = quant[1];

                        float cos_phi, sin_phi, cos_theta, sin_theta;

                        Tempest::FastSinCos(2.0f*Tempest::MathPi*(phi/255.0f), &sin_phi, &cos_phi);
                        Tempest::FastSinCos(Tempest::MathPi*(phi/255.0f), &sin_theta, &cos_theta);

		                Tempest::Vector3 axis = Tempest::Vector3{ cos_phi*sin_theta, sin_phi*sin_theta, cos_theta}*axis_scale;

                        output_vol_fs.write(reinterpret_cast<char*>(&axis), sizeof(axis));
                    }
                }
            }
        }
    }
    else if(ext == "vol")
    {
        TGE_ASSERT(false, "Stub");
    }
    else
    {
        Tempest::GenerateError("Unsupported format ('", ext, "'): ", input_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
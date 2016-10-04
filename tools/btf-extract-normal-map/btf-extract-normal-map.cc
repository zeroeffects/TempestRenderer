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
#include "tempest/image/btf.hh"
#include "tempest/utils/threads.hh"
#include "tempest/math/matrix-variadic.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/texture.hh"

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("btf-extract-normal-map", true);
    parser.createOption('X', "x-coordinate", "Specify starting point location along X axis.", true, "0");
    parser.createOption('Y', "y-coordinate", "Specify starting point location along Y axis.", true, "0");
    parser.createOption('W', "width", "Specify width of the BTF sample", true, "0");
    parser.createOption('H', "height", "Specify height of the BTF sample", true, "0");
    parser.createOption('o', "output", "Specify output file (default: normalmap.png)", true, "normalmap.png");

    auto status = parser.parse(argc, argv);

    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-extract-normal-map: error: input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-extract-normal-map <input-file>");
        return EXIT_FAILURE;
    }

    auto input_filename = parser.getUnassociatedArgument(0);
    Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(input_filename)));

    uint32_t btf_width = btf->Width,
             btf_height = btf->Height,
             btf_light_count = btf->LightCount,
             btf_view_count = btf->LightCount,
             btf_start_x = parser.extract<uint32_t>("x-coordinate"),
             btf_start_y = parser.extract<uint32_t>("y-coordinate"),
             range_y = parser.extract<uint32_t>("height"),
             range_x = parser.extract<uint32_t>("width"),
             btf_end_y = btf_start_y + range_y,
             btf_end_x = btf_start_x + range_x;
        
    if(btf_end_x == 0 || range_x == 0)
    {
        btf_end_x = btf_width;
        range_x = btf_width - btf_start_x;
    }

    if(btf_end_y == 0 || range_y == 0)
    {
        btf_end_y = btf_height;
        range_y = btf_height - btf_start_y;
    }

    if(btf_start_x >= btf_width || btf_end_x >= btf_width ||
       btf_start_y >= btf_height || btf_end_y >= btf_height)
    {
        Tempest::Log(Tempest::LogLevel::Error, "the specified BTF coordinates are out-of-bounds: (", btf_start_x, "--", btf_end_x,"), (", btf_start_y, "--", btf_end_y, ")");
        return EXIT_FAILURE;
    }

    const size_t vec3_comps = TGE_FIXED_ARRAY_SIZE(Tempest::Vector3().Components);
    size_t lv_lum_slice_size = btf_light_count*btf_view_count;
    size_t light_dirs_size = btf_light_count*TGE_FIXED_ARRAY_SIZE(Tempest::Vector3().Components);
    size_t data_size = lv_lum_slice_size + light_dirs_size;

    std::unique_ptr<float[]> data(new float[data_size]);

    float* offset = data.get();

    auto lv_lum_slice = offset;
    offset += lv_lum_slice_size;
    Tempest::Vector3* lights_converted = reinterpret_cast<Tempest::Vector3*>(offset);
    offset += light_dirs_size;
    
	Tempest::Vector3 normal;
	auto* normal_ptr = normal.Components;

    for(uint32_t light_idx = 0; light_idx < btf_light_count; ++light_idx)
    {
        lights_converted[light_idx] = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
    }

    TGE_ASSERT(offset == data.get() + data_size, "Invalid data population");

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    auto* normal_map_data = new uint32_t[range_x*range_y];

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = range_x;
    tex_desc.Height = range_y;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    Tempest::Texture normal_map(tex_desc, reinterpret_cast<uint8_t*>(normal_map_data));

    for(uint32_t btf_y = btf_start_y, nm_y = 0; btf_y < btf_end_y; ++btf_y, ++nm_y)
        for(uint32_t btf_x = btf_start_x, nm_x = 0; btf_x < btf_end_x; ++btf_x, ++nm_x)
        {
            Tempest::BTFParallelExtractLuminanceSlice(btf.get(), id, pool, btf_x, btf_y, &lv_lum_slice);

			for(uint32_t view_idx = 1; view_idx < btf_view_count; ++view_idx)
				for(uint32_t light_idx = 0; light_idx < btf_light_count; ++light_idx)
				{
					lv_lum_slice[light_idx] += lv_lum_slice[view_idx*btf_light_count + light_idx];
				}

            float scale_coef = 1.0f/btf_view_count;
			for(uint32_t light_idx = 0; light_idx < btf_light_count; ++light_idx)
			{
				lv_lum_slice[light_idx] *= scale_coef;
			}

            Tempest::MatrixTransposeLinearSolve(reinterpret_cast<float*>(lights_converted), (uint32_t)vec3_comps, btf_light_count, 
                                                lv_lum_slice, 1, btf_light_count, 
                                                &normal_ptr);

            normal_map_data[nm_y*range_x + nm_x] = Tempest::ToColor(Normalize(normal)*0.5f + Tempest::ToVector3(0.5f));
        }

    Tempest::SaveImage(tex_desc, normal_map_data, Tempest::Path(parser.extractString("output")));

    return EXIT_SUCCESS;
}
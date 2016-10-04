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

#include "tempest/image/exr-image.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/graphics/texture.hh"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr/tinyexr.h"

namespace Tempest
{
Texture* LoadEXRImage(const Path& file_path)
{
    const char* err;

    EXRImage exr_image;
    InitEXRImage(&exr_image);

    auto exr_scope = CreateAtScopeExit([&exr_image](){ FreeEXRImage(&exr_image); });

    auto filename = file_path.get();
    auto cstr_filename = filename.c_str();
    int ret = ParseMultiChannelEXRHeaderFromFile(&exr_image, cstr_filename, &err);
    if(ret != 0)
    {
        Log(LogLevel::Error, "Failed to EXR load file: ", err, ": ", cstr_filename);
        return nullptr;
    }

    if(exr_image.num_channels <= 0)
    {
        Log(LogLevel::Error, "Invalid number of channels in EXR file: ", cstr_filename);
        return nullptr;
    }

    if(exr_image.num_channels > 4)
    {
        Log(LogLevel::Error, "EXR file cannot be loaded into a single texture: Too many channels: ", cstr_filename);
        return nullptr;
    }

    auto pixel_format = exr_image.pixel_types[0];
    if(pixel_format == TINYEXR_PIXELTYPE_HALF)
    {
        pixel_format = TINYEXR_PIXELTYPE_FLOAT;
    }

    for(int i = 0; i < exr_image.num_channels; i++)
    {
        auto cur_pixel_format = exr_image.pixel_types[i];
        if(cur_pixel_format == TINYEXR_PIXELTYPE_HALF)
        {
            cur_pixel_format = exr_image.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        }
        if(cur_pixel_format != pixel_format)
        {
            Log(LogLevel::Error, "EXR file cannot be loaded into a single texture: ", cstr_filename);
            return nullptr;
        }
    }

    TextureDescription tex_desc;
    tex_desc.Width = exr_image.width;
    tex_desc.Height = exr_image.height;

    uint32_t base_size = 0;

    switch(pixel_format)
    {
    case TINYEXR_PIXELTYPE_FLOAT:
    {
        base_size = sizeof(float);
        switch(exr_image.num_channels)
        {
        case 1:
        {
            tex_desc.Format = DataFormat::R32F;
        } break;
        case 2:
        {
            tex_desc.Format = DataFormat::RG32F;
        } break;
        case 3:
        {
            tex_desc.Format = DataFormat::RGB32F;
        } break;
        case 4:
        {
            tex_desc.Format = DataFormat::RGBA32F;
        } break;
        }
    } break;
    case TINYEXR_PIXELTYPE_UINT:
    {
        base_size = sizeof(uint32_t);
        switch(exr_image.num_channels)
        {
        case 1:
        {
            tex_desc.Format = DataFormat::R32;
        } break;
        case 2:
        {
            tex_desc.Format = DataFormat::RG32;
        } break;
        case 3:
        {
            tex_desc.Format = DataFormat::RGB32;
        } break;
        case 4:
        {
            tex_desc.Format = DataFormat::RGBA32;
        } break;
        }
    } break;
    default:
    {
        Log(LogLevel::Error, "Unsupported EXR format: ", cstr_filename);
        return nullptr;
    }
    }

    ret = LoadMultiChannelEXRFromFile(&exr_image, cstr_filename, &err);
    if(ret != 0)
    {
        Log(LogLevel::Error, "Failed to load channels from EXR file: ", err, ": ", cstr_filename);
        return nullptr;
    }

    uint32_t tex_area = tex_desc.Width*tex_desc.Height;

    std::unique_ptr<uint8_t[]> data(new uint8_t[tex_area*DataFormatElementSize(tex_desc.Format)]);

    uint32_t channel_remap[4];

    bool assign_sequential = false;
    for(int chan_idx = 0; chan_idx < exr_image.num_channels; ++chan_idx)
    {
        auto* name = exr_image.channel_names[chan_idx];
        auto chan_sym = *name;
        if(chan_sym == 0 || name[1])
            break;
        switch(chan_sym)
        {
        case 'R': channel_remap[chan_idx] = 0; break;
        case 'G': channel_remap[chan_idx] = 1; break;
        case 'B': channel_remap[chan_idx] = 2; break;
        case 'A': channel_remap[chan_idx] = 3; break;
        }
    }

    if(assign_sequential)
    {
        uint32_t idx = 0;
        for(auto& chan : channel_remap)
            chan = idx++;
    }

    uint8_t* out_data_ptr = data.get();
	for(int y = 0; y < exr_image.height; ++y)
	{
        int y_imag = exr_image.line_order ? y : exr_image.height - 1 - y;

		for(int x = 0; x < exr_image.width; ++x)
		{
			for(int chan_idx = 0; chan_idx < exr_image.num_channels; ++chan_idx)
			{
				memcpy(out_data_ptr + channel_remap[chan_idx] * base_size,
					   exr_image.images[chan_idx] + (y_imag*exr_image.width + x)*base_size, base_size);
			}
			out_data_ptr += exr_image.num_channels*base_size;
		}
	}
    return new Texture(tex_desc, data.release());
}

bool SaveEXRImage(const TextureDescription& tex, const void* data, const Path& file_path)
{
    const char* err;

    auto filename = file_path.get();
    auto* cstr_filename = filename.c_str();

    EXRImage exr_image{};
    InitEXRImage(&exr_image);
    auto exr_scope = CreateAtScopeExit([&exr_image](){ FreeEXRImage(&exr_image); });

    struct ConversionDescription
    {
        uint32_t BaseSize = 0,
                 ChannelCount = 0;
    };

    int exr_type = 0;
    uint32_t base_size = 0, channels = 0;
    switch(tex.Format)
    {
    case DataFormat::R32:
    {
        base_size = sizeof(uint32_t);
        exr_image.num_channels = 1;
        exr_type = TINYEXR_PIXELTYPE_UINT;
    } break;
    case DataFormat::RG32:
    {
        base_size = sizeof(uint32_t);
        exr_image.num_channels = 2;
        exr_type = TINYEXR_PIXELTYPE_UINT;
    } break;
    case DataFormat::RGB32:
    {
        base_size = sizeof(uint32_t);
        exr_image.num_channels = 3;
        exr_type = TINYEXR_PIXELTYPE_UINT;
    } break;
    case DataFormat::RGBA32:
    {
        base_size = sizeof(uint32_t);
        exr_image.num_channels = 4;
        exr_type = TINYEXR_PIXELTYPE_UINT;
    } break;
    case DataFormat::R32F:
    {
        base_size = sizeof(float);
        exr_image.num_channels = 1;
        exr_type = TINYEXR_PIXELTYPE_FLOAT;
    } break;
    case DataFormat::RG32F:
    {
        base_size = sizeof(float);
        exr_image.num_channels = 2;
        exr_type = TINYEXR_PIXELTYPE_FLOAT;
    } break;
    case DataFormat::RGB32F:
    {
        base_size = sizeof(float);
        exr_image.num_channels = 3;
        exr_type = TINYEXR_PIXELTYPE_FLOAT;
    } break;
    case DataFormat::RGBA32F:
    {
        base_size = sizeof(float);
        exr_image.num_channels = 4;
        exr_type = TINYEXR_PIXELTYPE_FLOAT;
    } break;
    case DataFormat::R16F:
    {
        base_size = sizeof(uint16_t);
        exr_image.num_channels = 1;
        exr_type = TINYEXR_PIXELTYPE_HALF;
    } break;
    case DataFormat::RG16F:
    {
        base_size = sizeof(uint16_t);
        exr_image.num_channels = 2;
        exr_type = TINYEXR_PIXELTYPE_HALF;
    } break;
    case DataFormat::RGBA16F:
    {
        base_size = sizeof(uint16_t);
        exr_image.num_channels = 4;
        exr_type = TINYEXR_PIXELTYPE_HALF;
    } break;
    default:
    {
        Log(LogLevel::Error, "Unsupported data format type in the EXR save procedure: ", (uint32_t)tex.Format, ": ", cstr_filename);
        return false;
    }
    }

    exr_image.width = tex.Width;
    exr_image.height = tex.Height;
    exr_image.channel_names = reinterpret_cast<const char**>(malloc(exr_image.num_channels*sizeof(exr_image.channel_names)));
    exr_image.images = reinterpret_cast<uint8_t**>(malloc(exr_image.num_channels*sizeof(exr_image.images)));
    exr_image.pixel_types = reinterpret_cast<int*>(malloc(exr_image.num_channels*sizeof(exr_image.pixel_types)));
    exr_image.requested_pixel_types = reinterpret_cast<int*>(malloc(exr_image.num_channels*sizeof(exr_image.requested_pixel_types)));

    const char* predefined_names = "RGBA";

    uint32_t tex_area = tex.Width*tex.Height;
    for(int chan_idx = 0; chan_idx < exr_image.num_channels; ++chan_idx)
    {
        auto* name = reinterpret_cast<char*>(malloc(2 * sizeof(char)));
        name[0] = predefined_names[chan_idx];
        name[1] = 0;
        exr_image.channel_names[chan_idx] = name;
        exr_image.requested_pixel_types[chan_idx] = exr_image.pixel_types[chan_idx] = exr_type;
        auto* image = exr_image.images[chan_idx] = reinterpret_cast<uint8_t*>(malloc(tex_area*base_size));
        for(int y = 0; y < exr_image.height; ++y)
		{
			for(int x = 0; x < exr_image.width; ++x)
			{
				int out_idx = ((exr_image.height - 1 - y)*exr_image.width + x)*base_size,
					in_idx = (y*exr_image.width + x)*exr_image.num_channels*base_size + chan_idx*base_size;
				memcpy(image + out_idx, reinterpret_cast<const char*>(data) + in_idx, base_size);
			}
		}
    }
    
    exr_image.compression = TINYEXR_COMPRESSIONTYPE_NONE;
    auto ret = SaveMultiChannelEXRToFile(&exr_image, cstr_filename, &err);
    if(ret != 0)
    {
        Log(LogLevel::Error, "Failed to save EXR image: ", err, ": ", cstr_filename);
        return false;
    }

    return true;
}
}
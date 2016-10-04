/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2014 Zdravko Velinov
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

#include "tempest/image/png-image.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/memory.hh"
#include "tempest/graphics/texture.hh"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

//#define LIBPNG_SUPPORT

#ifdef LIBPNG_SUPPORT
#   include <png.h>
#endif
#include <cstring>
#include <fstream>

namespace Tempest
{
#ifdef LIBPNG_SUPPORT
static void LogErrorFunction(png_structp png_ptr, png_const_charp error_msg)
{
    Log(LogLevel::Error, "Fatal error has occurred while trying to load an png file: ", error_msg);
}

static void LogWarningFunction(png_structp png_ptr, png_const_charp warning_msg)
{
    Log(LogLevel::Warning, "PNG: ", warning_msg);
}

static void PNGReadData(png_structp png_ptr, png_bytep data, png_size_t length)
{
    if(png_ptr == NULL)
        return;
    std::fstream* fs = reinterpret_cast<std::fstream*>(png_get_io_ptr(png_ptr));
    TGE_ASSERT(fs, "Invalid file stream");
    fs->read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(length));
    if(!fs)
    {
        png_error(png_ptr, "Read Error");
    }
}
#endif

Texture* LoadPNGImage(const Path& file_path)
{
#ifndef LIBPNG_SUPPORT
	auto filename = file_path.get();
	int x, y, comp;

	stbi_set_flip_vertically_on_load(true);
	auto result = CREATE_SCOPED(stbi_uc*, free);
	result = stbi_load(filename.c_str(), &x, &y, &comp, 0);
	if(!result)
		return nullptr;

	int final_comp = NextPowerOf2(comp);

	std::unique_ptr<uint8_t[]> imgdata(new uint8_t[x*y*final_comp]); 

	TextureDescription tex_desc;
	tex_desc.Width = x;
	tex_desc.Height = y;
	switch(comp)
	{
	case 1:
	{
		memcpy(imgdata.get(), result, x*y*comp);
		tex_desc.Format = DataFormat::R8UNorm;
	} break;
	case 2:
	{
		memcpy(imgdata.get(), result, x*y*comp);
		tex_desc.Format = DataFormat::RG8UNorm;
	} break;
	case 3:
	{
		for(int idx = 0, idx_end = x*y; idx < idx_end; ++idx)
		{
			auto* src = result + idx*comp;
			auto* dst = imgdata.get() + idx*final_comp;

			dst[0] = src[0];
			dst[1] = src[1];
			dst[2] = src[2];
			dst[3] = 0xFF;
		}
		tex_desc.Format = DataFormat::RGBA8UNorm;
	} break;
	case 4:
	{
		memcpy(imgdata.get(), result, x*y*comp);
		tex_desc.Format = DataFormat::RGBA8UNorm;
	} break;
	default:
		Log(LogLevel::Error, "Unsupported png format: ", filename);
		return nullptr;
	}
	
	return new Texture(tex_desc, imgdata.release());
#else
    const std::streamsize num = 4;
    png_byte header[num];
    std::fstream fs;
    fs.open(file_path.get().c_str(), std::ios::in | std::ios::binary);
    if(!fs)
        return nullptr;
    fs.read(reinterpret_cast<char*>(header), num);
    int not_png = png_sig_cmp(header, 0, num*sizeof(png_byte));
    if(not_png)
        return nullptr;
    struct PNGData
    {
        png_structp     PNGStruct = nullptr;
        png_infop       PNGInfo = nullptr;
        png_infop       PNGEndInfo = nullptr;

        ~PNGData()
        {
            png_destroy_read_struct(&PNGStruct, &PNGInfo, &PNGEndInfo);
        }
    } data;

    data.PNGStruct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, &LogErrorFunction, &LogWarningFunction);
    if(!data.PNGStruct)
        return nullptr;
    data.PNGInfo = png_create_info_struct(data.PNGStruct);
    if(!data.PNGInfo)
        return nullptr;
    data.PNGEndInfo = png_create_info_struct(data.PNGStruct);
    if(!data.PNGEndInfo)
        return nullptr;

    png_set_read_fn(data.PNGStruct, &fs, &PNGReadData);
    png_set_sig_bytes(data.PNGStruct, num);
    png_uint_32 width, height;
    int     bit_depth, color_type, interlace_type,
        compression_type, filter_method;
    png_read_info(data.PNGStruct, data.PNGInfo);
    png_get_IHDR(data.PNGStruct, data.PNGInfo, &width, &height,
                 &bit_depth, &color_type, &interlace_type,
                 &compression_type, &filter_method);
    if(bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(data.PNGStruct);
    if(png_get_valid(data.PNGStruct, data.PNGInfo, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(data.PNGStruct);
    switch(color_type)
    {
    case PNG_COLOR_TYPE_PALETTE:
        png_set_palette_to_rgb(data.PNGStruct); // fall through
    case PNG_COLOR_TYPE_RGB:
        png_set_add_alpha(data.PNGStruct, 0xFFFF, PNG_FILLER_AFTER); break;
    }

    png_read_update_info(data.PNGStruct, data.PNGInfo);
    color_type = png_get_color_type(data.PNGStruct, data.PNGInfo);
    bit_depth = png_get_bit_depth(data.PNGStruct, data.PNGInfo);
    auto rowbytes = png_get_rowbytes(data.PNGStruct, data.PNGInfo);
    std::unique_ptr<uint8_t[]> imgdata(new uint8_t[height*rowbytes]);
    std::unique_ptr<uint8_t*[]> p_imgdata(new uint8_t*[height]);
    for(size_t i = 0; i < height; ++i)
        p_imgdata[i] = imgdata.get() + i*rowbytes;
    png_read_image(data.PNGStruct, reinterpret_cast<png_bytepp>(p_imgdata.get()));
    png_read_end(data.PNGStruct, data.PNGEndInfo);

    TextureDescription tex_desc;
    tex_desc.Width = static_cast<uint16_t>(width);
    tex_desc.Height = static_cast<uint16_t>(height);
    switch(color_type)
    {
    case PNG_COLOR_TYPE_GRAY:
    {
        switch(bit_depth)
        {
        case 8: tex_desc.Format = DataFormat::R8UNorm; break;
        case 16: tex_desc.Format = DataFormat::R16UNorm; break;
        default: Log(LogLevel::Error, "Unsupported bpp");
        }
        return new Texture(tex_desc, imgdata.release());
    } break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
    {
        size_t bpp = bit_depth/8;
        std::unique_ptr<uint8_t[]> conv_data(new uint8_t[height*width*bpp]);
        uint8_t* in_data = imgdata.get();
        uint8_t* out_data = conv_data.get();
        for(size_t y = 0; y < height; ++y)
        {
            in_data = imgdata.get() + y*rowbytes;
            for(size_t x = 0; x < width; ++x)
            {
                memcpy(out_data, in_data, bpp); out_data += bpp;
                memcpy(out_data, in_data, bpp); out_data += bpp;
                memcpy(out_data, in_data, bpp); out_data += bpp; in_data += bpp;
                memcpy(out_data, in_data, bpp); out_data += bpp; in_data += bpp;
            }
        }
        switch(bit_depth)
        {
        case 8: tex_desc.Format = DataFormat::RGBA8UNorm; break;
        case 16: tex_desc.Format = DataFormat::RGBA16UNorm; break;
        default: Log(LogLevel::Error, "Unsupported bpp");
        }
        return new Texture(tex_desc, conv_data.release());
    } break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
    {
        switch(bit_depth)
        {
        case 8: tex_desc.Format = DataFormat::RGBA8UNorm; break;
        case 16: tex_desc.Format = DataFormat::RGBA16UNorm; break;
        default: Log(LogLevel::Error, "Unsupported bpp");
        }
        return new Texture(tex_desc, imgdata.release());
    } break;
    default:
        Log(LogLevel::Error, "Unsupported format");
        return nullptr;
    }
#endif
}

bool SavePNGImage(const TextureDescription& tex_hdr, const void* data, const Path& file_path)
{
	auto filename = file_path.get();
	int ret = 1;
	switch(tex_hdr.Format)
	{
	case DataFormat::R8UNorm:
	{
		ret = stbi_write_png(filename.c_str(), tex_hdr.Width, tex_hdr.Height, 1, reinterpret_cast<const uint8_t*>(data) + 1*tex_hdr.Width*(tex_hdr.Height - 1)*sizeof(uint8_t), -1*tex_hdr.Width*sizeof(uint8_t));
	} break;
	case DataFormat::RG8UNorm:
	{
		ret = stbi_write_png(filename.c_str(), tex_hdr.Width, tex_hdr.Height, 2, reinterpret_cast<const uint8_t*>(data) + 2*tex_hdr.Width*(tex_hdr.Height - 1)*sizeof(uint8_t), -2*tex_hdr.Width*sizeof(uint8_t));
	} break;
//  case DataFormat::RGB8UNorm:
	case DataFormat::RGBA8UNorm:
	{
		ret = stbi_write_png(filename.c_str(), tex_hdr.Width, tex_hdr.Height, 4, reinterpret_cast<const uint8_t*>(data) + 4*tex_hdr.Width*(tex_hdr.Height - 1)*sizeof(uint8_t), -4*tex_hdr.Width*sizeof(uint8_t));
	} break;
	default:
		Log(LogLevel::Error, "Unsupported format: ", filename);
		return false;
	}

	if(ret == 0)
	{
		Log(LogLevel::Error, "Failed to write png file: ", filename);
		return false;
	}

	return true;
}
}

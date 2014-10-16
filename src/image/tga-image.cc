/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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

#include "tempest/image/tga-image.hh"
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/types.hh"

#include <fstream>
#include <memory>
#include <cstring>

namespace Tempest
{
typedef int8    tga_byte;
typedef int16   tga_short;
typedef int32   tga_long;
typedef char    tga_ascii;

#ifndef _MSC_VER
#define ATTR_PACKED __attribute__((packed))
#else
#define ATTR_PACKED
#endif

#ifdef _MSC_VER
#   pragma pack(push, 1)
#endif
struct ATTR_PACKED TGAHeader
{
    tga_byte    IDLength;
    tga_byte    ColorMapType;
    tga_byte    ImageType;
    
    tga_short   CMS_FirstEntryIndex;
    tga_short   CMS_ColorMapLength;
    tga_byte    CMS_ColorMapEntrySize;
    
    tga_short   IS_XOrigin;
    tga_short   IS_YOrigin;
    tga_short   IS_ImageWidth;
    tga_short   IS_ImageHeight;
    tga_byte    IS_PixelDepth;
    tga_byte    IS_ImageDescriptor;
};
#ifdef _MSC_VER
#   pragma pack(pop)
#endif

// struct TGAFooter
// {
//  tga_long ExtensionAreaOffset;
//  tga_long DeveloperDirectoryOffset;
//  tga_ascii Signature[18];
// };

enum TGAImageType
{
    TGA_NO_IMAGE_DATA                   = 0,
    TGA_UNCOMPRESSED_COLOR_MAPPED       = 1,
    TGA_UNCOMPRESSED_TRUE_COLOR         = 2,
    TGA_UNCOMPRESSED_BLACK_AND_WHITE    = 3,
    TGA_RUN_LENGTH_COLOR_MAPPED         = 9,
    TGA_RUN_LENGTH_TRUE_COLOR           = 10,
    TGA_RUN_LENGTH_BLACK_AND_WHITE      = 11
};

enum TGAMask
{
    TGA_IMAGE_TYPE_RLE_MASK             = 1 << 3,
    TGA_RIGHT_BIT_MASK                  = 1 << 4,
    TGA_TOP_BIT_MASK                    = 1 << 5,
    TGA_RLE_MASK                        = 1 << 7
};   
 
Texture* LoadTGAImage(const Path& file_path)
{
    TextureDescription tex_header;
    
    //STATIC_ASSERT(sizeof(TGAFooter) == 26, "");
    static_assert(sizeof(tga_byte) == 1 && sizeof(tga_short) == 2 && sizeof(tga_long) == 4 && sizeof(tga_ascii) == 1, "broken types");
    static_assert(sizeof(TGAHeader) == 18, "broken code");
    std::fstream fs(file_path.get().c_str(), std::ios::in | std::ios::binary);
    if(!fs)
    {
        Log(LogLevel::Error, "Failed to load TGA file: ", file_path.get());
        return nullptr;
    }
    TGAHeader tga_header;
    fs.read(reinterpret_cast<char*>(&tga_header), sizeof(TGAHeader));
    size_t  image_area = tga_header.IS_ImageWidth*tga_header.IS_ImageHeight,
            bytes_per_pixel = tga_header.IS_PixelDepth / 8,
            final_bytes_per_pixel;
    
    if(tga_header.ImageType == TGA_NO_IMAGE_DATA)
    {
        return nullptr;
    }
    switch(tga_header.ImageType)
    {
    case TGA_UNCOMPRESSED_BLACK_AND_WHITE: // fall through
    case TGA_RUN_LENGTH_BLACK_AND_WHITE:
    {
        if(tga_header.ColorMapType)
            return nullptr;
        switch(tga_header.IS_PixelDepth)
        {
        case 8: tex_header.Format = DataFormat::R8UNorm; break;
        case 16: tex_header.Format = DataFormat::R16UNorm; break;
        default:
            return nullptr;
        }
        final_bytes_per_pixel = bytes_per_pixel;
    } break;
    case TGA_UNCOMPRESSED_TRUE_COLOR: // fall through
    case TGA_RUN_LENGTH_TRUE_COLOR:
    {
        if((tga_header.IS_PixelDepth != 24 &&
            tga_header.IS_PixelDepth != 32) ||
           tga_header.ColorMapType)
            return nullptr;
        tex_header.Format = DataFormat::RGBA8UNorm;
        final_bytes_per_pixel = 4;
        
    } break;
    }

    size_t image_size = image_area*final_bytes_per_pixel;
    std::unique_ptr<uint8 []> data(new uint8[image_size]);
    
    if((tga_header.ImageType & TGA_IMAGE_TYPE_RLE_MASK) != 0)
    { 
        size_t i;
        char rle_header;
        if(tga_header.ColorMapType &&
           tga_header.IS_PixelDepth != 24 &&
           tga_header.IS_PixelDepth != 32)
        {
            return nullptr;
        }
        for(i = 0; i < image_size;)
        {
            fs.read(&rle_header, 1);
            if((rle_header & TGA_RLE_MASK) != 0)
            {
                size_t j, jend;
                rle_header ^= TGA_RLE_MASK;
                fs.read(reinterpret_cast<char*>(&data[i]), bytes_per_pixel);
                for(j = i + final_bytes_per_pixel, jend = j + rle_header*final_bytes_per_pixel;
                    j < jend; j += final_bytes_per_pixel)
                {
                    memcpy(&data[j], &data[i], final_bytes_per_pixel);
                }
                i = j;
            }
            else
            {
                ++rle_header;
                for(size_t j = 0; j < (size_t)rle_header; ++j, i += final_bytes_per_pixel)
                {
                    fs.read(reinterpret_cast<char*>(&data[i]), bytes_per_pixel);
                }
            }
        }
        if(i != image_size)
        {
            return nullptr;
        }
    }
    else
    {
        for(size_t i = 0; i < image_size; i += final_bytes_per_pixel)
        {
            fs.read(reinterpret_cast<char*>(&data[i]), bytes_per_pixel);
        }
    }
    
    for(size_t i = 0; i < image_size; i += final_bytes_per_pixel)
    {
        // swap R and B, because TGA uses BGRA
        std::swap(data[i], data[i + 2]);
    }

    if(tga_header.IS_ImageDescriptor & TGA_RIGHT_BIT_MASK)
    {
        for(size_t i = 0; i < (size_t)tga_header.IS_ImageHeight; ++i)
        {
            for(size_t j = 0, rj = tga_header.IS_ImageWidth - 1; j < (size_t)tga_header.IS_ImageWidth/2; ++j, --rj)
            {
                for(size_t k = 0; k < final_bytes_per_pixel; ++k)
                {
                    std::swap(data[(i*tga_header.IS_ImageHeight + j)*final_bytes_per_pixel + k],
                              data[(i*tga_header.IS_ImageHeight + rj)*final_bytes_per_pixel + k]);
                }
            }
        }
    }
    if(tga_header.IS_ImageDescriptor & TGA_TOP_BIT_MASK)
    {
        for(size_t i = 0, ri = tga_header.IS_ImageWidth - 1; i < (size_t)tga_header.IS_ImageHeight/2; ++i, --ri)
        {
            for(size_t j = 0; j < (size_t)tga_header.IS_ImageWidth; ++j)
            {
                for(size_t k = 0; k < final_bytes_per_pixel; ++k)
                {
                    std::swap(data[(i*tga_header.IS_ImageHeight + j)*final_bytes_per_pixel + k],
                              data[(ri*tga_header.IS_ImageHeight + j)*final_bytes_per_pixel + k]);
                }
            }
        }
    }
    
    tex_header.Width = static_cast<size_t>(tga_header.IS_ImageWidth);
    tex_header.Height = static_cast<size_t>(tga_header.IS_ImageHeight);
    tex_header.Depth = 1;
    tex_header.Tiling = TextureTiling::Flat;
    return new Texture(tex_header, data.release());
}
}
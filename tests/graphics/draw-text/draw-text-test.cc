#include "tempest/utils/testing.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/utils/file-system.hh"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"

#include <iostream>
#include <memory>
#include <cstdlib>

TGE_TEST("Testing text drawing capabilities")
{
    stbtt_fontinfo font;
    int ascent, baseline, ch = 0;
    float scale, xpos = 2; 
    char *text = "Hello World!";

    const char* filename = ROOT_SOURCE_DIR "/share/fonts/CourierCode/CourierCode-Roman.ttf";
    std::fstream fs(filename, std::ios::in|std::ios::binary);
    TGE_CHECK(fs.good(), "Failed to load font");

    auto file_start = fs.tellg();
    fs.seekg(0, std::ios::end);
    auto file_end = fs.tellg();
    fs.seekg(0, std::ios::beg);
    size_t file_size = file_end - file_start;

    std::unique_ptr<uint8_t[]> font_data(new uint8_t[file_size]);
    fs.read(reinterpret_cast<char*>(font_data.get()), file_size);
    TGE_CHECK(fs.good(), "Failed to load font");

    stbtt_InitFont(&font, font_data.get(), 0);

    scale = stbtt_ScaleForPixelHeight(&font, 15);
    stbtt_GetFontVMetrics(&font, &ascent, 0, 0);
    baseline = (int) (ascent*scale);

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = 200;
    tex_desc.Height = 200;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    size_t tex_size = (size_t)tex_desc.Width*tex_desc.Height;
    auto* tex_data = new uint32_t[tex_size];
    std::fill(tex_data, tex_data + tex_size, 0);

    Tempest::Texture out_tex(tex_desc, reinterpret_cast<uint8_t*>(tex_data));

    int lum_buffer_size = 128*128;
    std::unique_ptr<uint8_t[]> lum_buffer(new uint8_t[lum_buffer_size]);

    while(text[ch])
    {
        int advance, lsb, x0, y0, x1, y1;
        float x_shift = xpos - (float) floor(xpos);
        stbtt_GetCodepointHMetrics(&font, text[ch], &advance, &lsb);
        stbtt_GetCodepointBitmapBoxSubpixel(&font, text[ch], scale, scale, x_shift, 0, &x0, &y0, &x1, &y1);

        auto width = x1 - x0;

        if(width)
        {
            auto height = Tempest::Clamp(y1 - y0, 0, lum_buffer_size/width);

            stbtt_MakeCodepointBitmapSubpixel(&font, lum_buffer.get(), width, height, width, scale, scale, x_shift, 0, text[ch]);
        
            for(int y = 0, yend = std::min(tex_desc.Height - (baseline + y0), height); y < yend; ++y)
            {
                for(int x = 0, xend = std::min(tex_desc.Width - ((int)xpos + x0), width); x < width; ++x)
                {
                    auto lum = lum_buffer[y*width + x];
                    tex_data[(tex_desc.Height - 1 - (baseline + y0 + y))*tex_desc.Width + x0 + (int)xpos + x] = Tempest::rgba(255, 255, 255, lum);
                }
            }
        }

        xpos += (advance*scale);
        if(text[ch + 1])
        {
            xpos += scale*stbtt_GetCodepointKernAdvance(&font, text[ch], text[ch+1]);
        }
        ++ch;
    }

    Tempest::SaveImage(tex_desc, tex_data, Tempest::Path("text.png"));
}
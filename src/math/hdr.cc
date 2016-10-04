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

#include "tempest/math/hdr.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/threads.hh"
#include "tempest/utils/memory.hh"

namespace Tempest
{
static const float Epsilon = 1e-3f;

float ParallelLogAverage(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size)
{
    auto& header = hdr_tex->getHeader();
    auto width = header.Width,
         height = header.Height;
    uint32_t area = width*height;

    auto thread_count = pool.getThreadCount();

    size_t mean_values_size = thread_count*sizeof(Vector3);
    auto* log_total_values = TGE_TYPED_ALLOCA(Vector3, mean_values_size);
    memset(log_total_values, 0, mean_values_size);
    auto mean_compute = CreateParallelForLoop2D(width, height, chunk_size,
                                                [log_total_values, hdr_tex](uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto color = hdr_tex->fetchRGB(x, y);
        auto log_val = Vector3Log(Epsilon + color);
        if(std::isfinite(log_val.x) && std::isfinite(log_val.y) && std::isfinite(log_val.z))
            log_total_values[worker_id] += log_val;
    });
    pool.enqueueTask(&mean_compute);

    pool.waitAndHelp(id, &mean_compute);

    Vector3 log_total_value{};
    for(uint32_t worker_idx = 0; worker_idx < thread_count; ++worker_idx)
    {
        log_total_value += log_total_values[worker_idx];
    }
    float mean_value = RGBToLuminance(Vector3Exp(log_total_value / (float)area));
    return mean_value;
}

template<class TConvert>
Texture* ParallelConvertHDR(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod, float* out_exp_factor, const TConvert& convert)
{
    auto& header = hdr_tex->getHeader();
    auto width = header.Width,
         height = header.Height;
    auto mean_value = ParallelLogAverage(id, pool, hdr_tex, chunk_size);

    TextureDescription tga_tex_desc = header;
    tga_tex_desc.Format = DataFormat::RGBA8UNorm;
    auto tga_tex_data = new uint8_t[header.Width*header.Height*DataFormatElementSize(header.Format)];

    std::unique_ptr<Texture> tga_tex(new Texture(tga_tex_desc, tga_tex_data));
    auto tga_tex_ptr = tga_tex.get();

    float exp_factor = exp_mod / mean_value;
    auto hdr_processed = CreateParallelForLoop2D(width, height, chunk_size,
                            [&convert, exp_factor, hdr_tex, tga_tex_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
                            {
                                convert(hdr_tex, tga_tex_ptr, exp_factor, worker_id, x, y);
                            });

    if(out_exp_factor)
        *out_exp_factor = exp_factor;

    pool.enqueueTask(&hdr_processed);

    pool.waitAndHelp(id, &hdr_processed);
    return tga_tex.release();
}

Texture* ParallelConvertHDRToSRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod, float* out_exp_factor)
{
    return ParallelConvertHDR(id, pool, hdr_tex, chunk_size, exp_mod, out_exp_factor,
                              [](const Texture* hdr_tex, Texture* tga_tex_ptr, float exp_factor, uint32_t worker_id, uint32_t x, uint32_t y)
                              {
                                  auto value = exp_factor*hdr_tex->fetchRGB(x, y);

                                  auto tone_mapped = ReinhardOperator(value);

                                  Vector4 v4_color{ tone_mapped.x, tone_mapped.y, tone_mapped.z, hdr_tex->fetchAlpha(x, y) };
        
                                  auto color = ToColor(ConvertLinearToSRGB(v4_color));

                                  tga_tex_ptr->writeValue(color, x*sizeof(color), y);
                              });

}

Texture* ParallelConvertHDRToLDRSRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod, float* out_exp_factor)
{
    return ParallelConvertHDR(id, pool, hdr_tex, chunk_size, exp_mod, out_exp_factor,
        [](const Texture* hdr_tex, Texture* tga_tex_ptr, float exp_factor, uint32_t worker_id, uint32_t x, uint32_t y)
        {
            auto value = exp_factor*hdr_tex->fetchRGB(x, y);

            Vector4 v4_color{ value.x, value.y, value.z, hdr_tex->fetchAlpha(x, y) };
        
            auto color = ToColor(ConvertLinearToSRGB(v4_color));

            tga_tex_ptr->writeValue(color, x*sizeof(color), y);
        });
}

Texture* ParallelConvertHDRToLDRRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod, float* out_exp_factor)
{
    return ParallelConvertHDR(id, pool, hdr_tex, chunk_size, exp_mod, out_exp_factor,
        [](const Texture* hdr_tex, Texture* tga_tex_ptr, float exp_factor, uint32_t worker_id, uint32_t x, uint32_t y)
        {
            auto value = exp_factor*hdr_tex->fetchRGB(x, y);

            Vector4 v4_color{ value.x, value.y, value.z, hdr_tex->fetchAlpha(x, y) };
        
            auto color = ToColor(v4_color);

            tga_tex_ptr->writeValue(color, x*sizeof(color), y);
        });
}

Texture* ParallelConvertToLinearRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size)
{
    auto& header = hdr_tex->getHeader();
    auto width = header.Width,
         height = header.Height;

    TextureDescription tga_tex_desc = header;
    tga_tex_desc.Format = DataFormat::RGBA8UNorm;
    auto tga_tex_data = new uint8_t[header.Width*header.Height*DataFormatElementSize(header.Format)];

    std::unique_ptr<Texture> tga_tex(new Texture(tga_tex_desc, tga_tex_data));
    auto tga_tex_ptr = tga_tex.get();

    auto process = CreateParallelForLoop2D(width, height, chunk_size,
                            [hdr_tex, tga_tex_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
                            {
                                auto value = hdr_tex->fetchRGB(x, y);

                                Vector4 v4_color{ value.x, value.y, value.z, hdr_tex->fetchAlpha(x, y) };
        
                                auto color = ToColor(v4_color);

                                tga_tex_ptr->writeValue(color, x*sizeof(color), y);
                            });

    pool.enqueueTask(&process);

    pool.waitAndHelp(id, &process);
    return tga_tex.release();
}

Texture* ParallelConvertHDRToLuminance8(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod)
{
    auto& header = hdr_tex->getHeader();
    auto width = header.Width,
         height = header.Height;
    auto mean_value = ParallelLogAverage(id, pool, hdr_tex, chunk_size);

    TextureDescription tga_tex_desc = header;
    tga_tex_desc.Format = DataFormat::R8UNorm;
    auto tga_tex_data = new uint8_t[header.Width*header.Height*DataFormatElementSize(header.Format)];

    std::unique_ptr<Texture> tga_tex(new Texture(tga_tex_desc, tga_tex_data));
    auto tga_tex_ptr = tga_tex.get();

    float exp_factor = exp_mod / mean_value;
    auto hdr_processed = CreateParallelForLoop2D(width, height, chunk_size,
                                                 [&exp_factor, hdr_tex, tga_tex_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto value = exp_factor*hdr_tex->fetchRGB(x, y);

        auto tone_mapped = ReinhardOperator(value);

        float luminance = Tempest::ConvertLinearToSLuminance(Tempest::RGBToLuminance(tone_mapped));
        
        uint8_t lum = (uint8_t)Tempest::Clampf(luminance*255.0f, 0.0f, 255.0f);
        tga_tex_ptr->writeValue(lum, x*sizeof(lum), y);
    });

    pool.enqueueTask(&hdr_processed);

    pool.waitAndHelp(id, &hdr_processed);
    return tga_tex.release();
}

Texture* ParallelConvertHDRToLDRLuminance8(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod)
{
    auto& header = hdr_tex->getHeader();
    auto width = header.Width,
         height = header.Height;
    auto mean_value = ParallelLogAverage(id, pool, hdr_tex, chunk_size);

    TextureDescription tga_tex_desc = header;
    tga_tex_desc.Format = DataFormat::R8UNorm;
    auto tga_tex_data = new uint8_t[header.Width*header.Height*DataFormatElementSize(header.Format)];

    std::unique_ptr<Texture> tga_tex(new Texture(tga_tex_desc, tga_tex_data));
    auto tga_tex_ptr = tga_tex.get();

    float exp_factor = exp_mod / mean_value;
    auto hdr_processed = CreateParallelForLoop2D(width, height, chunk_size,
                                                 [&exp_factor, hdr_tex, tga_tex_ptr](uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto value = exp_factor*hdr_tex->fetchRGB(x, y);
        float luminance = Tempest::ConvertLinearToSLuminance(Tempest::RGBToLuminance(value));
        
        uint8_t lum = (uint8_t)Tempest::Clampf(luminance*255.0f, 0.0f, 255.0f);
        tga_tex_ptr->writeValue(lum, x*sizeof(lum), y);
    });

    pool.enqueueTask(&hdr_processed);

    pool.waitAndHelp(id, &hdr_processed);
    return tga_tex.release();
}
}
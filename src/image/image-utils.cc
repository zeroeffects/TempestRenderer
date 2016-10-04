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

#include "tempest/image/image-utils.hh"
#include "tempest/graphics/texture.hh"

#include "kissfft/kiss_fft.h"
#include "kissfft/tools/kiss_fftnd.h"
#include "kissfft/tools/kiss_fftndr.h"

#include <memory>

namespace Tempest
{
Texture* PowerSpectrumTexture(const Texture& tex, DataFormat format)
{
    auto& hdr = tex.getHeader();

    if(DataFormatChannels(hdr.Format) > 1 || DataFormatChannels(format) > 1)
        return nullptr;

    size_t tex_area = (size_t)hdr.Width*hdr.Height;

    const float* in_vec;
    std::unique_ptr<float[]> data;
    if(hdr.Format != DataFormat::R32F)
    {
        data = decltype(data)(new float[tex_area]);
        for(uint32_t y = 0, yend = hdr.Height; y < yend; ++y)
        {
            for(uint32_t x = 0, xend = hdr.Width; x < xend; ++x)
            {
                data[y*xend + x] = tex.fetchRed(x, y);
            }
        }
        in_vec = data.get();
    }
    else
    {
        in_vec = reinterpret_cast<const float*>(tex.getData());
    }

    std::unique_ptr<kiss_fft_cpx[]> result_vec(new kiss_fft_cpx[tex_area]);

    int dims[] =
    {
        hdr.Height,
        hdr.Width
    };

    /*
    // TODO-if-ever: this function seems to be broken
    auto cfg = CREATE_SCOPED(kiss_fftndr_cfg, ::kiss_fftr_free);
    cfg = kiss_fftndr_alloc(dims, TGE_FIXED_ARRAY_SIZE(dims), false, nullptr, nullptr);
    kiss_fftndr(cfg, in_vec, result_vec.get());

#if 0
    std::unique_ptr<float[]> interm_data(new float[tex_area]);
    cfg = kiss_fftndr_alloc(dims, TGE_FIXED_ARRAY_SIZE(dims), true, nullptr, nullptr);
    kiss_fftndri(cfg, result_vec.get(), interm_data.get());

     for(size_t idx = 0; idx < tex_area; ++idx)
    {
        auto& result = interm_data[idx];
        result /= tex_area;
    }

    for(uint32_t idx = 0; idx < tex_area; ++idx)
    {
        TGE_ASSERT(Tempest::ApproxEqual(interm_data[idx], in_vec[idx], 1e-3f), "Broken Fourier transform");
    }
#endif

    std::unique_ptr<kiss_fft_cpx[]> result_vec2(new kiss_fft_cpx[tex_area]),
                                    result_vec3(new kiss_fft_cpx[tex_area]);
    auto cfg_plain = kiss_fft_alloc(hdr.Width, false, nullptr, nullptr);
    for(size_t idx = 0; idx < hdr.Height; ++idx)
    {
        size_t offset = idx*hdr.Width;
        kiss_fft_stride(cfg_plain, in_cmpx.get() + offset, result_vec3.get() + offset, 1);
    }
    cfg_plain = kiss_fft_alloc(hdr.Height, false, nullptr, nullptr);
    for(size_t idx = 0; idx < hdr.Width; ++idx)
    {
        kiss_fft_stride(cfg_plain, result_vec3.get() + idx, result_vec2.get() + idx, hdr.Width);
    }

    for(size_t idx = 0; idx < tex_area; ++idx)
    {
        TGE_ASSERT(Tempest::ApproxEqual(result_vec[idx].r, result_vec2[idx].r, 1e-1f) &&
                   Tempest::ApproxEqual(result_vec[idx].i, result_vec2[idx].i, 1e-1f), "WTF");
    }
    */

    std::unique_ptr<kiss_fft_cpx[]> in_cmpx(new kiss_fft_cpx[tex_area]);
    for(size_t idx = 0; idx < tex_area; ++idx)
    {
        auto& in_val = in_cmpx[idx];
        in_val.r = in_vec[idx];
        in_val.i = 0.0f;
    }

    auto cfg_nd = kiss_fftnd_alloc(dims, TGE_FIXED_ARRAY_SIZE(dims), false, nullptr, nullptr);
    kiss_fftnd(cfg_nd, in_cmpx.get(), result_vec.get());

    switch(format)
    {
    case DataFormat::R32F:
    {
        if(!data)
        {
            data = decltype(data)(new float[tex_area]);
        }

        for(uint32_t idx = 0; idx < tex_area; ++idx)
        {
            auto& value = result_vec[idx];
            data[idx] = value.r*value.r + value.i*value.i;
        }

        Tempest::TextureDescription out_hdr = hdr;
        out_hdr.Format = format;

        return new Texture(out_hdr, reinterpret_cast<uint8_t*>(data.release()));
    } break;
    default:
        TGE_ASSERT(false, "Unsupported");
    }

    return nullptr;

}

template<class T>
Texture* GaussianBlurWrapImpl(Texture& tex)
{
    auto& hdr = tex.getHeader();
    auto* data = reinterpret_cast<T*>(tex.getData());
    std::unique_ptr<T[]> new_data(new T[hdr.Width*hdr.Height]);
    for(int64_t y = 0, yend = hdr.Height; y < yend; ++y)
    {
        for(int64_t x = 0, xend = hdr.Width; x < xend; ++x)
        {
            T total{};
            total += data[Modulo(y - 1, yend)*xend + Modulo(x - 1, xend)]*(1.0f/16.0f);
            total += data[Modulo(y - 1, yend)*xend + x]*(1.0f/8.0f);
            total += data[Modulo(y - 1, yend)*xend + Modulo(x + 1, xend)]*(1.0f/16.0f);
            
            total += data[y*xend + Modulo(x - 1, xend)]*(1.0f/8.0f);
            total += data[y*xend + x]*(1.0f/4.0f);
            total += data[y*xend + Modulo(x + 1, xend)]*(1.0f/8.0f);

            total += data[Modulo(y + 1, yend)*xend + Modulo(x - 1, xend)]*(1.0f/16.0f);
            total += data[Modulo(y + 1, yend)*xend + x]*(1.0f/8.0f);
            total += data[Modulo(y + 1, yend)*xend + Modulo(x + 1, xend)]*(1.0f/16.0f);

            new_data[y*xend + x] = total;
        }
    }
    return new Texture(hdr, reinterpret_cast<uint8_t*>(new_data.release()));
}

Texture* GaussianBlurWrap(Texture& tex)
{
    auto& hdr = tex.getHeader();
    switch(hdr.Format)
    {
    case DataFormat::R32F: return GaussianBlurWrapImpl<float>(tex);
    case DataFormat::RG32F: return GaussianBlurWrapImpl<Tempest::Vector2>(tex);
    case DataFormat::RGB32F: return GaussianBlurWrapImpl<Tempest::Vector3>(tex);
    case DataFormat::RGBA32F: return GaussianBlurWrapImpl<Tempest::Vector4>(tex);
    default:
        TGE_ASSERT(false, "Unsupported format");
    }
    return nullptr;
}
}
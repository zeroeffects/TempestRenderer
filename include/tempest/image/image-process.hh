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

#ifndef _TEMPEST_IMAGE_PROCESS_HH_
#define _TEMPEST_IMAGE_PROCESS_HH_

#include "tempest/image/image.hh"

namespace Tempest
{
template<class TData, class TOperator>
bool CrossFadeSaveImageTyped(const Tempest::TextureDescription& tex_desc, void* data, size_t cross_fade_pixels, TOperator& op, const Tempest::Path& filepath)
{
    TGE_ASSERT(Tempest::DataFormatElementSize(tex_desc.Format) == sizeof(TData), "Invalid data type");
    if(Tempest::DataFormatElementSize(tex_desc.Format) != sizeof(TData))
        return false;

    Tempest::TextureDescription mod_tex_desc = tex_desc;
    mod_tex_desc.Width -= static_cast<uint16_t>(2*cross_fade_pixels);
    mod_tex_desc.Height -= static_cast<uint16_t>(2*cross_fade_pixels);

    auto data_vec = reinterpret_cast<TData*>(data);
    std::unique_ptr<TData[]> result_vec(new TData[mod_tex_desc.Width*mod_tex_desc.Height]);
    for(size_t y = 0, yend = mod_tex_desc.Height; y < yend; ++y)
    {
        for(size_t x = 0, xend = mod_tex_desc.Width; x < xend; ++x)
        {
            result_vec[y*mod_tex_desc.Width + x] = data_vec[(y + cross_fade_pixels)*tex_desc.Width + x + cross_fade_pixels];
        }
    }

    for(size_t y = 0, yend = mod_tex_desc.Height; y < yend; ++y)
    {
        for(size_t x = 0, xend = cross_fade_pixels; x < xend; ++x)
        {
            {
            float t = static_cast<float>(x + 1 + cross_fade_pixels)/(2*(cross_fade_pixels + 1) - 1);
            auto& cur_vec = result_vec[y*mod_tex_desc.Width + x];
            auto& other_vec = data_vec[(y + cross_fade_pixels)*tex_desc.Width + x + mod_tex_desc.Width + cross_fade_pixels];
            cur_vec = op(other_vec, cur_vec, t);
            }

            {
            float t = static_cast<float>(x + 1)/(2*(cross_fade_pixels + 1) - 1);
            auto& cur_vec = result_vec[y*mod_tex_desc.Width + (x + mod_tex_desc.Width - cross_fade_pixels)];
            auto& other_vec = data_vec[(y + cross_fade_pixels)*tex_desc.Width + x];
            cur_vec = op(cur_vec, other_vec, t);
            }
        }
    }
            
    for(size_t y = 0, yend = cross_fade_pixels; y < yend; ++y)
    {
        for(size_t x = 0, xend = mod_tex_desc.Width; x < xend; ++x)
        {
            {
            float t = static_cast<float>(y + 1 + cross_fade_pixels)/(2*(cross_fade_pixels + 1) - 1);
            auto& cur_vec = result_vec[y*mod_tex_desc.Width + x];
            auto& other_vec = data_vec[(y + mod_tex_desc.Height + cross_fade_pixels)*tex_desc.Width + x + cross_fade_pixels];
            cur_vec = op(other_vec, cur_vec, t);
            }

            {
            float t = static_cast<float>(y + 1)/(2*(cross_fade_pixels + 1) - 1);
            auto& cur_vec = result_vec[(y + mod_tex_desc.Height - cross_fade_pixels)*mod_tex_desc.Width + x];
            auto& other_vec = data_vec[y*tex_desc.Width + x + cross_fade_pixels];
            cur_vec = op(cur_vec, other_vec, t);
            }
        }
    }

    return Tempest::SaveImage(mod_tex_desc, result_vec.get(), filepath);
}
}

#endif // _TEMPEST_IMAGE_PROCESS_HH_
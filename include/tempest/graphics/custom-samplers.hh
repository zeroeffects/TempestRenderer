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

#include "tempest/math/quaternion.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/math/functions.hh"

namespace Tempest
{
inline EXPORT_CUDA Tempest::Quaternion SampleQuaternionSlerp(const void* tex, uint32_t tex_width, uint32_t tex_height, const Tempest::Vector2& tc)
{
    float tex_width_f = static_cast<float>(tex_width);
    float tex_height_f = static_cast<float>(tex_height);

    Tempest::Vector2 tc_unorm{ tex_width_f*tc.x - 0.5f, tex_height_f*tc.y - 0.5f };

    auto x_trunc = FastFloorToInt(tc_unorm.x),
         y_trunc = FastFloorToInt(tc_unorm.y);

    int x = Modulo(x_trunc, (int)tex_width);
    int y = Modulo(y_trunc, (int)tex_height);
    
    float tx = tc_unorm.x - (float)x_trunc,
          ty = tc_unorm.y - (float)y_trunc;

    Tempest::Vector2 tc_norm_trunc_shift{ (x + 1.0f)/tex_width_f, (y + 1.0f)/tex_height_f };

    Tempest::Quaternion q00, q01, q10, q11;

    {
    auto comp0 = Gather2D(tex, tc_norm_trunc_shift, 0);
    
    q00.x = comp0.w;
    q01.x = comp0.z;
    q10.x = comp0.x;
    q11.x = comp0.y;
    }

    {
    auto comp1 = Gather2D(tex, tc_norm_trunc_shift, 1);
     
    q00.y = comp1.w;
    q01.y = comp1.z;
    q10.y = comp1.x;
    q11.y = comp1.y;
    }

    {
    auto comp2 = Gather2D(tex, tc_norm_trunc_shift, 2);
    
    q00.z = comp2.w;
    q01.z = comp2.z;
    q10.z = comp2.x;
    q11.z = comp2.y;
    }

    {
    auto comp3 = Gather2D(tex, tc_norm_trunc_shift, 3);

    q00.w = comp3.w;
    q01.w = comp3.z;
    q10.w = comp3.x;
    q11.w = comp3.y;
    }

    return Tempest::Normalize(Slerp(Slerp(q00, q01, tx),
                                    Slerp(q10, q11, tx), ty));;
}
}
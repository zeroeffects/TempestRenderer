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

#ifndef _TEMPEST_HDR_HH_
#define _TEMPEST_HDR_HH_

#include "tempest/math/vector3.hh"

namespace Tempest
{
inline EXPORT_CUDA Vector3 ReinhardOperator(const Tempest::Vector3& color)
{
    return color / (1.0f + color);
}

inline EXPORT_CUDA Vector3 ModifiedReinhardOperator(const Tempest::Vector3& color, float white)
{
    return color * (1.0f + color/(white*white)) / (1.0f + color);
}

namespace Uncharted2ToneMapping
{
const float A = 0.15f;
const float B = 0.50f;
const float C = 0.10f;
const float D = 0.20f;
const float E = 0.02f;
const float F = 0.30f;
const float W = 11.2f;

inline EXPORT_CUDA Vector3 Impl(const Vector3& x)
{
    return ((x*(A*x + C*B) + D*E)/(x*(A*x + B) + D*F)) - E/F;
}

inline EXPORT_CUDA float Impl(float x)
{
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
}

inline EXPORT_CUDA Vector3 Uncharted2ToneMappingOperator(const Tempest::Vector3& color)
{
    const float ExposureBias = 2.0f;
    Vector3 x = ExposureBias*color;

    auto curr = Uncharted2ToneMapping::Impl(x);

    float white_scale = 1.0f/Uncharted2ToneMapping::Impl(Uncharted2ToneMapping::W);
    return curr*white_scale;
}

// Taken from Krzysztof Narkowicz great blog post on tone-mapping
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
inline EXPORT_CUDA Vector3 ACESFilm(const Tempest::Vector3& color)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return Tempest::Vector3Clamp((color*(a*color+b))/(color*(c*color+d)+e), 0.0f, 1.0f);
}

class ThreadPool;
class Texture;

Texture* ParallelConvertHDRToSRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod = 0.18f, float* out_exp_factor = nullptr);
Texture* ParallelConvertHDRToLDRSRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod = 0.18f, float* out_exp_factor = nullptr);
Texture* ParallelConvertHDRToLDRRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod = 0.18f, float* out_exp_factor = nullptr);
Texture* ParallelConvertToLinearRGB(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size);
Texture* ParallelConvertHDRToLuminance8(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod = 0.18f);
Texture* ParallelConvertHDRToLDRLuminance8(uint32_t id, ThreadPool& pool, const Texture* hdr_tex, uint32_t chunk_size, float exp_mod = 0.18f);
}

#endif // _TEMPEST_HDR_HH_
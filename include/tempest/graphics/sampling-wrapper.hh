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

#ifndef _TEMPEST_SAMPLING_WRAPPER_HH_
#define _TEMPEST_SAMPLING_WRAPPER_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/graphics/texture.hh"

#ifdef __CUDACC__
#   include <surface_functions.h>
#endif

namespace Tempest
{
#ifdef __CUDA_ARCH__
inline __device__ Spectrum SampleSpectrum(const void*  tex, const Vector2& tc)
{
    float4 color = tex2D<float4>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y);
    return RGBToSpectrum(reinterpret_cast<Vector3&>(color));
}

inline __device__ float SampleRed(const void* tex, const Vector2& tc)
{
    return tex2D<float>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y);
}

inline __device__ Vector2 SampleRG(const void* tex, const Vector2& tc)
{
    auto texel = tex2D<float2>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y);
    return *reinterpret_cast<Vector2*>(&texel);
}

inline __device__ Vector3 SampleRGB(const void* tex, const Vector2& tc)
{
    auto texel = tex2D<float4>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y);
    return *reinterpret_cast<Vector3*>(&texel);
}

inline __device__ float FetchRed(const void* tex, const uint32_t x, const uint32_t y)
{
    return {}; //TODO:IMPLEMENT!!!
}

inline __device__ Vector2 FetchRG(const void* tex, const uint32_t x, const uint32_t y)
{
    return {}; //TODO:IMPLEMENT!!!
}

inline __device__ Vector3 FetchRGB(const void* tex, const uint32_t x, const uint32_t y)
{
    return {}; //TODO:IMPLEMENT!!!
}

inline __device__ Vector4 FetchRGBA(const void* tex, const uint32_t x, const uint32_t y)
{
    return {}; //TODO:IMPLEMENT!!!
}

inline __device__ Vector4 SampleRGBA(const void* tex, const Vector2& tc)
{
	auto texel = tex2D<float4>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y);
    return *reinterpret_cast<Vector4*>(&texel);
}

inline __device__ void Surface2DWrite(const Vector3& val, void* tex, uint32_t x_bytes, uint32_t y)
{
    surf2Dwrite(val.x, reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes, y);
	surf2Dwrite(val.y, reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes +   sizeof(float), y);
	surf2Dwrite(val.z, reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes + 2*sizeof(float), y);
}

inline __device__ void Surface2DWrite(const Vector2& val, void* tex, uint32_t x_bytes, uint32_t y)
{
    surf2Dwrite(reinterpret_cast<const float2&>(val), reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes, y);
}

inline __device__ void Surface2DWrite(const Vector4& val, void* tex, uint32_t x_bytes, uint32_t y)
{
    surf2Dwrite(reinterpret_cast<const float4&>(val), reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes, y);
}

inline __device__ void Surface2DWrite(const Quaternion& val, void* tex, uint32_t x_bytes, uint32_t y)
{
    surf2Dwrite(reinterpret_cast<const float4&>(val), reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes, y);
}

template<class T>
inline __device__ void Surface2DWrite(T val, void* tex, uint32_t x_bytes, uint32_t y)
{
    surf2Dwrite(val, reinterpret_cast<cudaSurfaceObject_t>(tex), x_bytes, y);
}

inline __device__ Vector4 Gather2D(const void* tex, Vector2 tc, int comp)
{
    auto value = tex2Dgather<float4>(reinterpret_cast<cudaTextureObject_t>(tex), tc.x, tc.y, comp);
    return *reinterpret_cast<Vector4*>(&value);
}
#else
inline Spectrum SampleSpectrum(const void* tex, const Vector2& tc)
{
    return reinterpret_cast<const Texture*>(tex)->sampleSpectrum(tc);
}

inline float SampleRed(const void* tex, const Vector2& tc)
{
    return reinterpret_cast<const Texture*>(tex)->sampleRed(tc);
}

inline Vector2 SampleRG(const void* tex, const Vector2& tc)
{
    return reinterpret_cast<const Texture*>(tex)->sampleRG(tc);
}

inline Vector3 SampleRGB(const void* tex, const Vector2& tc)
{
    return reinterpret_cast<const Texture*>(tex)->sampleRGB(tc);
}

inline Vector4 SampleRGBA(const void* tex, const Vector2& tc)
{
    return reinterpret_cast<const Texture*>(tex)->sampleRGBA(tc);
}

inline float FetchRed(const void* tex, const uint32_t x, const uint32_t y)
{
    return reinterpret_cast<const Texture*>(tex)->fetchRed(x, y);
}

inline Vector2 FetchRG(const void* tex, const uint32_t x, const uint32_t y)
{
    return reinterpret_cast<const Texture*>(tex)->fetchRG(x, y);
}

inline Vector3 FetchRGB(const void* tex, const uint32_t x, const uint32_t y)
{
    return reinterpret_cast<const Texture*>(tex)->fetchRGB(x, y);
}

inline Vector4 FetchRGBA(const void* tex, const uint32_t x, const uint32_t y)
{
    return reinterpret_cast<const Texture*>(tex)->fetchRGBA(x, y);
}

template<class T>
inline void Surface2DWrite(T val, void* tex, uint32_t x_bytes, uint32_t y)
{
    reinterpret_cast<Texture*>(tex)->writeValue(val, x_bytes, y);
}

inline Vector4 Gather2D(const void* tex, Vector2 tc, int comp)
{
    return reinterpret_cast<const Texture*>(tex)->gather(tc, comp);
}
#endif
}

#endif // _TEMPEST_SAMPLING_WRAPPER_HH_

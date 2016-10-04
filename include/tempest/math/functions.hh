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

#ifndef _TEMPEST_MATH_FUNCTIONS_HH_
#define _TEMPEST_MATH_FUNCTIONS_HH_

#include <limits>
#include <cmath>
#include <random>
#include <complex>
#include <cfloat>
#include <algorithm>

#include "tempest/utils/assert.hh"
#include "tempest/compute/compute-macros.hh"

#ifdef __CUDA_ARCH__
struct half
{
	uint16_t value;

	__device__ operator float() const { return __half2float(value); }
};
#else
#	include "half/include/half.hpp"
typedef half_float::half half;
#endif

namespace Tempest
{
//! Mathematical constant
EXPORT_CUDA_CONSTANT float SqrtPi       = 1.772454f;
EXPORT_CUDA_CONSTANT float MathTau      = 6.283185f;
EXPORT_CUDA_CONSTANT float MathPi       = 3.141592f;
EXPORT_CUDA_CONSTANT float Sqrt2        = 1.41421356f;
EXPORT_CUDA_CONSTANT float RcpSqrt3     = 0.5773502691f;

inline EXPORT_CUDA float Length(float f) { return fabsf(f); }

#define TEMPEST_WEAK_FLOAT_EPSILON 1e-6f

inline EXPORT_CUDA float ApproxEqual(float lhs, float rhs, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return fabsf(rhs - lhs) <= epsilon;
}

inline EXPORT_CUDA float ApproxNotEqual(float lhs, float rhs, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return fabsf(rhs - lhs) > epsilon;
}

//s*
inline EXPORT_CUDA int FastFloorToInt(float f)
{
    int k = (int)f;
    uint32_t a = *((uint32_t*)&f) >> 31;
    int res = k - a;
    return res;
}

inline EXPORT_CUDA int64_t FastFloorToInt64(float f)
{
    int64_t k = (int64_t)f;
    uint32_t a = *((uint32_t*)&f) >> 31;
    int64_t res = k - a;
    return res;
}
/*/
inline int FastFloorToInt(float x)
{
    return (int)x - (x<0);
}
//*/

#ifdef __CUDA_ARCH__
#   define FastFloor(x) ((float)Tempest::FastFloorToInt(x))
#else
#   define FastFloor(x) ((float)Tempest::FastFloorToInt(x))
#endif

/*
// Not really faster than the other tricks
inline float FastFloor(float x)
{
    __m128 xsse = _mm_setr_ps(x, 0, 0, 0);
    xsse = _mm_floor_ss(xsse, xsse);
    return xsse.m128_f32[0];
}
*/

#ifdef __CUDA_ARCH__
inline EXPORT_CUDA float FastCeil(float x) { return ceilf(x); }
#   define MinD fmin
#   define MaxD fmax
#   define Minf fminf
#   define Maxf fmaxf
#   define Mini(a, b) ((a) < (b) ? (a) : (b))
#   define Maxi(a, b) ((a) > (b) ? (a) : (b))
#else
inline float FastCeil(float x)
{
    __m128 xsse = _mm_setr_ps(x, 0, 0, 0);
    xsse = _mm_ceil_ss(xsse, xsse);
    return _mm_cvtss_f32(xsse);
}
#   define MinD std::fmin
#   define MaxD std::fmax
#   define Minf std::fminf
#   define Maxf std::fmaxf
#   define Mini std::min
#   define Maxi std::max
#endif

inline EXPORT_CUDA float GenericMax(float lhs, float rhs) { return Maxf(lhs, rhs); }

inline EXPORT_CUDA float GenericSqrt(float val) { return sqrtf(val); }

inline EXPORT_CUDA float GenericLog10(float val) { return log10f(val); }

//! Converts degrees to radians
/*!
    \param val a floating-point number argument
    \return the angle in radians
*/
inline EXPORT_CUDA float ToRadians(float val) { return (val * MathPi) / 180.0f; }

inline EXPORT_CUDA float ToDegress(float val) { return (val * 180.0f) / MathPi; }

inline EXPORT_CUDA float SumArithmeticProgression(float start, float step, float step_count)
{
    return (start + step*step_count)*step_count/2;
}

template<class T>
inline EXPORT_CUDA T Clamp(T val, T minval, T maxval)
{
    return val <= minval ? minval : (val >= maxval ? maxval : val);
}

inline EXPORT_CUDA float Clampf(float val, float minval, float maxval)
{
    return Maxf(Minf(val, maxval), minval);
}

inline EXPORT_CUDA uint32_t rgbaR(uint32_t color)
{
    return color & 0xFF;
}

inline EXPORT_CUDA uint32_t rgbaG(uint32_t color)
{
    return (color >> 8) & 0xFF;
}

inline EXPORT_CUDA uint32_t rgbaB(uint32_t color)
{
    return (color >> 16) & 0xFF;
}

inline EXPORT_CUDA uint32_t rgbaA(uint32_t color)
{
    return (color >> 24) & 0xFF;
}

inline EXPORT_CUDA uint32_t rgba(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
    return r | (g << 8) | (b << 16) | (a << 24);
}

// SSE mayhaps?
inline EXPORT_CUDA uint32_t Interpolate(uint32_t c00, uint32_t c01, uint32_t c10, uint32_t c11, uint32_t s, uint32_t t)
{
    return ((c00 * (255u - s) + s * c01) * (255u - t) +
            (c10 * (255u - s) + s * c11) * t + 127u) / (255u*255u);
}

inline EXPORT_CUDA uint32_t InterpolateColor(uint32_t c00, uint32_t c01, uint32_t c10, uint32_t c11, uint32_t s, uint32_t t)
{
    return rgba(Interpolate(rgbaR(c00), rgbaR(c01), rgbaR(c10), rgbaR(c11), s, t),
                Interpolate(rgbaG(c00), rgbaG(c01), rgbaG(c10), rgbaG(c11), s, t),
                Interpolate(rgbaB(c00), rgbaB(c01), rgbaB(c10), rgbaB(c11), s, t),
                Interpolate(rgbaA(c00), rgbaA(c01), rgbaA(c10), rgbaA(c11), s, t));
}

inline EXPORT_CUDA float sfrand(unsigned int& seed)
{
    unsigned int a;
    unsigned m = seed *= 16807;
    a = (m&0x007fffff) | 0x40000000;
    return( *((float*)&a)*0.5f - 1.0f );
}

inline EXPORT_CUDA unsigned IntelFastRand(unsigned& seed)
{
    seed = (214013*seed+2531011);
	return (seed>>16)&0x7FFF;
}

inline EXPORT_CUDA uint32_t Hash32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;

  return h;
}

inline EXPORT_CUDA uint64_t Hash64(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
  
    return h;
}

// Lame but cheap
inline EXPORT_CUDA unsigned FastRandRange(unsigned min, unsigned max, unsigned& seed)
{
    seed = Hash32(seed);
    return min + (seed % (max - min));
}

inline EXPORT_CUDA float FloatIntelFastRand(unsigned& seed)
{
    return (float)IntelFastRand(seed)/0x7FFF;
}

#define FastFloatRand FloatIntelFastRand
#define FastUintRand FastRandRange

inline EXPORT_CUDA float GaussianNormalization(float stddev)
{
    return 0.398942280f / stddev; // 1.0f / sqrtf(2.0*MathPi)
}

inline EXPORT_CUDA float FresnelSchlick(float refl, float cos_theta)
{
    return refl + (1 - refl) * pow(1 - cos_theta, 5.0f);
}

// Basically cos derived from Snell's law
inline EXPORT_CUDA float CosTransmittance(float out_over_in_refr_idx, float cos_inc)
{
    return sqrtf(1.0f - (1.0f - cos_inc*cos_inc)/(out_over_in_refr_idx*out_over_in_refr_idx));
}

// Really nice explanation
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/#more-1921
inline EXPORT_CUDA float FresnelConductor(float refr_idx, float refr_idx_img,  float cos_inc)
{
   float cos_inc2 = cos_inc*cos_inc;
   float sin_inc2 = 1 - cos_inc2;
   float eta2 = refr_idx*refr_idx;
   float etak2 = refr_idx_img*refr_idx_img;

   float t0 = eta2 - etak2 - sin_inc2;
   float a2plusb2 = sqrt(t0*t0 + 4*eta2*etak2);
   float t1 = a2plusb2 + cos_inc2;
   float a = sqrt(0.5f*(a2plusb2 + t0));
   float t2 = 2*a*cos_inc;
   float r_perp = (t1 - t2)/(t1 + t2);

   float t3 = cos_inc2*a2plusb2 + sin_inc2*sin_inc2;
   float t4 = t2 * sin_inc2;   
   float r_parallel = r_perp*(t3 - t4)/(t3 + t4);

   return 0.5f*(r_perp + r_parallel);
}

inline EXPORT_CUDA float FresnelDielectric(float out_over_in_refr_idx, float cos_inc, float cos_trans)
{
    float r_parallel = (out_over_in_refr_idx*cos_inc - cos_trans)/(out_over_in_refr_idx*cos_inc + cos_trans);
    float r_perp = (cos_inc - out_over_in_refr_idx*cos_trans)/(cos_inc + out_over_in_refr_idx*cos_trans);
    return 0.5f*(r_parallel*r_parallel + r_perp*r_perp);
}

inline EXPORT_CUDA float FresnelDielectricBravais(float parallel_out_over_in_refr_idx, float perp_out_over_in_refr_idx, float cos_inc, float cos_trans)
{
    float r_parallel = (parallel_out_over_in_refr_idx*cos_inc - cos_trans)/(parallel_out_over_in_refr_idx*cos_inc + cos_trans);
    float r_perp = (cos_inc - perp_out_over_in_refr_idx*cos_trans)/(cos_inc + perp_out_over_in_refr_idx*cos_trans);
    return 0.5f*(r_parallel*r_parallel + r_perp*r_perp);
}

inline EXPORT_CUDA float SmoothStep(float lower, float upper, float value)
{
    float ext = upper - lower;
    float t = Clamp((value - lower)/ext, 0.0f, 1.0f); 
    return t*t*(3 - 2*t);
}

inline EXPORT_CUDA float Step(float edge, float value)
{
    return float(value >= edge);
}

inline void SinCos_Tan(float omega, float* s, float* c)
{
    float t = tanf(omega*0.5f);
    float t2 = t*t;
    float div = 1.0f/(1.0f + t2);
    *s = 2.0f*t*div;
    *c = (1.0f - t2)*div;
}

inline EXPORT_CUDA float Sinc(float angle)
{
    return angle == 0.0f ? 1.0f : sinf(angle)/angle;
}

inline void SinCos_Naive(float omega, float* s, float* c)
{
    *s = sinf(omega);
    *c = cosf(omega);
}

inline void SinCos_SqrtCorrect(float omega, float* s, float* c)
{
    float cos_theta = *c = cosf(omega);
    *s = (1.0f - 2.0f*(fmodf(fabsf(omega), 2.0f*MathPi) > MathPi))*sqrtf(1.0f - cos_theta*cos_theta);
}

#ifdef __CUDA_ARCH__
inline void EXPORT_CUDA dummy_function() {}
#   define FastSinCos dummy_function(), __sincosf
#else
#   define FastSinCos SinCos_Tan
#endif

#ifdef _MSC_VER
#   define BYTE_SWAP_16 _byteswap_ushort
#   define BYTE_SWAP_32 _byteswap_ulong
#   define BYTE_SWAP_64 _byteswap_uint64
#else
#   define BYTE_SWAP_16 __builtin_bswap16
#   define BYTE_SWAP_32 __builtin_bswap32
#   define BYTE_SWAP_64 __builtin_bswap64
#endif

// Complex roots are not considered
inline void EXPORT_CUDA SolveDepressedCubic(float A, float B, float* roots, uint32_t* root_count)
{
    *root_count = 0;
    float interm_c = A/3.0f;

    float D = B*B + 4.0f*interm_c*interm_c*interm_c;
    if(D < 0.0f)
    {
        float theta = atan2f(sqrtf(-D), -B)/3.0f;
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        float mag = -2.0f*sqrtf(-interm_c);
        float r0 = roots[(*root_count)++] = mag*cos_theta;
        float cos_theta_240 = -cos_theta*0.5f;
        float sin_theta_240 = sin_theta*0.8660254037844f;
        roots[(*root_count)++] = mag*(cos_theta_240 - sin_theta_240);
        roots[(*root_count)++] = mag*(cos_theta_240 + sin_theta_240);
    }
    else if(D < 1e-6f)
    {
        float u = -B*0.5f;
        float t = cbrtf(u);
        float s = interm_c/t;

        roots[(*root_count)++] = s - t;
    }
    else
    {
        float sqrt_D = sqrtf(D);

        float u0 = (-B - sqrt_D)*0.5f;

        float t0 = cbrtf(u0);
        float s0 = interm_c/t0;
        float r0 = roots[(*root_count)++] = s0 - t0;

        // after polynomial division
        // a_quadratic = 1;
        // b_quadratic = r0;
        // c_quadratic = r0*r0 + A;

        float D_quadratic = -4.0f*A - 3*r0*r0;
        if(D_quadratic < 0.0f)
            return;

        if(D_quadratic < 1e-6f)
        {
            roots[(*root_count)++] = -r0*0.5f;
            return;
        }

        float sqrt_Dq = sqrtf(D_quadratic);
        roots[(*root_count)++] = (-r0 + sqrt_Dq)*0.5f;
        roots[(*root_count)++] = (-r0 - sqrt_Dq)*0.5f;
    }
}

inline uint32_t EXPORT_CUDA NextPowerOf2(uint32_t v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    ++v;
    return v;
}

// Complex roots are not considered
inline void EXPORT_CUDA SolveTriviallyDepressedCubic(float coeff_3, float coeff_1, float coeff_0, float* roots, uint32_t* root_count)
{
    *root_count = 0;
    if(fabs(coeff_3) > 1e-6f)
    {
        SolveDepressedCubic(coeff_1/coeff_3, -coeff_0/coeff_3, roots, root_count);
    }
    else if(fabs(coeff_1) > 1e-6f)
    {
        roots[(*root_count)++] = -coeff_0/coeff_1;
    }
}

inline float EXPORT_CUDA Cos2x(float cos_theta) { return 2.0f*cos_theta*cos_theta - 1.0f; }
// Ignores sign
inline float EXPORT_CUDA FastCos0_5x(float cos_theta) { return sqrtf((1.0f + cos_theta)*0.5f); }

inline float EXPORT_CUDA LinearInterpolate(float lower_value, float upper_value, float mix_factor)
{
    return lower_value + mix_factor*(upper_value - lower_value);
}

template<class T>
inline T EXPORT_CUDA GenericLinearInterpolate(T lower_value, T upper_value, float mix_factor)
{
    return (1.0f - mix_factor)*lower_value + mix_factor*upper_value;
}

template<int idx>
inline float ElementFloat(__m128 var)
{
    int elem = _mm_extract_ps(var, idx);
    return *((float*)&elem);
}

template<int idx>
inline int ElementInt(__m128i var)
{
    return _mm_extract_epi32(var, idx);
}

template<int idx>
inline int ElementInt(__m128 var)
{
    return _mm_extract_ps(var, idx);
}

template <typename T>
inline EXPORT_CUDA void Swap(T& a, T& b)
{
  T c = std::move(a); a = std::move(b); b = std::move(c);
}

inline void SwapBounds(float& b0, float& b1)
{
	if(b0 > b1)
		std::swap(b0, b1);
}

inline EXPORT_CUDA float Sign(float x) { return x >= 0.0f ? 1.0f : -1.0f; }

inline EXPORT_CUDA float section(float h, float r = 1.0f) // returns the positive root of intersection of line y = h with circle centered at the origin and radius r
{
    TGE_ASSERT(r >= 0, "Radius needs to be positive for section calculation"); // assume r is positive, leads to some simplifications in the formula below (can factor out r from the square root)
    return (h < r)? sqrtf(r * r - h * h) : 0; // http://www.wolframalpha.com/input/?i=r+*+sin%28acos%28x+%2F+r%29%29+%3D+h
}

inline EXPORT_CUDA float circle_segmentArea(float x, float h, float r = 1.0f) // indefinite integral of circle segment
{
    return 0.5f * (sqrtf(1.0f - x * x / (r * r)) * x * r + r * r * asinf(x / r) - 2.0f * h * x); // http://www.wolframalpha.com/input/?i=r+*+sin%28acos%28x+%2F+r%29%29+-+h
}

inline EXPORT_CUDA float circle_intersectionArea_infty(float x0, float x1, float h, float r) // area of intersection of an infinitely tall box with left edge at x0, right edge at x1, bottom edge at h and top edge at infinity, with circle centered at the origin with radius r
{
    if(x0 > x1)
        Swap(x0, x1); // this must be sorted otherwise we get negative area
    float s = section(h, r);
    return circle_segmentArea(Maxf(-s, Minf(s, x1)), h, r) - circle_segmentArea(Maxf(-s, Minf(s, x0)), h, r); // integrate the area
}

inline EXPORT_CUDA float circle_intersectionArea_finite(float x0, float x1, float y0, float y1, float r) // area of the intersection of a finite box with a circle centered at the origin with radius r
{
    if(y0 > y1)
        Swap(y0, y1); // this will simplify the reasoning
    if(y0 < 0) {
        if(y1 < 0)
            return circle_intersectionArea_finite(x0, x1, -y0, -y1, r); // the box is completely under, just flip it above and try again
        else
            return circle_intersectionArea_finite(x0, x1, 0, -y0, r) + circle_intersectionArea_finite(x0, x1, 0, y1, r); // the box is both above and below, divide it to two boxes and go again
    } else {
        TGE_ASSERT(y1 >= 0, "Coordinates set incorrectly, transform to upper half circle"); // y0 >= 0, which means that y1 >= 0 also (y1 >= y0) because of the swap at the beginning
        return circle_intersectionArea_infty(x0, x1, y0, r) - circle_intersectionArea_infty(x0, x1, y1, r); // area of the lower box minus area of the higher box
    }
}

inline EXPORT_CUDA float circle_intersectionArea(float x0, float x1, float y0, float y1, float circleOrg_x, float circleOrg_y, float radius) // area of the intersection of a general box with a general circle
{
    //Shift coords relative to circle center
    x0 -= circleOrg_x; x1 -= circleOrg_x;
    y0 -= circleOrg_y; y1 -= circleOrg_y;

    return circle_intersectionArea_finite(x0, x1, y0, y1, radius);
}

inline EXPORT_CUDA float circle_intersectionArea(float x0, float x1, float y0, float y1, float radius)
{
    //Convenience function for circle center at origin
    return circle_intersectionArea_finite(x0, x1, y0, y1, radius);
}

template<class T>
inline EXPORT_CUDA T Modulo(T lhs, T rhs)
{
    return ((lhs % rhs) + rhs) % rhs;
}

#ifndef _MSC_VER
#   define __lzcnt __builtin_clz
#   define __popcnt __builtin_popcount
#   define __popcnt64 __builtin_popcountll
#endif
}

#endif // _TEMPEST_MATH_FUNCTIONS_HH_

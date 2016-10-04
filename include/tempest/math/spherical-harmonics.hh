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

#ifndef _TEMPEST_SPHERICAL_HARMONICS_HH_
#define _TEMPEST_SPHERICAL_HARMONICS_HH_

#include "tempest/math/numerical-methods.hh"

#include <type_traits>

namespace Tempest
{
template<class T>
T Factorial(T val)
{
    if(val == T{})
        return 1;

    T total = val;
    while(--val > 0)
        total *= val;
    return total;
}

template<class T>
T DoubleFactorial(T val)
{
    if(val == T{})
        return 1;

    T total = val;
    while((val -= 2) > 0)
        total *= val;
    return total;
}

float AssociatedLegendrePolynomial(int32_t m, int32_t l, float value)
{
    if(m == 0 && l == 0)
        return 1;

    if(m == l)
    {
        return DoubleFactorial(2*m - 1)*powf(1 - value*value, m*0.5f);
    }

    if(l == m + 1)
    {
        return (1 + 2*m)*value*AssociatedLegendrePolynomial(m, m, value);
    }

    return ((2*l - 1)*value*AssociatedLegendrePolynomial(m, l - 1, value) - (l + m - 1)*AssociatedLegendrePolynomial(m, l - 2, value))/(l - m);
}

float SH00(const Vector3& dir)
{
    return 0.2820948f;
}

float SH_11(const Vector3& dir)
{
    return 0.4886025f*dir.y;
}

float SH01(const Vector3& dir)
{
    return 0.4886025f*dir.z;
}

float SH11(const Vector3& dir)
{
    return 0.4886025f*dir.x;
}

float SH_22(const Vector3& dir)
{
    return 1.0925484f*dir.x*dir.y;
}

float SH_12(const Vector3& dir)
{
    return 1.0925484f*dir.y*dir.z;
}

float SH02(const Vector3& dir)
{
    return 0.3153916f*(3.0f*dir.z*dir.z - 1.0f);
}

float SH12(const Vector3& dir)
{
    return 1.0925484f*dir.x*dir.z;
}

float SH22(const Vector3& dir)
{
    return 0.5462742f*(dir.x*dir.x - dir.y*dir.y);
}

float SH_33(const Vector3& dir)
{
    return 0.5900436f*(dir.y*(3.0f*dir.x*dir.x - dir.y*dir.y));
}

float SH_23(const Vector3& dir)
{
    return 2.8906114f*dir.y*dir.x*dir.z;
}

float SH_13(const Vector3& dir)
{
    return 0.4570458f*dir.y*(-1.0f + 5.0f*dir.z*dir.z);
}

float SH03(const Vector3& dir)
{
    return 0.3731763f*dir.z*(5.0f*dir.z*dir.z - 3.0f);
}

float SH13(const Vector3& dir)
{
    return 0.4570458f*dir.x*(-1.0f + 5.0f*dir.z*dir.z);
}

float SH23(const Vector3& dir)
{
    return 1.4453057f*(dir.x*dir.x - dir.y*dir.y)*dir.z;
}

float SH33(const Vector3& dir)
{
    return 0.5900435f*dir.x*(dir.x*dir.x - 3.0f*dir.y*dir.y);
}

float SphericalHarmonicCoefficient(int32_t m, int32_t l)
{
    TGE_ASSERT(abs(m) <= l, "Invalid spherical harmonic");
    
    float result = sqrtf((2*l + 1)*Factorial(l - abs(m))/(4.0f*MathPi*Factorial(l + abs(m))));

    if(m != 0)
        result *= Tempest::Sqrt2;

    return result;
}

float SphericalHarmonicPartialEvaluate(int32_t m, int32_t l, const Vector3& dir)
{
    TGE_ASSERT(abs(m) <= l, "Invalid spherical harmonic");
    float sin_theta = sqrtf(dir.x*dir.x + dir.y*dir.y);
    float sin_phi = sin_theta ? dir.y/sin_theta : 1.0f;
    float cos_phi = sin_theta ? dir.x/sin_theta : 0.0f;

    float cos_theta = dir.z;

    float phi = atan2f(sin_phi, cos_phi);

    if(m > 0)
        return cosf(m*phi)*AssociatedLegendrePolynomial(m, l, cos_theta);
    else if(m < 0)
        return sinf(abs(m)*phi)*AssociatedLegendrePolynomial(abs(m), l, cos_theta);
    else
        return AssociatedLegendrePolynomial(0, l, cos_theta);
}

float SphericalHarmonicEvaluate(int32_t m, int32_t l, const Vector3& dir)
{
    return SphericalHarmonicCoefficient(m, l)*SphericalHarmonicPartialEvaluate(m, l, dir);
}

template<class TFunc>
auto MonteCarloSphericalHarmonicsIntegrator(int32_t order, uint32_t samples, const TFunc& func) -> typename std::add_pointer<decltype(func(Vector3{}))>::type
{
    if(order <= 0)
        return {};

    typedef decltype(func(Vector3{})) output_type;
    auto* coefficients = new output_type[order*order];
    size_t idx = 0;
    for(int32_t cur_order = 0; cur_order < order; ++cur_order)
    {
        for(int32_t cur_degree = -cur_order; cur_degree <= cur_order; ++cur_degree, ++idx)
        {
            int32_t l = cur_order;
            int32_t m = cur_degree;

            float const_coef = SphericalHarmonicCoefficient(m, l);

            coefficients[idx] = StratifiedMonteCarloIntegratorSphere<output_type>(samples,
                                    [l, m, const_coef, &func](const Vector3& dir)
                                    {
                                        return const_coef*SphericalHarmonicPartialEvaluate(m, l, dir)*func(dir);
                                    });
        }
    }

    TGE_ASSERT(idx == order*order, "Invalid index");

    return coefficients;
}
}

#endif // _TEMPEST_SPHERICAL_HARMONICS_HH_
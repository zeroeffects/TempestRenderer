/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_SAMPLING1_HH_
#define _TEMPEST_SAMPLING1_HH_

#include <cmath>

#include "tempest/math/functions.hh"

namespace Tempest
{
// Based on implementation by V. Natarajan 
// http://home.online.no/~pjacklam/notes/invnorm/impl/natarajan/normsinv.h
// Original article 
// http://home.online.no/~pjacklam/notes/invnorm/
inline EXPORT_CUDA float NormInv(float p)
{
    #define  A1  (-3.969683028665376e+01)
    #define  A2   2.209460984245205e+02
    #define  A3  (-2.759285104469687e+02)
    #define  A4   1.383577518672690e+02
    #define  A5  (-3.066479806614716e+01)
    #define  A6   2.506628277459239e+00

    #define  B1  (-5.447609879822406e+01)
    #define  B2   1.615858368580409e+02
    #define  B3  (-1.556989798598866e+02)
    #define  B4   6.680131188771972e+01
    #define  B5  (-1.328068155288572e+01)

    #define  C1  (-7.784894002430293e-03)
    #define  C2  (-3.223964580411365e-01)
    #define  C3  (-2.400758277161838e+00)
    #define  C4  (-2.549732539343734e+00)
    #define  C5   4.374664141464968e+00
    #define  C6   2.938163982698783e+00

    #define  D1   7.784695709041462e-03
    #define  D2   3.224671290700398e-01
    #define  D3   2.445134137142996e+00
    #define  D4   3.754408661907416e+00

    #define P_LOW   0.02425
    /* P_high = 1 - p_low*/
    #define P_HIGH  0.97575

    double x = 0.0f;
    double q, r;
    if ((0 < p )  && (p < P_LOW)){
       q = sqrt(-2*log(p));
       x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
    }
    else if ((P_LOW <= p) && (p <= P_HIGH)){
        q = p - 0.5;
        r = q*q;
        x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q /(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
    }
    else
    {
        if ((P_HIGH < p)&&(p < 1)){
            q = sqrt(-2*log(1-p));
            x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
        }
    }

    return (float)x;
}

inline EXPORT_CUDA float GaussianSampling(float mean, float stddev, float r)
{
    return mean + stddev*NormInv(r);
}

// Mostly for debug purposes
inline EXPORT_CUDA float GaussianCDF(float mean, float stddev, float x)
{
    return 0.5f*(1.0f + erff((x - mean)/(stddev*Sqrt2)));
}

inline EXPORT_CUDA float GaussianPDF(float value, float stddev)
{
    float inv_stddev = 1.0f / stddev;
    float part = value * inv_stddev;
    return expf(- 0.5f * part * part) * inv_stddev * 0.398942280f;
}

inline EXPORT_CUDA float TruncatedGaussianNormalization(float len, float stddev)
{
    float sqrt_2pi = 2.5066282746f;
    return stddev*sqrt_2pi*erff(len/(stddev*Sqrt2));
}
}

#endif // _TEMPEST_SAMPLING1_HH_
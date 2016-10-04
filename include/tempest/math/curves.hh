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

#ifndef _TEMPEST_CURVES_HH_
#define _TEMPEST_CURVES_HH_

#include "tempest/math/matrix4.hh"

namespace Tempest
{
const Matrix4 BezierCurveMatrix( 1.0f,  0.0f,  0.0f, 0.0f,
                                -3.0f,  3.0f,  0.0f, 0.0f,
                                 3.0f, -6.0f,  3.0f, 0.0f,
                                -1.0f,  3.0f, -3.0f, 1.0f);

const Matrix4 CatmullRomCurveMatrix( 0.0f*0.5f,  2.0f*0.5f,  0.0f*0.5f,  0.0f*0.5f,
                                    -1.0f*0.5f,  0.0f*0.5f,  1.0f*0.5f,  0.0f*0.5f,
                                     2.0f*0.5f, -5.0f*0.5f,  4.0f*0.5f, -1.0f*0.5f,
                                    -1.0f*0.5f,  3.0f*0.5f, -3.0f*0.5f,  1.0f*0.5f);

template<class T>
T BezierCurvePoint(const T& p0, const T& p1, const T& p2, const T& p3, float t)
{
    float inv_t = 1 - t;
    float inv_t2 = inv_t*inv_t;
    float inv_t3 = inv_t2*inv_t;
    float t2 = t*t;
    float t3 = t2*t;
    return inv_t3*p0 + 3.0f*inv_t2*t*p1 + 3.0f*inv_t*t2*p2 + t3*p3;
}

template<class T>
T BezierCurveTangent(const T& p0, const T& p1, const T& p2, const T& p3, float t)
{
    float inv_t = 1 - t;
    float inv_t2 = inv_t*inv_t;
    float t2 = t*t;
    return -3.0f*inv_t2*p0 + (3.0f*inv_t2 - 6.0*t*inv_t)*p1 + (6.0f*t*inv_t - 3.0f*t2)*p2 + 3.0f*t2*p3;
}

template<class T>
T BezierCurveNormal(const T& p0, const T& p1, const T& p2, const T& p3, float t)
{
    float inv_t = 1 - t;
    float inv_t2 = inv_t*inv_t;
    float inv_t3 = inv_t2*inv_t;
    float t2 = t*t;
    float t3 = t2*t;
    return -6.0f*(inv_t*p0 + (2.0 - 3.0*t)*p1 + (3.0f*t - 1.0f)*p2 - t*p3);
}

// Don't use for anything other than debug purposes
template<class T>
T EvaluateCurve(const Matrix4& matrix, const T* coef, float t)
{
    return coef[0]*matrix.get(0, 0) + coef[1]*matrix.get(1, 0) + coef[2]*matrix.get(2, 0) + coef[3]*matrix.get(3, 0) + 
           t*(coef[0]*matrix.get(0, 1) + coef[1]*matrix.get(1, 1) + coef[2]*matrix.get(2, 1) + coef[3]*matrix.get(3, 1) +
               t*(coef[0]*matrix.get(0, 2) + coef[1]*matrix.get(1, 2) + coef[2]*matrix.get(2, 2) + coef[3]*matrix.get(3, 2)) +
                   t*(coef[0]*matrix.get(0, 3) + coef[1]*matrix.get(1, 3) + coef[2]*matrix.get(2, 3) + coef[3]*matrix.get(3, 3)));
}

template<class T>
void ConvertCurveCoeffient(const Matrix4& inc_matrix, const Matrix4& out_matrix, const T* in_coef, T* out_coef)
{
    Matrix4 correspondence_matrix = out_matrix.inverse()*inc_matrix;

    // So that inplace works
    T c0 = in_coef[0];
    T c1 = in_coef[1];
    T c2 = in_coef[2];
    T c3 = in_coef[3];

    out_coef[0] = c0*correspondence_matrix.get(0, 0) + c1*correspondence_matrix.get(1, 0) + c2*correspondence_matrix.get(2, 0) + c3*correspondence_matrix.get(3, 0);
    out_coef[1] = c0*correspondence_matrix.get(0, 1) + c1*correspondence_matrix.get(1, 1) + c2*correspondence_matrix.get(2, 1) + c3*correspondence_matrix.get(3, 1);
    out_coef[2] = c0*correspondence_matrix.get(0, 2) + c1*correspondence_matrix.get(1, 2) + c2*correspondence_matrix.get(2, 2) + c3*correspondence_matrix.get(3, 2);
    out_coef[3] = c0*correspondence_matrix.get(0, 3) + c1*correspondence_matrix.get(1, 3) + c2*correspondence_matrix.get(2, 3) + c3*correspondence_matrix.get(3, 3);
}

template<class T>
void CatmullRomToBezier(const T* in_coef, T* out_coef)
{
    ConvertCurveCoeffient(CatmullRomCurveMatrix, BezierCurveMatrix, in_coef, out_coef);
}

template<class T>
void BezierToCatmullRom(const T* in_coef, T* out_coef)
{
    ConvertCurveCoeffient(BezierCurveMatrix, CatmullRomCurveMatrix, in_coef, out_coef);
}

template<class T>
T CatmullRomCurvePoint(const T& p0, const T& p1, const T& p2, const T& p3, float t)
{
    return 0.5f*( (2.0f*p1) + t*((-p0 + p2) + t*((2.0f*p0 - 5.0f*p1 + 4.0f*p2 - p3) + t*(-p0 + 3.0f*p1 - 3.0f*p2 + p3))));
}
}

#endif // _TEMPEST_CURVES_HH_
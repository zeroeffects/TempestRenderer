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

#include <cstring>

#include "tempest/utils/config.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/math/dual-quaternion.hh"

namespace Tempest
{
Matrix4::Matrix4(const Quaternion& r, const Vector3& t)
{
    set(r,t);
}

Matrix4::Matrix4(const DualQuaternion& dq)
{
    Quaternion translation = 2.0f * dq.dual * Inverse(dq.non_dual);
    set(dq.non_dual, ExtractVector(translation));
}

void Matrix4::set(const Quaternion& r, const Vector3& t)
{
    float   xx = r.x*r.x,
            xy = r.x*r.y,
            xz = r.x*r.z,
            xw = r.x*r.w,
            yy = r.y*r.y,
            yz = r.y*r.z,
            yw = r.y*r.w,
            zz = r.z*r.z,
            zw = r.z*r.w;
    (*this)(0, 0) = 1.0f - 2.0f*(yy + zz);
    (*this)(0, 1) =        2.0f*(xy + zw);
    (*this)(0, 2) =        2.0f*(xz - yw);
    (*this)(0, 3) = 0.0f;
    (*this)(1, 0) =        2.0f*(xy - zw);
    (*this)(1, 1) = 1.0f - 2.0f*(xx + zz);
    (*this)(1, 2) =        2.0f*(yz + xw);
    (*this)(1, 3) = 0.0f;
    (*this)(2, 0) =        2.0f*(xz + yw);
    (*this)(2, 1) =        2.0f*(yz - xw);
    (*this)(2, 2) = 1.0f - 2.0f*(xx + yy);
    (*this)(2, 3) = 0.0f;
    (*this)(3, 0) = t.x;
    (*this)(3, 1) = t.y;
    (*this)(3, 2) = t.z;
    (*this)(3, 3) = 1.0f;
}

/********************/
/* Common SIMD code */
/********************/
#if defined(HAS_SSE) || defined(HAS_ARM_NEON)
Matrix4& Matrix4::operator+=(const Matrix4& mat)
{
    matrix[0] += mat.matrix[0];
    matrix[1] += mat.matrix[1];
    matrix[2] += mat.matrix[2];
    matrix[3] += mat.matrix[3];
    return *this;
}

Matrix4& Matrix4::operator-=(const Matrix4& mat)
{
    matrix[0] -= mat.matrix[0];
    matrix[1] -= mat.matrix[1];
    matrix[2] -= mat.matrix[2];
    matrix[3] -= mat.matrix[3];
    return *this;
}

Matrix4 Matrix4::operator-(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0] = matrix[0] - mat.matrix[0];
    r.matrix[1] = matrix[1] - mat.matrix[1];
    r.matrix[2] = matrix[2] - mat.matrix[2];
    r.matrix[3] = matrix[3] - mat.matrix[3];
    return r;
}

Matrix4 Matrix4::operator+(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0] = matrix[0] + mat.matrix[0];
    r.matrix[1] = matrix[1] + mat.matrix[1];
    r.matrix[2] = matrix[2] + mat.matrix[2];
    r.matrix[3] = matrix[3] + mat.matrix[3];
    return r;
}


Vector3 Matrix4::operator*(const Vector3& vec) const
{
    Vector3 res;

    simd128 vec4f;
    vec4f = matrix[0] * vec.x;
    vec4f += matrix[1] * vec.y;
    vec4f += matrix[2] * vec.z;
    vec4f += matrix[3];
    vec4f /= SHUFFLE_3(vec4f);

    res.x = vec4f.m128_f32[0];
    res.y = vec4f.m128_f32[1];
    res.z = vec4f.m128_f32[2];

    return res;
}

Vector4 Matrix4::operator*(const Vector4& vec) const
{
    Vector4 res;

    simd128 vec4f;
    vec4f  = matrix[0] * vec.x;
    vec4f += matrix[1] * vec.y;
    vec4f += matrix[2] * vec.z;
    vec4f += matrix[3] * vec.w;

    res.x = vec4f.m128_f32[0];
    res.y = vec4f.m128_f32[1];
    res.z = vec4f.m128_f32[2];
    res.w = vec4f.m128_f32[3];

    return res;
}

void Matrix4::rotateX(float pitch)
{

    float   s, c;
    FastSinCos(pitch, &s, &c);

    simd128 tmp = matrix[1];

    matrix[1] = tmp*c    + matrix[2]*s;
    matrix[2] = tmp*(-s) + matrix[2]*c;
}

void Matrix4::rotateY(float yaw)
{
    float   s, c;
    FastSinCos(yaw, &s, &c);

    simd128 tmp = matrix[0];

    matrix[0] = tmp*c + matrix[2]*(-s);
    matrix[2] = tmp*s + matrix[2]*c;
}

void Matrix4::rotateZ(float roll)
{
    float   s, c;
    FastSinCos(roll, &s, &c);

    simd128 tmp = matrix[0];

    matrix[0] = tmp*c    + matrix[1]*s;
    matrix[1] = tmp*(-s) + matrix[1]*c;
}

void Matrix4::rotate(float angle, const Vector3& axis)
{
    float   x = axis.x, y = axis.y, z = axis.z,
            s, c;
    FastSinCos(angle, &s, &c);
    float xc = (1 - c) * x,
          yc = (1 - c) * y;

    simd128 col0 = matrix[0], col1 = matrix[1], col2 = matrix[2];

    float m00 = xc*x + c,   m01 = xc*y - z*s,   m02 = xc*z + y*s,
          m10 = xc*y + z*s, m11 = yc*y + c,     m12 = yc*z -x*s,
          m20 = xc*z - y*s, m21 = yc*z + x*s,   m22 = z*z*(1-c) + c;
    matrix[0] = col0 * m00 + col1 * m10 + col2 * m20;
    matrix[1] = col0 * m01 + col1 * m11 + col2 * m21;
    matrix[2] = col0 * m02 + col1 * m12 + col2 * m22;
}

void Matrix4::rotate(const Vector3& euler)
{
    float sx, cx, sy, cy, sz, cz;

    FastSinCos(euler.x, &sx, &cx);
    FastSinCos(euler.y, &sy, &cy);
    FastSinCos(euler.z, &sz, &cz);

    float cx_sy = cx*sy,
          sx_sy = sx*sy;

    simd128 col0 = matrix[0], col1 = matrix[1], col2 = matrix[2];

    float   m00 =  cy*cz,            m01 = -cy*sz,            m02 =  sy,
            m10 =  sx_sy*cz + cx*sz, m11 = -sx_sy*sz + cx*cz, m12 = -sx*cy,
            m20 = -cx_sy*cz + sx*sz, m21 =  cx_sy*sz + sx*cz, m22 =  cx*cy;
    matrix[0] = col0 * m00 + col1 * m10 + col2 * m20;
    matrix[1] = col0 * m01 + col1 * m11 + col2 * m21;
    matrix[2] = col0 * m02 + col1 * m12 + col2 * m22;
}

void Matrix4::scale(const Vector3& s)
{
    matrix[0] *= s.x;
    matrix[1] *= s.y;
    matrix[2] *= s.z;
}

void Matrix4::scale(float s)
{
    matrix[0] *= s;
    matrix[1] *= s;
    matrix[2] *= s;
}

Vector3 Matrix4::transformRotate(const Vector3& vec) const
{
    Vector3 res;

    simd128 vec4f;
    vec4f = matrix[0] * vec.x;
    vec4f += matrix[1] * vec.y;
    vec4f += matrix[2] * vec.z;
    vec4f /= SHUFFLE_3(matrix[3]);

    res.x = vec4f.m128_f32[0];
    res.y = vec4f.m128_f32[1];
    res.z = vec4f.m128_f32[2];

    return res;
}

void Matrix4::translate(const Vector3& vec)
{
    matrix[3] = matrix[0] * vec.x + matrix[1] * vec.y + matrix[2] * vec.z + matrix[3];
}

void Matrix4::translate(const Vector2& vec)
{
    matrix[3] = (matrix[0] * vec.x + matrix[1] * vec.y + matrix[3]);
}

void Matrix4::translateX(float x)
{
    matrix[3] = matrix[0] * x + matrix[3];
}

void Matrix4::translateY(float y)
{
    matrix[3] = matrix[1] * y + matrix[3];
}

void Matrix4::translateZ(float z)
{
    matrix[3] = matrix[2] * z + matrix[3];
}

Matrix4 Matrix4::operator*(const Matrix4& mat2) const 
{
    Matrix4 r;
    r.matrix[0] = matrix[0] * SHUFFLE_0(mat2.matrix[0]);
    i_mad(r.matrix[0], matrix[1], SHUFFLE_1(mat2.matrix[0]));
    i_mad(r.matrix[0], matrix[2], SHUFFLE_2(mat2.matrix[0]));
    i_mad(r.matrix[0], matrix[3], SHUFFLE_3(mat2.matrix[0]));

    r.matrix[1] = matrix[0] * SHUFFLE_0(mat2.matrix[1]);
    i_mad(r.matrix[1], matrix[1], SHUFFLE_1(mat2.matrix[1]));
    i_mad(r.matrix[1], matrix[2], SHUFFLE_2(mat2.matrix[1]));
    i_mad(r.matrix[1], matrix[3], SHUFFLE_3(mat2.matrix[1]));

    r.matrix[2] = matrix[0] * SHUFFLE_0(mat2.matrix[2]);
    i_mad(r.matrix[2], matrix[1], SHUFFLE_1(mat2.matrix[2]));
    i_mad(r.matrix[2], matrix[2], SHUFFLE_2(mat2.matrix[2]));
    i_mad(r.matrix[2], matrix[3], SHUFFLE_3(mat2.matrix[2]));

    r.matrix[3] = matrix[0] * SHUFFLE_0(mat2.matrix[3]);
    i_mad(r.matrix[3], matrix[1], SHUFFLE_1(mat2.matrix[3]));
    i_mad(r.matrix[3], matrix[2], SHUFFLE_2(mat2.matrix[3]));
    i_mad(r.matrix[3], matrix[3], SHUFFLE_3(mat2.matrix[3]));

    return r;
}
#endif

/*******/
/* SSE */
/*******/
#if defined(HAS_SSE) && !defined(__CUDA_ARCH__)
Matrix4::Matrix4(float* _mat)
{
    matrix[0] = _mm_loadu_ps(_mat);
    matrix[1] = _mm_loadu_ps(_mat + 4);
    matrix[2] = _mm_loadu_ps(_mat + 8);
    matrix[3] = _mm_loadu_ps(_mat + 12);
}

Matrix4::Matrix4(float m00, float m10, float m20, float m30,
                 float m01, float m11, float m21, float m31,
                 float m02, float m12, float m22, float m32,
                 float m03, float m13, float m23, float m33)
{

    matrix[0] = _mm_setr_ps(m00, m01, m02, m03);
    matrix[1] = _mm_setr_ps(m10, m11, m12, m13);
    matrix[2] = _mm_setr_ps(m20, m21, m22, m23);
    matrix[3] = _mm_setr_ps(m30, m31, m32, m33);
}

void Matrix4::transposeSelf()
{
    _MM_TRANSPOSE4_PS(matrix[0], matrix[1], matrix[2], matrix[3]);
}

void Matrix4::identity()
{
    matrix[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    matrix[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    matrix[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    matrix[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::inverse() const
{
#define SHUFFLE_COL(mat, a0, a1, a2, a3) _mm_shuffle_ps(mat, mat, _MM_SHUFFLE(a0, a1, a2, a3))

    Matrix4 result;
    simd128 b0, b1, b2, b3, b4, b5;

    simd128 s00 = SHUFFLE_COL(matrix[0], 0, 0, 0, 1),
            s01 = SHUFFLE_COL(matrix[0], 1, 1, 2, 2),
            s02 = SHUFFLE_COL(matrix[0], 2, 3, 3, 3),
            s10 = SHUFFLE_COL(matrix[1], 0, 0, 0, 1),
            s11 = SHUFFLE_COL(matrix[1], 1, 1, 2, 2),
            s12 = SHUFFLE_COL(matrix[1], 2, 3, 3, 3),
            s20 = SHUFFLE_COL(matrix[2], 0, 0, 0, 1),
            s21 = SHUFFLE_COL(matrix[2], 1, 1, 2, 2),
            s22 = SHUFFLE_COL(matrix[2], 2, 3, 3, 3),
            s30 = SHUFFLE_COL(matrix[3], 0, 0, 0, 1),
            s31 = SHUFFLE_COL(matrix[3], 1, 1, 2, 2),
            s32 = SHUFFLE_COL(matrix[3], 2, 3, 3, 3);

    b0 = s10 * s21 * s32;
    b1 = s11 * s22 * s30;
    b2 = s12 * s20 * s31;
    
    b3 = s10 * s22 * s31;
    b4 = s11 * s20 * s32;
    b5 = s12 * s21 * s30;

    result.matrix[0] = b0 + b1 + b2 - b3 - b4 - b5;
    
    b0 = s00 * s21 * s32;
    b1 = s01 * s22 * s30;
    b2 = s02 * s20 * s31;
    
    b3 = s00 * s22 * s31;
    b4 = s01 * s20 * s32;
    b5 = s02 * s21 * s30;

    result.matrix[1] = b0 + b1 + b2 - b3 - b4 - b5;

    b0 = s00 * s11 * s32;
    b1 = s01 * s12 * s30;
    b2 = s02 * s10 * s31;
    
    b3 = s00 * s12 * s31;
    b4 = s01 * s10 * s32;
    b5 = s02 * s11 * s30;

    result.matrix[2] = b0 + b1 + b2 - b3 - b4 - b5;

    b0 = s00 * s11 * s22;
    b1 = s01 * s12 * s20;
    b2 = s02 * s10 * s21;
    
    b3 = s00 * s12 * s21;
    b4 = s01 * s10 * s22;
    b5 = s02 * s11 * s20;

    result.matrix[3] = b0 + b1 + b2 - b3 - b4 - b5;

    simd128 inv0 = _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f);
    simd128 inv1 = SHUFFLE_COL(inv0, 0, 1, 0, 1);

    result.matrix[0] *= inv0;
    result.matrix[1] *= inv1;
    result.matrix[2] *= inv0;
    result.matrix[3] *= inv1;

    simd128 det = result.matrix[0] * matrix[0];
    det += SHUFFLE_COL(det, 3, 3, 1, 1);
    det += SHUFFLE_COL(det, 2, 2, 2, 2);
    det  = SHUFFLE_COL(det, 0, 0, 0, 0);

    result.matrix[0] /= det;
    result.matrix[1] /= det;
    result.matrix[2] /= det;
    result.matrix[3] /= det;

    result.transposeSelf();

    return result;
}
/************/
/* ARM NEON */
/************/
#elif defined(HAS_ARM_NEON) && !defined(__CUDA_ARCH__)
Matrix4::Matrix4(float* _mat)
{
/*
    matrix[0] = _mm_loadu_ps(_mat);
    matrix[1] = _mm_loadu_ps(_mat + 4);
    matrix[2] = _mm_loadu_ps(_mat + 8);
    matrix[3] = _mm_loadu_ps(_mat + 12);
*/
}

Matrix4::Matrix4(float m00, float m01, float m02, float m03,
                 float m10, float m11, float m12, float m13,
                 float m20, float m21, float m22, float m23,
                 float m30, float m31, float m32, float m33)
{
/*
    matrix[0] = _mm_setr_ps(m00, m01, m02, m03);
    matrix[1] = _mm_setr_ps(m10, m11, m12, m13);
    matrix[2] = _mm_setr_ps(m20, m21, m22, m23);
    matrix[3] = _mm_setr_ps(m30, m31, m32, m33);
*/  
}

void Matrix4::transposeSelf()
{
    __asm__ ("vtrn.32 %[m0], %[m1]\n\t"
             "vtrn.32 %[m2], %[m3]\n\t"
             "vswp %[m0][1], %[m2][0]\n\t"
             "vswp %[m1][1], %[m3][0]\n\t"
             : [m0] "+w" (matrix[0]), [m1] "+w" (matrix[1]), [m2] "+w" (matrix[2]), [m3] "+w" (matrix[3])
             :
             );
}

void Matrix4::identity()
{
/*
    matrix[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    matrix[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    matrix[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    matrix[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
*/
}

Matrix4 Matrix4::inverse() const
{
/*
#define SHUFFLE_COL(mat, a0, a1, a2, a3) _mm_shuffle_ps(mat, mat, _MM_SHUFFLE(a0, a1, a2, a3))
    Matrix4 result;
    simd128 b0, b1, b2, b3, b4, b5;

    simd128 s00 = SHUFFLE_COL(matrix[0], 0, 0, 0, 1),
            s01 = SHUFFLE_COL(matrix[0], 1, 1, 2, 2),
            s02 = SHUFFLE_COL(matrix[0], 2, 3, 3, 3),
            s10 = SHUFFLE_COL(matrix[1], 0, 0, 0, 1),
            s11 = SHUFFLE_COL(matrix[1], 1, 1, 2, 2),
            s12 = SHUFFLE_COL(matrix[1], 2, 3, 3, 3),
            s20 = SHUFFLE_COL(matrix[2], 0, 0, 0, 1),
            s21 = SHUFFLE_COL(matrix[2], 1, 1, 2, 2),
            s22 = SHUFFLE_COL(matrix[2], 2, 3, 3, 3),
            s30 = SHUFFLE_COL(matrix[3], 0, 0, 0, 1),
            s31 = SHUFFLE_COL(matrix[3], 1, 1, 2, 2),
            s32 = SHUFFLE_COL(matrix[3], 2, 3, 3, 3);

    b0 = s10 * s21 * s32;
    b1 = s11 * s22 * s30;
    b2 = s12 * s20 * s31;
    
    b3 = s10 * s22 * s31;
    b4 = s11 * s20 * s32;
    b5 = s12 * s21 * s30;

    result.matrix[0] = b0 + b1 + b2 - b3 - b4 - b5;
    
    b0 = s00 * s21 * s32;
    b1 = s01 * s22 * s30;
    b2 = s02 * s20 * s31;
    
    b3 = s00 * s22 * s31;
    b4 = s01 * s20 * s32;
    b5 = s02 * s21 * s30;

    result.matrix[1] = b0 + b1 + b2 - b3 - b4 - b5;

    b0 = s00 * s11 * s32;
    b1 = s01 * s12 * s30;
    b2 = s02 * s10 * s31;
    
    b3 = s00 * s12 * s31;
    b4 = s01 * s10 * s32;
    b5 = s02 * s11 * s30;

    result.matrix[2] = b0 + b1 + b2 - b3 - b4 - b5;

    b0 = s00 * s11 * s22;
    b1 = s01 * s12 * s20;
    b2 = s02 * s10 * s21;
    
    b3 = s00 * s12 * s21;
    b4 = s01 * s10 * s22;
    b5 = s02 * s11 * s20;

    result.matrix[3] = b0 + b1 + b2 - b3 - b4 - b5;

    simd128 inv0 = _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f);
    simd128 inv1 = SHUFFLE_COL(inv0, 0, 1, 0, 1);

    result.matrix[0] *= inv0;
    result.matrix[1] *= inv1;
    result.matrix[2] *= inv0;
    result.matrix[3] *= inv1;

    simd128 det = result.matrix[0] * matrix[0];
    det += SHUFFLE_COL(det, 3, 3, 1, 1);
    det += SHUFFLE_COL(det, 2, 2, 2, 2);
    det  = SHUFFLE_COL(det, 0, 0, 0, 0);

    result.matrix[0] /= det;
    result.matrix[1] /= det;
    result.matrix[2] /= det;
    result.matrix[3] /= det;

    result.transposeSelf();

    return result;
*/
}
#endif
}

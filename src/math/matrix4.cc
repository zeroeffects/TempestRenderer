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

Matrix4& Matrix4::operator=(const Matrix4& m)
{
    memcpy(matrix, m.matrix, sizeof(matrix));
    return *this;
}

void Matrix4::set(const Quaternion& r, const Vector3& t)
{
    float   xx = r.coordinate.x*r.coordinate.x,
            xy = r.coordinate.x*r.coordinate.y,
            xz = r.coordinate.x*r.coordinate.z,
            xw = r.coordinate.x*r.coordinate.w,
            yy = r.coordinate.y*r.coordinate.y,
            yz = r.coordinate.y*r.coordinate.z,
            yw = r.coordinate.y*r.coordinate.w,
            zz = r.coordinate.z*r.coordinate.z,
            zw = r.coordinate.z*r.coordinate.w;
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
    (*this)(3, 0) = t.coordinate.x;
    (*this)(3, 1) = t.coordinate.y;
    (*this)(3, 2) = t.coordinate.z;
    (*this)(3, 3) = 1.0f;
}

Matrix4::Matrix4(const DualQuaternion& dq)
{
    Quaternion translation = 2.0f * dq.dual * dq.non_dual.inverse();
    set(dq.non_dual, translation.vector());
}

Matrix4& Matrix4::operator*=(const Matrix4& mat)
{
    *this = *this * mat;
    return *this;
}

bool Matrix4::operator==(const Matrix4& mat) const
{
    return approx_eq((*this)(0, 0), mat(0, 0)) && approx_eq((*this)(0, 1), mat(0, 1)) && approx_eq((*this)(0, 2), mat(0, 2)) && approx_eq((*this)(0, 3), mat(0, 3)) &&
           approx_eq((*this)(1, 0), mat(1, 0)) && approx_eq((*this)(1, 1), mat(1, 1)) && approx_eq((*this)(1, 2), mat(1, 2)) && approx_eq((*this)(1, 3), mat(1, 3)) &&
           approx_eq((*this)(2, 0), mat(2, 0)) && approx_eq((*this)(2, 1), mat(2, 1)) && approx_eq((*this)(2, 2), mat(2, 2)) && approx_eq((*this)(2, 3), mat(2, 3)) &&
           approx_eq((*this)(3, 0), mat(3, 0)) && approx_eq((*this)(3, 1), mat(3, 1)) && approx_eq((*this)(3, 2), mat(3, 2)) && approx_eq((*this)(3, 3), mat(3, 3));
}

Matrix4 Matrix4::transpose() const
{
    Matrix4 mat(*this);
    mat.transposeSelf();

    return mat;
}

void Matrix4::invertSelf()
{
    *this = inverse();
}

Vector3 Matrix4::relativeX() const
{
    return Vector3((*this)(0, 0), (*this)(0, 1), (*this)(0, 2));
}

Vector3 Matrix4::relativeY() const
{
    return Vector3((*this)(1, 0), (*this)(1, 1), (*this)(1, 2));
}

Vector3 Matrix4::relativeZ() const
{
    return Vector3((*this)(2, 0), (*this)(2, 1), (*this)(2, 2));
}

Vector3 Matrix4::translation() const
{
    return Vector3((*this)(3, 0)/(*this)(3, 3), (*this)(3, 1)/(*this)(3, 3), (*this)(3, 2)/(*this)(3, 3));
}

Vector3 Matrix4::scaling() const
{
    Vector3 s;
    s.coordinate.x = sqrtf((*this)(0, 0)*(*this)(0, 0) + (*this)(0, 1)*(*this)(0, 1) + (*this)(0, 2)*(*this)(0, 2));
    s.coordinate.y = sqrtf((*this)(1, 0)*(*this)(1, 0) + (*this)(1, 1)*(*this)(1, 1) + (*this)(1, 2)*(*this)(1, 2));
    s.coordinate.z = sqrtf((*this)(2, 0)*(*this)(2, 0) + (*this)(2, 1)*(*this)(2, 1) + (*this)(2, 2)*(*this)(2, 2));
    return s;
}

// TODO: scaling 0.0
void Matrix4::decompose(Vector3& translation, Vector3& scaling, Vector3& euler)
{
    translation = this->translation();
    scaling = this->scaling();
    float m20 = (*this)(2, 0)/scaling.coordinate.z;
    if(m20 >= 0.999)
    {
        euler.coordinate.x = 0.0f;
        euler.coordinate.y = math_pi*0.5f;
        euler.coordinate.z = atan2((*this)(0, 1), (*this)(1, 1));
    }
    else if(m20 <= -0.999)
    {
        euler.coordinate.x = 0.0f;
        euler.coordinate.y = -math_pi*0.5f;
        euler.coordinate.z = atan2((*this)(0, 1), (*this)(1, 1));
    }
    else
    {
        euler.coordinate.x = atan2(-(*this)(2, 1), (*this)(2, 2));
        euler.coordinate.y = asin(m20);
        euler.coordinate.z = atan2(-(*this)(1, 0)/scaling.coordinate.y, (*this)(0, 0)/scaling.coordinate.x);
    }
}

/********************/
/* Common SIMD code */
/********************/
#if defined(HAS_SSE) && defined(HAS_ARM_NEON)
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
    vec4f = matrix[0] * vec.coordinate.x;
    vec4f += matrix[1] * vec.coordinate.y;
    vec4f += matrix[2] * vec.coordinate.z;
    vec4f += matrix[3];
    vec4f /= SHUFFLE_3(vec4f);

    res.coordinate.x = vec4f.m128_f32[0];
    res.coordinate.y = vec4f.m128_f32[1];
    res.coordinate.z = vec4f.m128_f32[2];

    return res;
}

Vector4 Matrix4::operator*(const Vector4& vec) const
{
    Vector4 res;

    simd128 vec4f;
    vec4f  = matrix[0] * vec.coordinate.x;
    vec4f += matrix[1] * vec.coordinate.y;
    vec4f += matrix[2] * vec.coordinate.z;
    vec4f += matrix[3] * vec.coordinate.w;

    res.coordinate.x = vec4f.m128_f32[0];
    res.coordinate.y = vec4f.m128_f32[1];
    res.coordinate.z = vec4f.m128_f32[2];
    res.coordinate.w = vec4f.m128_f32[3];

    return res;
}

void Matrix4::rotateX(float pitch)
{
    float   c = cosf(pitch),
            s = sinf(pitch);

    simd128 tmp = matrix[1];

    matrix[1] = tmp*c    + matrix[2]*s;
    matrix[2] = tmp*(-s) + matrix[2]*c;
}

void Matrix4::rotateY(float yaw)
{
    float   c = cosf(yaw),
            s = sinf(yaw);

    simd128 tmp = matrix[0];

    matrix[0] = tmp*c + matrix[2]*(-s);
    matrix[2] = tmp*s + matrix[2]*c;
}

void Matrix4::rotateZ(float roll)
{
    float   c = cosf(roll),
            s = sinf(roll);

    simd128 tmp = matrix[0];

    matrix[0] = tmp*c    + matrix[1]*s;
    matrix[1] = tmp*(-s) + matrix[1]*c;
}

void Matrix4::rotate(float angle, const Vector3& axis)
{
    float   x = axis.coordinate.x, y = axis.coordinate.y, z = axis.coordinate.z,
            c = cosf(angle),
            s = sinf(angle),
            xc = (1 - c) * x,
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
    float   cx = cosf(euler.coordinate.x),
            sx = sinf(euler.coordinate.x),
            cy = cosf(euler.coordinate.y),
            sy = sinf(euler.coordinate.y),
            cz = cosf(euler.coordinate.z),
            sz = sinf(euler.coordinate.z),
            cx_sy = cx*sy,
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
    matrix[0] *= s.coordinate.x;
    matrix[1] *= s.coordinate.y;
    matrix[2] *= s.coordinate.z;
}

void Matrix4::scale(float s)
{
    matrix[0] *= s;
    matrix[1] *= s;
    matrix[2] *= s;
}

Vector3 Matrix4::transform_rotate(const Vector3& vec) const
{
    Vector3 res;

    simd128 vec4f;
    vec4f = matrix[0] * vec.coordinate.x;
    vec4f += matrix[1] * vec.coordinate.y;
    vec4f += matrix[2] * vec.coordinate.z;
    vec4f /= SHUFFLE_3(vec4f);

    res.coordinate.x = vec4f.m128_f32[0];
    res.coordinate.y = vec4f.m128_f32[1];
    res.coordinate.z = vec4f.m128_f32[2];

    return res;
}

void Matrix4::translate(const Vector3& vec)
{
    matrix[3] = matrix[0] * vec.coordinate.x + matrix[1] * vec.coordinate.y + matrix[2] * vec.coordinate.z + matrix[3];
}

void Matrix4::translate(const Vector2& vec)
{
    matrix[3] = (matrix[0] * vec.coordinate.x + matrix[1] * vec.coordinate.y + matrix[3]);
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
    assert(&mat2 != this);
    r.matrix[0] = matrix[0] * SHUFFLE_0(mat2.matrix[0]);
    i_mad(r.matrix[0], matrix[1], SHUFFLE_1(mat2.matrix[0]);
    i_mad(r.matrix[0], matrix[2], SHUFFLE_2(mat2.matrix[0]);
    i_mad(r.matrix[0], matrix[3], SHUFFLE_3(mat2.matrix[0]);

    r.matrix[1] = matrix[0] * SHUFFLE_0(mat2.matrix[1]);
    i_mad(r.matrix[1], matrix[1], SHUFFLE_1(mat2.matrix[1]);
    i_mad(r.matrix[1], matrix[2], SHUFFLE_2(mat2.matrix[1]);
    i_mad(r.matrix[1], matrix[3], SHUFFLE_3(mat2.matrix[1]);

    r.matrix[2] = matrix[0] * SHUFFLE_0(mat2.matrix[2]);
    i_mad(r.matrix[2], matrix[1], SHUFFLE_1(mat2.matrix[2]);
    i_mad(r.matrix[2], matrix[2], SHUFFLE_2(mat2.matrix[2]);
    i_mad(r.matrix[2], matrix[3], SHUFFLE_3(mat2.matrix[2]);

    r.matrix[3] = matrix[0] * SHUFFLE_0(mat2.matrix[3]);
    i_mad(r.matrix[3], matrix[1], SHUFFLE_1(mat2.matrix[3]);
    i_mad(r.matrix[3], matrix[2], SHUFFLE_2(mat2.matrix[3]);
    i_mad(r.matrix[3], matrix[3], SHUFFLE_3(mat2.matrix[3]);

    return r;
}
/************/
/* Non-SIMD */
/************/
#else
Matrix4& Matrix4::operator+=(const Matrix4& mat)
{
    matrix[0][0] += mat.matrix[0][0];
    matrix[0][1] += mat.matrix[0][1];
    matrix[0][2] += mat.matrix[0][2];
    matrix[0][3] += mat.matrix[0][3];
    matrix[1][0] += mat.matrix[1][0];
    matrix[1][1] += mat.matrix[1][1];
    matrix[1][2] += mat.matrix[1][2];
    matrix[1][3] += mat.matrix[1][3];
    matrix[2][0] += mat.matrix[2][0];
    matrix[2][1] += mat.matrix[2][1];
    matrix[2][2] += mat.matrix[2][2];
    matrix[2][3] += mat.matrix[2][3];
    matrix[3][0] += mat.matrix[3][0];
    matrix[3][1] += mat.matrix[3][1];
    matrix[3][2] += mat.matrix[3][2];
    matrix[3][3] += mat.matrix[3][3];
    return *this;
}

Matrix4& Matrix4::operator-=(const Matrix4& mat)
{
    matrix[0][0] -= mat.matrix[0][0];
    matrix[0][1] -= mat.matrix[0][1];
    matrix[0][2] -= mat.matrix[0][2];
    matrix[0][3] -= mat.matrix[0][3];
    matrix[1][0] -= mat.matrix[1][0];
    matrix[1][1] -= mat.matrix[1][1];
    matrix[1][2] -= mat.matrix[1][2];
    matrix[1][3] -= mat.matrix[1][3];
    matrix[2][0] -= mat.matrix[2][0];
    matrix[2][1] -= mat.matrix[2][1];
    matrix[2][2] -= mat.matrix[2][2];
    matrix[2][3] -= mat.matrix[2][3];
    matrix[3][0] -= mat.matrix[3][0];
    matrix[3][1] -= mat.matrix[3][1];
    matrix[3][2] -= mat.matrix[3][2];
    matrix[3][3] -= mat.matrix[3][3];
    return *this;
}

Matrix4& Matrix4::operator/=(float f)
{
    matrix[0][0] /= f;
    matrix[0][1] /= f;
    matrix[0][2] /= f;
    matrix[0][3] /= f;
    matrix[1][0] /= f;
    matrix[1][1] /= f;
    matrix[1][2] /= f;
    matrix[1][3] /= f;
    matrix[2][0] /= f;
    matrix[2][1] /= f;
    matrix[2][2] /= f;
    matrix[2][3] /= f;
    matrix[3][0] /= f;
    matrix[3][1] /= f;
    matrix[3][2] /= f;
    matrix[3][3] /= f;
    return *this;
}

Matrix4 Matrix4::operator-(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] - mat.matrix[0][0];
    r.matrix[0][1] = matrix[0][1] - mat.matrix[0][1];
    r.matrix[0][2] = matrix[0][2] - mat.matrix[0][2];
    r.matrix[0][3] = matrix[0][3] - mat.matrix[0][3];
    r.matrix[1][0] = matrix[1][0] - mat.matrix[1][0];
    r.matrix[1][1] = matrix[1][1] - mat.matrix[1][1];
    r.matrix[1][2] = matrix[1][2] - mat.matrix[1][2];
    r.matrix[1][3] = matrix[1][3] - mat.matrix[1][3];
    r.matrix[2][0] = matrix[2][0] - mat.matrix[2][0];
    r.matrix[2][1] = matrix[2][1] - mat.matrix[2][1];
    r.matrix[2][2] = matrix[2][2] - mat.matrix[2][2];
    r.matrix[2][3] = matrix[2][3] - mat.matrix[2][3];
    r.matrix[3][0] = matrix[3][0] - mat.matrix[3][0];
    r.matrix[3][1] = matrix[3][1] - mat.matrix[3][1];
    r.matrix[3][2] = matrix[3][2] - mat.matrix[3][2];
    r.matrix[3][3] = matrix[3][3] - mat.matrix[3][3];
    return r;
}

Matrix4 Matrix4::operator+(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] + mat.matrix[0][0];
    r.matrix[0][1] = matrix[0][1] + mat.matrix[0][1];
    r.matrix[0][2] = matrix[0][2] + mat.matrix[0][2];
    r.matrix[0][3] = matrix[0][3] + mat.matrix[0][3];
    r.matrix[1][0] = matrix[1][0] + mat.matrix[1][0];
    r.matrix[1][1] = matrix[1][1] + mat.matrix[1][1];
    r.matrix[1][2] = matrix[1][2] + mat.matrix[1][2];
    r.matrix[1][3] = matrix[1][3] + mat.matrix[1][3];
    r.matrix[2][0] = matrix[2][0] + mat.matrix[2][0];
    r.matrix[2][1] = matrix[2][1] + mat.matrix[2][1];
    r.matrix[2][2] = matrix[2][2] + mat.matrix[2][2];
    r.matrix[2][3] = matrix[2][3] + mat.matrix[2][3];
    r.matrix[3][0] = matrix[3][0] + mat.matrix[3][0];
    r.matrix[3][1] = matrix[3][1] + mat.matrix[3][1];
    r.matrix[3][2] = matrix[3][2] + mat.matrix[3][2];
    r.matrix[3][3] = matrix[3][3] + mat.matrix[3][3];
    return r;
}

Matrix4 Matrix4::operator/(float f) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] / f;
    r.matrix[0][1] = matrix[0][1] / f;
    r.matrix[0][2] = matrix[0][2] / f;
    r.matrix[0][3] = matrix[0][3] / f;
    r.matrix[1][0] = matrix[1][0] / f;
    r.matrix[1][1] = matrix[1][1] / f;
    r.matrix[1][2] = matrix[1][2] / f;
    r.matrix[1][3] = matrix[1][3] / f;
    r.matrix[2][0] = matrix[2][0] / f;
    r.matrix[2][1] = matrix[2][1] / f;
    r.matrix[2][2] = matrix[2][2] / f;
    r.matrix[2][3] = matrix[2][3] / f;
    r.matrix[3][0] = matrix[3][0] / f;
    r.matrix[3][1] = matrix[3][1] / f;
    r.matrix[3][2] = matrix[3][2] / f;
    r.matrix[3][3] = matrix[3][3] / f;
    return r;
}

Vector3 Matrix4::operator*(const Vector3& vec) const
{
    Vector4 res;
    res.coordinate.x  = matrix[0][0] * vec.coordinate.x;
    res.coordinate.y  = matrix[0][1] * vec.coordinate.x;
    res.coordinate.z  = matrix[0][2] * vec.coordinate.x;
    res.coordinate.w  = matrix[0][3] * vec.coordinate.x;
    res.coordinate.x += matrix[1][0] * vec.coordinate.y;
    res.coordinate.y += matrix[1][1] * vec.coordinate.y;
    res.coordinate.z += matrix[1][2] * vec.coordinate.y;
    res.coordinate.w += matrix[1][3] * vec.coordinate.y;
    res.coordinate.x += matrix[2][0] * vec.coordinate.z;
    res.coordinate.y += matrix[2][1] * vec.coordinate.z;
    res.coordinate.z += matrix[2][2] * vec.coordinate.z;
    res.coordinate.w += matrix[2][3] * vec.coordinate.z;
    res.coordinate.x += matrix[3][0];
    res.coordinate.y += matrix[3][1];
    res.coordinate.z += matrix[3][2];
    res.coordinate.w += matrix[3][3];   
    res.coordinate.x /= res.coordinate.w;
    res.coordinate.y /= res.coordinate.w;
    res.coordinate.z /= res.coordinate.w;

    return Vector3(res.x(), res.y(), res.z());
}

Vector4 Matrix4::operator*(const Vector4& vec) const
{
    Vector4 res;
    res.coordinate.x  = matrix[0][0] * vec.coordinate.x;
    res.coordinate.y  = matrix[0][1] * vec.coordinate.x;
    res.coordinate.z  = matrix[0][2] * vec.coordinate.x;
    res.coordinate.w  = matrix[0][3] * vec.coordinate.x;
    res.coordinate.x += matrix[1][0] * vec.coordinate.y;
    res.coordinate.y += matrix[1][1] * vec.coordinate.y;
    res.coordinate.z += matrix[1][2] * vec.coordinate.y;
    res.coordinate.w += matrix[1][3] * vec.coordinate.y;
    res.coordinate.x += matrix[2][0] * vec.coordinate.z;
    res.coordinate.y += matrix[2][1] * vec.coordinate.z;
    res.coordinate.z += matrix[2][2] * vec.coordinate.z;
    res.coordinate.w += matrix[2][3] * vec.coordinate.z;
    res.coordinate.x += matrix[3][0] * vec.coordinate.w;
    res.coordinate.y += matrix[3][1] * vec.coordinate.w;
    res.coordinate.z += matrix[3][2] * vec.coordinate.w;
    res.coordinate.w += matrix[3][3] * vec.coordinate.w;    

    return res;
}

void Matrix4::rotateX(float pitch)
{
    float   c = cosf(pitch),
            s = sinf(pitch),
            tmp0 = matrix[1][0],
            tmp1 = matrix[1][1],
            tmp2 = matrix[1][2],
            tmp3 = matrix[1][3];

    matrix[1][0] = tmp0*c    + matrix[2][0]*s;
    matrix[1][1] = tmp1*c    + matrix[2][1]*s;
    matrix[1][2] = tmp2*c    + matrix[2][2]*s;
    matrix[1][3] = tmp3*c    + matrix[2][3]*s;
    matrix[2][0] = tmp0*(-s) + matrix[2][0]*c;
    matrix[2][1] = tmp1*(-s) + matrix[2][1]*c;
    matrix[2][2] = tmp2*(-s) + matrix[2][2]*c;
    matrix[2][3] = tmp3*(-s) + matrix[2][3]*c;
}

void Matrix4::rotateY(float yaw)
{
    float   c = cosf(yaw),
            s = sinf(yaw),
            tmp0 = matrix[0][0],
            tmp1 = matrix[0][1],
            tmp2 = matrix[0][2],
            tmp3 = matrix[0][3];

    matrix[0][0] = tmp0*c + matrix[2][0]*(-s);
    matrix[0][1] = tmp1*c + matrix[2][1]*(-s);
    matrix[0][2] = tmp2*c + matrix[2][2]*(-s);
    matrix[0][3] = tmp3*c + matrix[2][3]*(-s);
    matrix[2][0] = tmp0*s + matrix[2][0]*c;
    matrix[2][1] = tmp1*s + matrix[2][1]*c;
    matrix[2][2] = tmp2*s + matrix[2][2]*c;
    matrix[2][3] = tmp3*s + matrix[2][3]*c;
}

void Matrix4::rotateZ(float roll)
{
    float   c = cosf(roll),
            s = sinf(roll),
            tmp0 = matrix[0][0],
            tmp1 = matrix[0][1],
            tmp2 = matrix[0][2],
            tmp3 = matrix[0][3];

    matrix[0][0] = tmp0*c    + matrix[1][0]*s;
    matrix[0][1] = tmp1*c    + matrix[1][1]*s;
    matrix[0][2] = tmp2*c    + matrix[1][2]*s;
    matrix[0][3] = tmp3*c    + matrix[1][3]*s;
    matrix[1][0] = tmp0*(-s) + matrix[1][0]*c;
    matrix[1][1] = tmp1*(-s) + matrix[1][1]*c;
    matrix[1][2] = tmp2*(-s) + matrix[1][2]*c;
    matrix[1][3] = tmp3*(-s) + matrix[1][3]*c;
}

void Matrix4::rotate(float angle, const Vector3& axis)
{
    float   x = axis.coordinate.x, y = axis.coordinate.y, z = axis.coordinate.z,
            c = cosf(angle),
            s = sinf(angle),
            xc = (1 - c) * x,
            yc = (1 - c) * y;

    float m00 = xc*x + c,   m01 = xc*y - z*s,   m02 = xc*z + y*s,
          m10 = xc*y + z*s, m11 = yc*y + c,     m12 = yc*z -x*s,
          m20 = xc*z - y*s, m21 = yc*z + x*s,   m22 = z*z*(1-c) + c;
          
    float tmp0 = matrix[0][0], tmp1 = matrix[1][0], tmp2 = matrix[2][0];
    matrix[0][0] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][0] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][0] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][1], tmp1 = matrix[1][1], tmp2 = matrix[2][1];
    matrix[0][1] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][1] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][1] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][2], tmp1 = matrix[1][2], tmp2 = matrix[2][2];
    matrix[0][2] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][2] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][2] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][3], tmp1 = matrix[1][3], tmp2 = matrix[2][3];
    matrix[0][3] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][3] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][3] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
}

void Matrix4::rotate(const Vector3& euler)
{
    float   cx = cosf(euler.coordinate.x),
            sx = sinf(euler.coordinate.x),
            cy = cosf(euler.coordinate.y),
            sy = sinf(euler.coordinate.y),
            cz = cosf(euler.coordinate.z),
            sz = sinf(euler.coordinate.z),
            cx_sy = cx*sy,
            sx_sy = sx*sy;

    float   m00 =  cy*cz,            m01 = -cy*sz,            m02 =  sy,
            m10 =  sx_sy*cz + cx*sz, m11 = -sx_sy*sz + cx*cz, m12 = -sx*cy,
            m20 = -cx_sy*cz + sx*sz, m21 =  cx_sy*sz + sx*cz, m22 =  cx*cy;
    
    float tmp0 = matrix[0][0], tmp1 = matrix[1][0], tmp2 = matrix[2][0];
    matrix[0][0] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][0] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][0] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][1], tmp1 = matrix[1][1], tmp2 = matrix[2][1];
    matrix[0][1] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][1] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][1] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][2], tmp1 = matrix[1][2], tmp2 = matrix[2][2];
    matrix[0][2] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][2] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][2] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][3], tmp1 = matrix[1][3], tmp2 = matrix[2][3];
    matrix[0][3] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][3] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][3] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
}

void Matrix4::scale(const Vector3& s)
{
    matrix[0][0] *= s.coordinate.x;
    matrix[0][1] *= s.coordinate.x;
    matrix[0][2] *= s.coordinate.x;
    matrix[0][3] *= s.coordinate.x;
    matrix[1][0] *= s.coordinate.y;
    matrix[1][1] *= s.coordinate.y;
    matrix[1][2] *= s.coordinate.y;
    matrix[1][3] *= s.coordinate.y;
    matrix[2][0] *= s.coordinate.z;
    matrix[2][1] *= s.coordinate.z;
    matrix[2][2] *= s.coordinate.z;
    matrix[2][3] *= s.coordinate.z;
}

void Matrix4::scale(float s)
{
    matrix[0][0] *= s;
    matrix[0][1] *= s;
    matrix[0][2] *= s;
    matrix[0][3] *= s;
    matrix[1][0] *= s;
    matrix[1][1] *= s;
    matrix[1][2] *= s;
    matrix[1][3] *= s;
    matrix[2][0] *= s;
    matrix[2][1] *= s;
    matrix[2][2] *= s;
    matrix[2][3] *= s;
}

Vector3 Matrix4::transform_rotate(const Vector3& vec) const
{
    Vector4 vec4f;
    vec4f.coordinate.x  = matrix[0][0] * vec.coordinate.x;
    vec4f.coordinate.y  = matrix[0][1] * vec.coordinate.x;
    vec4f.coordinate.z  = matrix[0][2] * vec.coordinate.x;
    vec4f.coordinate.w  = matrix[0][3] * vec.coordinate.x;
    vec4f.coordinate.x += matrix[1][0] * vec.coordinate.y;
    vec4f.coordinate.y += matrix[1][1] * vec.coordinate.y;
    vec4f.coordinate.z += matrix[1][2] * vec.coordinate.y;
    vec4f.coordinate.w += matrix[1][3] * vec.coordinate.y;
    vec4f.coordinate.x += matrix[2][0] * vec.coordinate.z;
    vec4f.coordinate.y += matrix[2][1] * vec.coordinate.z;
    vec4f.coordinate.z += matrix[2][2] * vec.coordinate.z;
    vec4f.coordinate.w += matrix[2][3] * vec.coordinate.z;
    vec4f.coordinate.x /= vec4f.coordinate.w;
    vec4f.coordinate.y /= vec4f.coordinate.w;
    vec4f.coordinate.z /= vec4f.coordinate.w;

    return Vector3(vec4f.x(), vec4f.y(), vec4f.z());
}

void Matrix4::translate(const Vector3& vec)
{
    matrix[3][0] = matrix[0][0] * vec.coordinate.x + matrix[1][0] * vec.coordinate.y + matrix[2][0] * vec.coordinate.z + matrix[3][0];
    matrix[3][1] = matrix[0][1] * vec.coordinate.x + matrix[1][1] * vec.coordinate.y + matrix[2][1] * vec.coordinate.z + matrix[3][1];
    matrix[3][2] = matrix[0][2] * vec.coordinate.x + matrix[1][2] * vec.coordinate.y + matrix[2][2] * vec.coordinate.z + matrix[3][2];
    matrix[3][3] = matrix[0][3] * vec.coordinate.x + matrix[1][3] * vec.coordinate.y + matrix[2][3] * vec.coordinate.z + matrix[3][3];
}

void Matrix4::translate(const Vector2& vec)
{
    matrix[3][0] = matrix[0][0] * vec.coordinate.x + matrix[1][0] * vec.coordinate.y + matrix[3][0];
    matrix[3][1] = matrix[0][1] * vec.coordinate.x + matrix[1][1] * vec.coordinate.y + matrix[3][1];
    matrix[3][2] = matrix[0][2] * vec.coordinate.x + matrix[1][2] * vec.coordinate.y + matrix[3][2];
    matrix[3][3] = matrix[0][3] * vec.coordinate.x + matrix[1][3] * vec.coordinate.y + matrix[3][3];
}

void Matrix4::translateX(float x)
{
    matrix[3][0] = matrix[0][0] * x + matrix[3][0];
    matrix[3][1] = matrix[0][1] * x + matrix[3][1];
    matrix[3][2] = matrix[0][2] * x + matrix[3][2];
    matrix[3][3] = matrix[0][3] * x + matrix[3][3];
}

void Matrix4::translateY(float y)
{
    matrix[3][0] = matrix[1][0] * y + matrix[3][0];
    matrix[3][1] = matrix[1][1] * y + matrix[3][1];
    matrix[3][2] = matrix[1][2] * y + matrix[3][2];
    matrix[3][3] = matrix[1][3] * y + matrix[3][3];
}

void Matrix4::translateZ(float z)
{
    matrix[3][0] = matrix[2][0] * z + matrix[3][0];
    matrix[3][1] = matrix[2][1] * z + matrix[3][1];
    matrix[3][2] = matrix[2][2] * z + matrix[3][2];
    matrix[3][3] = matrix[2][3] * z + matrix[3][3];
}

Matrix4 Matrix4::operator*(const Matrix4& mat2) const 
{
    Matrix4 r;
    TGE_ASSERT(&mat2 != this, "Multiplication of matrix by itself is not supported");
    r.matrix[0][0]  = matrix[0][0] * mat2.matrix[0][0];
    r.matrix[0][1]  = matrix[0][1] * mat2.matrix[0][0];
    r.matrix[0][2]  = matrix[0][2] * mat2.matrix[0][0];
    r.matrix[0][3]  = matrix[0][3] * mat2.matrix[0][0];
    r.matrix[0][0] += matrix[1][0] * mat2.matrix[0][1];
    r.matrix[0][1] += matrix[1][1] * mat2.matrix[0][1];
    r.matrix[0][2] += matrix[1][2] * mat2.matrix[0][1];
    r.matrix[0][3] += matrix[1][3] * mat2.matrix[0][1];
    r.matrix[0][0] += matrix[2][0] * mat2.matrix[0][2];
    r.matrix[0][1] += matrix[2][1] * mat2.matrix[0][2];
    r.matrix[0][2] += matrix[2][2] * mat2.matrix[0][2];
    r.matrix[0][3] += matrix[2][3] * mat2.matrix[0][2];
    r.matrix[0][0] += matrix[3][0] * mat2.matrix[0][3];
    r.matrix[0][1] += matrix[3][1] * mat2.matrix[0][3];
    r.matrix[0][2] += matrix[3][2] * mat2.matrix[0][3];
    r.matrix[0][3] += matrix[3][3] * mat2.matrix[0][3];

    r.matrix[1][0]  = matrix[0][0] * mat2.matrix[1][0];
    r.matrix[1][1]  = matrix[0][1] * mat2.matrix[1][0];
    r.matrix[1][2]  = matrix[0][2] * mat2.matrix[1][0];
    r.matrix[1][3]  = matrix[0][3] * mat2.matrix[1][0];
    r.matrix[1][0] += matrix[1][0] * mat2.matrix[1][1];
    r.matrix[1][1] += matrix[1][1] * mat2.matrix[1][1];
    r.matrix[1][2] += matrix[1][2] * mat2.matrix[1][1];
    r.matrix[1][3] += matrix[1][3] * mat2.matrix[1][1];
    r.matrix[1][0] += matrix[2][0] * mat2.matrix[1][2];
    r.matrix[1][1] += matrix[2][1] * mat2.matrix[1][2];
    r.matrix[1][2] += matrix[2][2] * mat2.matrix[1][2];
    r.matrix[1][3] += matrix[2][3] * mat2.matrix[1][2];
    r.matrix[1][0] += matrix[3][0] * mat2.matrix[1][3];
    r.matrix[1][1] += matrix[3][1] * mat2.matrix[1][3];
    r.matrix[1][2] += matrix[3][2] * mat2.matrix[1][3];
    r.matrix[1][3] += matrix[3][3] * mat2.matrix[1][3];
    
    r.matrix[2][0]  = matrix[0][0] * mat2.matrix[2][0];
    r.matrix[2][1]  = matrix[0][1] * mat2.matrix[2][0];
    r.matrix[2][2]  = matrix[0][2] * mat2.matrix[2][0];
    r.matrix[2][3]  = matrix[0][3] * mat2.matrix[2][0];
    r.matrix[2][0] += matrix[1][0] * mat2.matrix[2][1];
    r.matrix[2][1] += matrix[1][1] * mat2.matrix[2][1];
    r.matrix[2][2] += matrix[1][2] * mat2.matrix[2][1];
    r.matrix[2][3] += matrix[1][3] * mat2.matrix[2][1];
    r.matrix[2][0] += matrix[2][0] * mat2.matrix[2][2];
    r.matrix[2][1] += matrix[2][1] * mat2.matrix[2][2];
    r.matrix[2][2] += matrix[2][2] * mat2.matrix[2][2];
    r.matrix[2][3] += matrix[2][3] * mat2.matrix[2][2];
    r.matrix[2][0] += matrix[3][0] * mat2.matrix[2][3];
    r.matrix[2][1] += matrix[3][1] * mat2.matrix[2][3];
    r.matrix[2][2] += matrix[3][2] * mat2.matrix[2][3];
    r.matrix[2][3] += matrix[3][3] * mat2.matrix[2][3];
    
    r.matrix[3][0]  = matrix[0][0] * mat2.matrix[3][0];
    r.matrix[3][1]  = matrix[0][1] * mat2.matrix[3][0];
    r.matrix[3][2]  = matrix[0][2] * mat2.matrix[3][0];
    r.matrix[3][3]  = matrix[0][3] * mat2.matrix[3][0];
    r.matrix[3][0] += matrix[1][0] * mat2.matrix[3][1];
    r.matrix[3][1] += matrix[1][1] * mat2.matrix[3][1];
    r.matrix[3][2] += matrix[1][2] * mat2.matrix[3][1];
    r.matrix[3][3] += matrix[1][3] * mat2.matrix[3][1];
    r.matrix[3][0] += matrix[2][0] * mat2.matrix[3][2];
    r.matrix[3][1] += matrix[2][1] * mat2.matrix[3][2];
    r.matrix[3][2] += matrix[2][2] * mat2.matrix[3][2];
    r.matrix[3][3] += matrix[2][3] * mat2.matrix[3][2];
    r.matrix[3][0] += matrix[3][0] * mat2.matrix[3][3];
    r.matrix[3][1] += matrix[3][1] * mat2.matrix[3][3];
    r.matrix[3][2] += matrix[3][2] * mat2.matrix[3][3];
    r.matrix[3][3] += matrix[3][3] * mat2.matrix[3][3];
    
    return r;
}
#endif

/*******/
/* SSE */
/*******/
#ifdef HAS_SSE
Matrix4::Matrix4(float* _mat)
{
    matrix[0] = _mm_loadu_ps(_mat);
    matrix[1] = _mm_loadu_ps(_mat + 4);
    matrix[2] = _mm_loadu_ps(_mat + 8);
    matrix[3] = _mm_loadu_ps(_mat + 12);
}

Matrix4::Matrix4(float m00, float m01, float m02, float m03,
                 float m10, float m11, float m12, float m13,
                 float m20, float m21, float m22, float m23,
                 float m30, float m31, float m32, float m33)
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
#elif defined(HAS_ARM_NEON)
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
/************/
/* Non-SIMD */
/************/
#else
Matrix4::Matrix4(float* _mat)
{
    memcpy(matrix, _mat, sizeof(matrix));
}

Matrix4::Matrix4(float m00, float m01, float m02, float m03,
                 float m10, float m11, float m12, float m13,
                 float m20, float m21, float m22, float m23,
                 float m30, float m31, float m32, float m33)
{
    matrix[0][0] = m00;
    matrix[0][1] = m01;
    matrix[0][2] = m02;
    matrix[0][3] = m03;
    matrix[1][0] = m10;
    matrix[1][1] = m11;
    matrix[1][2] = m12;
    matrix[1][3] = m13;
    matrix[2][0] = m20;
    matrix[2][1] = m21;
    matrix[2][2] = m22;
    matrix[2][3] = m23;
    matrix[3][0] = m30;
    matrix[3][1] = m31;
    matrix[3][2] = m32;
    matrix[3][3] = m33;
}

void Matrix4::transposeSelf()
{
    matrix[0][1] = matrix[1][0], matrix[0][2] = matrix[2][0], matrix[0][3] = matrix[3][0];
    matrix[1][0] = matrix[0][1], matrix[1][2] = matrix[2][1], matrix[1][3] = matrix[3][1];
    matrix[2][0] = matrix[0][2], matrix[2][1] = matrix[1][2], matrix[2][3] = matrix[3][2];
    matrix[3][0] = matrix[0][3], matrix[3][1] = matrix[1][3], matrix[3][2] = matrix[2][3];
}

void Matrix4::identity()
{
    matrix[0][0] = 1.0f, matrix[0][1] = 0.0f, matrix[0][2] = 0.0f, matrix[0][3] = 0.0f;
    matrix[1][0] = 0.0f, matrix[1][1] = 1.0f, matrix[1][2] = 0.0f, matrix[1][3] = 0.0f;
    matrix[2][0] = 0.0f, matrix[2][1] = 0.0f, matrix[2][2] = 1.0f, matrix[2][3] = 0.0f;
    matrix[3][0] = 0.0f, matrix[3][1] = 0.0f, matrix[3][2] = 0.0f, matrix[3][3] = 1.0f;
}

Matrix4 Matrix4::inverse() const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[1][1]*matrix[2][2]*matrix[3][3] + matrix[2][1]*matrix[3][2]*matrix[1][3] + matrix[3][1]*matrix[1][2]*matrix[2][3] - matrix[1][1]*matrix[3][2]*matrix[2][3] - matrix[2][1]*matrix[1][2]*matrix[3][3] - matrix[3][1]*matrix[2][2]*matrix[1][3];
    r.matrix[1][0] = matrix[1][0]*matrix[3][2]*matrix[2][3] + matrix[2][0]*matrix[1][2]*matrix[3][3] + matrix[3][0]*matrix[2][2]*matrix[1][3] - matrix[1][0]*matrix[2][2]*matrix[3][3] - matrix[2][0]*matrix[3][2]*matrix[1][3] - matrix[3][0]*matrix[1][2]*matrix[2][3];
    r.matrix[2][0] = matrix[1][0]*matrix[2][1]*matrix[3][3] + matrix[2][0]*matrix[3][1]*matrix[1][3] + matrix[3][0]*matrix[1][1]*matrix[2][3] - matrix[1][0]*matrix[3][1]*matrix[2][3] - matrix[2][0]*matrix[1][1]*matrix[3][3] - matrix[3][0]*matrix[2][1]*matrix[1][3];
    r.matrix[3][0] = matrix[1][0]*matrix[3][1]*matrix[2][2] + matrix[2][0]*matrix[1][1]*matrix[3][2] + matrix[3][0]*matrix[2][1]*matrix[1][2] - matrix[1][0]*matrix[2][1]*matrix[3][2] - matrix[2][0]*matrix[3][1]*matrix[1][2] - matrix[3][0]*matrix[1][1]*matrix[2][2];
    r.matrix[0][1] = matrix[0][1]*matrix[3][2]*matrix[2][3] + matrix[2][1]*matrix[0][2]*matrix[3][3] + matrix[3][1]*matrix[2][2]*matrix[0][3] - matrix[0][1]*matrix[2][2]*matrix[3][3] - matrix[2][1]*matrix[3][2]*matrix[0][3] - matrix[3][1]*matrix[0][2]*matrix[2][3];
    r.matrix[1][1] = matrix[0][0]*matrix[2][2]*matrix[3][3] + matrix[2][0]*matrix[3][2]*matrix[0][3] + matrix[3][0]*matrix[0][2]*matrix[2][3] - matrix[0][0]*matrix[3][2]*matrix[2][3] - matrix[2][0]*matrix[0][2]*matrix[3][3] - matrix[3][0]*matrix[2][2]*matrix[0][3];
    r.matrix[2][1] = matrix[0][0]*matrix[3][1]*matrix[2][3] + matrix[2][0]*matrix[0][1]*matrix[3][3] + matrix[3][0]*matrix[2][1]*matrix[0][3] - matrix[0][0]*matrix[2][1]*matrix[3][3] - matrix[2][0]*matrix[3][1]*matrix[0][3] - matrix[3][0]*matrix[0][1]*matrix[2][3];
    r.matrix[3][1] = matrix[0][0]*matrix[2][1]*matrix[3][2] + matrix[2][0]*matrix[3][1]*matrix[0][2] + matrix[3][0]*matrix[0][1]*matrix[2][2] - matrix[0][0]*matrix[3][1]*matrix[2][2] - matrix[2][0]*matrix[0][1]*matrix[3][2] - matrix[3][0]*matrix[2][1]*matrix[0][2];
    r.matrix[0][2] = matrix[0][1]*matrix[1][2]*matrix[3][3] + matrix[1][1]*matrix[3][2]*matrix[0][3] + matrix[3][1]*matrix[0][2]*matrix[1][3] - matrix[0][1]*matrix[3][2]*matrix[1][3] - matrix[1][1]*matrix[0][2]*matrix[3][3] - matrix[3][1]*matrix[1][2]*matrix[0][3];
    r.matrix[1][2] = matrix[0][0]*matrix[3][2]*matrix[1][3] + matrix[1][0]*matrix[0][2]*matrix[3][3] + matrix[3][0]*matrix[1][2]*matrix[0][3] - matrix[0][0]*matrix[1][2]*matrix[3][3] - matrix[1][0]*matrix[3][2]*matrix[0][3] - matrix[3][0]*matrix[0][2]*matrix[1][3];
    r.matrix[2][2] = matrix[0][0]*matrix[1][1]*matrix[3][3] + matrix[1][0]*matrix[3][1]*matrix[0][3] + matrix[3][0]*matrix[0][1]*matrix[1][3] - matrix[0][0]*matrix[3][1]*matrix[1][3] - matrix[1][0]*matrix[0][1]*matrix[3][3] - matrix[3][0]*matrix[1][1]*matrix[0][3];
    r.matrix[3][2] = matrix[0][0]*matrix[3][1]*matrix[1][2] + matrix[1][0]*matrix[0][1]*matrix[3][2] + matrix[3][0]*matrix[1][1]*matrix[0][2] - matrix[0][0]*matrix[1][1]*matrix[3][2] - matrix[1][0]*matrix[3][1]*matrix[0][2] - matrix[3][0]*matrix[0][1]*matrix[1][2];
    r.matrix[0][3] = matrix[0][1]*matrix[2][2]*matrix[1][3] + matrix[1][1]*matrix[0][2]*matrix[2][3] + matrix[2][1]*matrix[1][2]*matrix[0][3] - matrix[0][1]*matrix[1][2]*matrix[2][3] - matrix[1][1]*matrix[2][2]*matrix[0][3] - matrix[2][1]*matrix[0][2]*matrix[1][3];
    r.matrix[1][3] = matrix[0][0]*matrix[1][2]*matrix[2][3] + matrix[1][0]*matrix[2][2]*matrix[0][3] + matrix[2][0]*matrix[0][2]*matrix[1][3] - matrix[0][0]*matrix[2][2]*matrix[1][3] - matrix[1][0]*matrix[0][2]*matrix[2][3] - matrix[2][0]*matrix[1][2]*matrix[0][3];
    r.matrix[2][3] = matrix[0][0]*matrix[2][1]*matrix[1][3] + matrix[1][0]*matrix[0][1]*matrix[2][3] + matrix[2][0]*matrix[1][1]*matrix[0][3] - matrix[0][0]*matrix[1][1]*matrix[2][3] - matrix[1][0]*matrix[2][1]*matrix[0][3] - matrix[2][0]*matrix[0][1]*matrix[1][3];
    r.matrix[3][3] = matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[1][0]*matrix[2][1]*matrix[0][2] + matrix[2][0]*matrix[0][1]*matrix[1][2] - matrix[0][0]*matrix[2][1]*matrix[1][2] - matrix[1][0]*matrix[0][1]*matrix[2][2] - matrix[2][0]*matrix[1][1]*matrix[0][2];
    float det = matrix[0][0]*r.matrix[0][0] + matrix[1][0]*r.matrix[0][1] + matrix[2][0]*r.matrix[0][2] + matrix[3][0]*r.matrix[0][3];
    r /= det;
    
    return r;
}
#endif
}

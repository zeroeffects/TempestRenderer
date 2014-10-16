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

#include "tempest/math/quaternion.hh"
#include "tempest/math/matrix4.hh"

#include <cmath>

namespace Tempest
{
Quaternion::Quaternion(const Matrix4& matrix)
{
    coordinate.w = sqrt(std::max(0.0f, 1.0f + matrix(0, 0) + matrix(1, 1) + matrix(2, 2)))/2.0f;
    coordinate.x = sqrt(std::max(0.0f, 1.0f + matrix(0, 0) - matrix(1, 1) - matrix(2, 2)))/2.0f;
    coordinate.y = sqrt(std::max(0.0f, 1.0f - matrix(0, 0) + matrix(1, 1) - matrix(2, 2)))/2.0f;
    coordinate.z = sqrt(std::max(0.0f, 1.0f - matrix(0, 0) - matrix(1, 1) + matrix(2, 2)))/2.0f;
    coordinate.x = std::copysign(coordinate.x, matrix(1, 2) - matrix(2, 1));
    coordinate.y = std::copysign(coordinate.y, matrix(2, 0) - matrix(0, 2));
    coordinate.z = std::copysign(coordinate.z, matrix(0, 1) - matrix(1, 0));
}

Quaternion::Quaternion(const Quaternion& quat)
    :   coordinate(quat.coordinate)
{
}

Quaternion& Quaternion::operator=(const Quaternion& quat)
{
    coordinate = quat.coordinate;
    return *this;
}

Quaternion& operator*=(Quaternion& lhs, const Quaternion& rhs)
{
    lhs = lhs * rhs;
    return lhs;
}

Quaternion& operator+=(Quaternion& lhs, const Quaternion& rhs)
{
    lhs.coordinate.x += rhs.coordinate.x;
    lhs.coordinate.y += rhs.coordinate.y;
    lhs.coordinate.z += rhs.coordinate.z;
    lhs.coordinate.w += rhs.coordinate.w;
    return lhs;
}

Quaternion& operator-=(Quaternion& lhs, const Quaternion& rhs)
{
    lhs.coordinate.x -= rhs.coordinate.x;
    lhs.coordinate.y -= rhs.coordinate.y;
    lhs.coordinate.z -= rhs.coordinate.z;
    lhs.coordinate.w -= rhs.coordinate.w;
    return lhs;
}
    
Quaternion& operator/=(Quaternion& lhs, const Quaternion& rhs)
{
    return lhs *= rhs.inverse();
}

Quaternion& operator*=(Quaternion& q, float f)
{
    q.coordinate.x *= f;
    q.coordinate.y *= f;
    q.coordinate.z *= f;
    q.coordinate.w *= f;
    return q;
}

Quaternion& operator/=(Quaternion& q, float f)
{
    q.coordinate.x /= f;
    q.coordinate.y /= f;
    q.coordinate.z /= f;
    q.coordinate.w /= f;
    return q;
}

bool operator==(const Quaternion& lhs, const Quaternion& rhs)
{
    return approx_eq(lhs.coordinate.x, rhs.coordinate.x) &&
           approx_eq(lhs.coordinate.y, rhs.coordinate.y) &&
           approx_eq(lhs.coordinate.z, rhs.coordinate.z) &&
           approx_eq(lhs.coordinate.w, rhs.coordinate.w);
}

Quaternion operator/(const Quaternion& lhs, const Quaternion& rhs)
{
    return lhs * rhs.inverse();
}

Quaternion operator*(const Quaternion& lhs, const Quaternion& rhs)
{
    Quaternion result;
    result.coordinate.w = lhs.coordinate.w*rhs.coordinate.w - lhs.coordinate.x*rhs.coordinate.x - lhs.coordinate.y*rhs.coordinate.y - lhs.coordinate.z*rhs.coordinate.z;
    result.coordinate.x = lhs.coordinate.w*rhs.coordinate.x + lhs.coordinate.x*rhs.coordinate.w + lhs.coordinate.y*rhs.coordinate.z - lhs.coordinate.z*rhs.coordinate.y;
    result.coordinate.y = lhs.coordinate.w*rhs.coordinate.y - lhs.coordinate.x*rhs.coordinate.z + lhs.coordinate.y*rhs.coordinate.w + lhs.coordinate.z*rhs.coordinate.x;
    result.coordinate.z = lhs.coordinate.w*rhs.coordinate.z + lhs.coordinate.x*rhs.coordinate.y - lhs.coordinate.y*rhs.coordinate.x + lhs.coordinate.z*rhs.coordinate.w;
    return result;
}

Quaternion operator+(const Quaternion& lhs, const Quaternion& rhs)
{
    Quaternion result;
    result.coordinate.x = lhs.coordinate.x + rhs.coordinate.x;
    result.coordinate.y = lhs.coordinate.y + rhs.coordinate.y;
    result.coordinate.z = lhs.coordinate.z + rhs.coordinate.z;
    result.coordinate.w = lhs.coordinate.w + rhs.coordinate.w;
    return result;
}

Quaternion operator-(const Quaternion& lhs, const Quaternion& rhs)
{
    Quaternion result;
    result.coordinate.x = lhs.coordinate.x - rhs.coordinate.x;
    result.coordinate.y = lhs.coordinate.y - rhs.coordinate.y;
    result.coordinate.z = lhs.coordinate.z - rhs.coordinate.z;
    result.coordinate.w = lhs.coordinate.w - rhs.coordinate.w;
    return result;
}

Quaternion operator*(const Quaternion& q, float f)
{
    return Quaternion(q.coordinate.x*f, q.coordinate.y*f, q.coordinate.z*f, q.coordinate.w*f);
}

Quaternion operator/(const Quaternion& q, float f)
{
    return Quaternion(q.coordinate.x/f, q.coordinate.y/f, q.coordinate.z/f, q.coordinate.w/f);
}


Quaternion operator*(float f, const Quaternion& q)
{
    return Quaternion(q.coordinate.x*f, q.coordinate.y*f, q.coordinate.z*f, q.coordinate.w*f);
}

void Quaternion::identity()
{
    coordinate.x = coordinate.y = coordinate.z = 0.0f;
    coordinate.w = 1.0f;
}

Quaternion Quaternion::conjugate() const
{
    Quaternion result;
    result.coordinate.x = -coordinate.x;
    result.coordinate.y = -coordinate.y;
    result.coordinate.z = -coordinate.z;
    result.coordinate.w = coordinate.w;
    return result;
}

void Quaternion::conjugateSelf()
{
    coordinate.x = -coordinate.x;
    coordinate.y = -coordinate.y;
    coordinate.z = -coordinate.z;
}

Quaternion Quaternion::inverse() const
{
    return conjugate();
}

void Quaternion::invertSelf()
{
    conjugateSelf();
}

float Quaternion::length() const
{
    return sqrt(coordinate.x*coordinate.x + coordinate.y*coordinate.y + coordinate.z*coordinate.z + coordinate.w*coordinate.w);
}

void Quaternion::normalize()
{
    float l = length();
    if(l == 0)
        return;
    coordinate.x /= l;
    coordinate.y /= l;
    coordinate.z /= l;
    coordinate.w /= l;
}

void Quaternion::rotateX(float pitch)
{
    float   rx = sinf(pitch/2.0f),
            rw = cosf(pitch/2.0f);
    Quaternion tmp(*this);
    coordinate.w =  tmp.coordinate.w*rw - tmp.coordinate.x*rx;
    coordinate.x =  tmp.coordinate.w*rx + tmp.coordinate.x*rw;
    coordinate.y =  tmp.coordinate.y*rw + tmp.coordinate.z*rx;
    coordinate.z = -tmp.coordinate.y*rx + tmp.coordinate.z*rw;
    normalize();
}

void Quaternion::rotateY(float yaw)
{
    float   ry = sinf(yaw/2.0f),
            rw = cosf(yaw/2.0f);
    Quaternion tmp(*this);
    coordinate.w = tmp.coordinate.w*rw - tmp.coordinate.y*ry;
    coordinate.x = tmp.coordinate.x*rw - tmp.coordinate.z*ry;
    coordinate.y = tmp.coordinate.w*ry + tmp.coordinate.y*rw;
    coordinate.z = tmp.coordinate.x*ry + tmp.coordinate.z*rw;
    normalize();
}

void Quaternion::rotateZ(float roll)
{
    float   rz = sinf(roll/2.0f),
            rw = cosf(roll/2.0f);
    Quaternion tmp(*this);
    coordinate.w =  tmp.coordinate.w*rw - tmp.coordinate.z*rz;
    coordinate.x =  tmp.coordinate.x*rw + tmp.coordinate.y*rz;
    coordinate.y = -tmp.coordinate.x*rz + tmp.coordinate.y*rw;
    coordinate.z =  tmp.coordinate.w*rz + tmp.coordinate.z*rw;
    normalize();
}

void Quaternion::rotate(float angle, const Vector3& axis)
{
    Quaternion rotation;
    float sin_a = sinf(angle/2.0f);
    float cos_a = cosf(angle/2.0f);
    rotation.coordinate.x = axis.coordinate.x * sin_a;
    rotation.coordinate.y = axis.coordinate.y * sin_a;
    rotation.coordinate.z = axis.coordinate.z * sin_a;
    rotation.coordinate.w = cos_a;
    rotation.normalize();
    *this *= rotation;
}

float Quaternion::dot(const Quaternion& q) const
{
    return coordinate.x*q.coordinate.x + coordinate.y*q.coordinate.y + coordinate.z*q.coordinate.z + coordinate.w*q.coordinate.w;
}

Quaternion operator-(const Quaternion& q)
{
    return Quaternion(-q.coordinate.x, -q.coordinate.y, -q.coordinate.z, -q.coordinate.w);
}

Vector3 Quaternion::transform(const Vector3& v) const
{
    return v + 2.0f*vector().cross(vector().cross(v) + scalar()*v);
}
}


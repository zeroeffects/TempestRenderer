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

#include "tempest/math/dual-quaternion.hh"
#include "tempest/math/matrix4.hh"

namespace Tempest
{
DualQuaternion::DualQuaternion()
{
    static_assert(sizeof(DualQuaternion) == 2*sizeof(Quaternion), "Quaternion has the wrong size");
}

DualQuaternion::DualQuaternion(const Quaternion& ndp, const Quaternion& dp)
    :   non_dual(ndp),
        dual(dp) {}

DualQuaternion::DualQuaternion(const DualQuaternion& d)
    :   non_dual(d.non_dual),
        dual(d.dual) {}

DualQuaternion::DualQuaternion(const Matrix4& mat)
    :   non_dual(ToQuaternion(mat))
{
    dual = Quaternion{mat(3, 0)/2.0f, mat(3, 1)/2.0f, mat(3, 2)/2.0f, 0.0f} * non_dual;
}

DualQuaternion& DualQuaternion::operator=(const DualQuaternion& d)
{
    non_dual = d.non_dual;
    dual = d.dual;
    return *this;
}

DualQuaternion& operator*=(DualQuaternion& lhs, const DualQuaternion& rhs)
{
    lhs.dual = lhs.dual*rhs.non_dual + lhs.non_dual*rhs.dual;
    lhs.non_dual *= rhs.non_dual;
    return lhs;
}

DualQuaternion& operator+=(DualQuaternion& lhs, const DualQuaternion& rhs)
{
    lhs.non_dual += rhs.non_dual;
    lhs.dual += rhs.dual;
    return lhs;
}

DualQuaternion& operator-=(DualQuaternion& lhs, const DualQuaternion& rhs)
{
    lhs.non_dual -= rhs.non_dual;
    lhs.dual -= rhs.dual;
    return lhs;
}

DualQuaternion& operator*=(DualQuaternion& d, float f)
{
    d.dual *= f;
    d.non_dual *= f;
    return d;
}

bool operator==(const DualQuaternion& lhs, const DualQuaternion& rhs)
{
    return lhs.non_dual == rhs.non_dual && lhs.dual == rhs.dual;
}

DualQuaternion operator*(const DualQuaternion& lhs, const DualQuaternion& rhs)
{
    return DualQuaternion(lhs.non_dual*rhs.non_dual, lhs.dual*rhs.non_dual + lhs.non_dual*rhs.dual);
}

DualQuaternion operator+(const DualQuaternion& lhs, const DualQuaternion& rhs)
{
    return DualQuaternion(lhs.non_dual + rhs.non_dual, lhs.dual + rhs.dual);
}

DualQuaternion operator-(const DualQuaternion& lhs, const DualQuaternion& rhs)
{
    return DualQuaternion(lhs.non_dual - rhs.non_dual, lhs.dual - rhs.dual);
}

DualQuaternion operator-(const DualQuaternion& q)
{
    return DualQuaternion(-q.non_dual, -q.dual);
}

DualQuaternion operator*(const DualQuaternion& d, float f)
{
    return DualQuaternion(d.non_dual*f, d.dual*f);
}

DualQuaternion operator*(float f, const DualQuaternion& d)
{
    return DualQuaternion(d.non_dual*f, d.dual*f);
}

DualQuaternion DualQuaternion::conjugate() const
{
    return DualQuaternion(non_dual, -dual);
}

void DualQuaternion::conjugateSelf()
{
    dual = -dual;
}

DualQuaternion DualQuaternion::conjugateQuaternion() const
{
    return DualQuaternion(Conjugate(non_dual), Conjugate(dual));
}

void DualQuaternion::conjugateQuaternionSelf()
{
    ConjugateSelf(&non_dual);
    ConjugateSelf(&dual);
}

DualQuaternion DualQuaternion::inverse() const
{
    return conjugateQuaternion();
}

void DualQuaternion::invertSelf()
{
    non_dual = -Inverse(non_dual);
    dual *= non_dual*non_dual;
}

float DualQuaternion::length() const
{
    float lnd = Length(non_dual);
    return lnd + Dot(non_dual, dual)/lnd;
}

void DualQuaternion::normalize()
{
    float lnd = Length(non_dual);
    non_dual /= lnd, dual /= lnd;
}

void DualQuaternion::rotateX(float pitch)
{
    Quaternion qr = IdentityQuaternion();
    RotateX(qr, pitch);
    non_dual *= qr;
    dual *= qr;
}

void DualQuaternion::rotateY(float yaw)
{
    Quaternion qr = IdentityQuaternion();
    RotateY(qr, yaw);
    non_dual *= qr;
    dual *= qr;
}

void DualQuaternion::rotateZ(float roll)
{
    Quaternion qr = IdentityQuaternion();
    RotateZ(qr, roll);
    non_dual *= qr;
    dual *= qr;
}

void DualQuaternion::rotate(float angle, const Vector3& axis)
{
    Quaternion qr = IdentityQuaternion();
    Rotate(qr, angle, axis);
    non_dual *= qr;
    dual *= qr;
}

void DualQuaternion::translate(const Vector3& vec)
{
	Quaternion qtr{ vec.x/2.0f, vec.y/2.0f, vec.z/2.0f, 0.0f };
    dual += non_dual*qtr;
}

void DualQuaternion::translateX(float x)
{
	Quaternion qtr{ x/2.0f, 0.0f, 0.0f, 0.0f };
    dual += non_dual*qtr;
}

void DualQuaternion::translateY(float y)
{
	Quaternion qtr{ 0.0f, y/2.0f, 0.0f, 0.0f };
    dual += non_dual*qtr;
}

void DualQuaternion::translateZ(float z)
{
	Quaternion qtr{ 0.0f, 0.0f, z/2.0f, 0.0f };
    dual += non_dual*qtr;
}

Vector3 DualQuaternion::transform(const Vector3& v) const
{
    return Transform(non_dual, v) + 2.0f*(ExtractScalar(non_dual)*ExtractVector(dual) -
                                          ExtractScalar(dual)*ExtractVector(non_dual) +
                                          Cross(ExtractVector(non_dual), ExtractVector(dual)));
}

void DualQuaternion::identity()
{
    non_dual = IdentityQuaternion();
    dual.x = 0.0f;
    dual.y = 0.0f;
    dual.z = 0.0f;
    dual.w = 0.0f;
}

DualQuaternion interpolate(float t, const DualQuaternion& q1, const DualQuaternion& q2)
{
    TGE_ASSERT(0.0f <= t && t <= 1.0f, "Invalid interpolation");
    DualQuaternion qres((1.0f-t)*q1 + t*q2);
    qres.normalize();
    return qres;
}
}
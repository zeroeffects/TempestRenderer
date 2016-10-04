/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 2013 2014 Zdravko Velinov
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

#ifndef _TEMPEST_QUATERNION_HH_
#define _TEMPEST_QUATERNION_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/matrix3.hh"

namespace Tempest
{
//! Quaternion representation of spatial rotation
/*! \ingroup CarinaMath
    Quaternions are used for representing spatial rotation. In the case of dual quaternions
    and some other mathematical structures they could be used to represent other types of
    linear transfromations, e.g. translation.
*/
union Quaternion
{
    struct
    {
    float x, /*!< x-coordinate component */
          y, /*!< y-coordinate component */
		  z, /*!< z-coordinate component */
          w; /*!< w-coordinate component */
    };
    struct
    {
	    Vector3 Vector;
	    float   Scalar;
    };
    Vector4 V4;
    float Components[4];
};

template<class TMatrix>
inline EXPORT_CUDA Quaternion MatrixToQuaternion(const TMatrix& matrix)
{
	Quaternion ret;
	ret.w = sqrtf(Maxf(0.0f, 1.0f + matrix(0, 0) + matrix(1, 1) + matrix(2, 2)))/2.0f;
    ret.x = sqrtf(Maxf(0.0f, 1.0f + matrix(0, 0) - matrix(1, 1) - matrix(2, 2)))/2.0f;
    ret.y = sqrtf(Maxf(0.0f, 1.0f - matrix(0, 0) + matrix(1, 1) - matrix(2, 2)))/2.0f;
    ret.z = sqrtf(Maxf(0.0f, 1.0f - matrix(0, 0) - matrix(1, 1) + matrix(2, 2)))/2.0f;
    ret.x = std::copysign(ret.x, matrix(1, 2) - matrix(2, 1));
    ret.y = std::copysign(ret.y, matrix(2, 0) - matrix(0, 2));
    ret.z = std::copysign(ret.z, matrix(0, 1) - matrix(1, 0));
	return ret;
}

inline EXPORT_CUDA Quaternion ToQuaternion(const Matrix4& matrix)
{
	return MatrixToQuaternion(matrix);
}

inline EXPORT_CUDA Quaternion ToQuaternion(const Matrix3& matrix)
{
	return MatrixToQuaternion(matrix);
}

inline EXPORT_CUDA Quaternion ToQuaternion(const Vector3& euler)
{
    float c1, c2, c3, s1, s2, s3;
    Tempest::FastSinCos(0.5f*euler.x, &s1, &c1);
    Tempest::FastSinCos(0.5f*euler.y, &s2, &c2);
    Tempest::FastSinCos(0.5f*euler.z, &s3, &c3);

    return { s1*c2*c3 + c1*s2*s3,
             - s1*c2*s3 + c1*s2*c3,
             c1*c2*s3 + s1*s2*c3,
             c1*c2*c3 - s1*s2*s3 };
}

/* // Not same multiplication convention
inline EXPORT_CUDA Vector3 ToEulerAngles(const Quaternion& quat)
{
    float pole_test = quat.x*quat.y + quat.z*quat.w;
    if(pole_test > 0.499f)
    {
        return { 2.0f*atan2f(quat.x, quat.w),
                 Tempest::MathPi*0.5f,
                 0.0f };
    }
    else if(pole_test < -0.499f)
    {
        return { -2.0f*atan2f(quat.x, quat.w),
                 Tempest::MathPi*0.5f,
                 0.0f };
    }

    float xx = quat.x*quat.x,
          yy = quat.y*quat.y,
          zz = quat.z*quat.z;

    return { atan2f(2.0f*quat.y*quat.w - 2.0f*quat.x*quat.z, 1.0f - 2.0f*(yy + zz)),
             asinf(2.0f*pole_test),
             atan2f(2.0f*(quat.x*quat.w - quat.y*quat.z), 1.0f - 2.0f*(xx + zz)) };
}
*/

inline EXPORT_CUDA Matrix3 ToMatrix3(const Quaternion& r)
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
    return Matrix3({ 1.0f - 2.0f*(yy + zz),
                            2.0f*(xy + zw),
                            2.0f*(xz - yw) },
                   {        2.0f*(xy - zw),
                     1.0f - 2.0f*(xx + zz),
                            2.0f*(yz + xw) },
                   {        2.0f*(xz + yw),
                            2.0f*(yz - xw),
                     1.0f - 2.0f*(xx + yy) });
}

inline EXPORT_CUDA Vector3 ToTangent(const Quaternion& r)
{
    float xy = r.x*r.y,
          xz = r.x*r.z,
          yy = r.y*r.y,
          yw = r.y*r.w,
          zz = r.z*r.z,
          zw = r.z*r.w;
    return { 1.0f - 2.0f*(yy + zz),
                    2.0f*(xy + zw),
                    2.0f*(xz - yw) };
}

inline EXPORT_CUDA Vector3 ToBinormal(const Quaternion& r)
{
    float   xx = r.x*r.x,
            xy = r.x*r.y,
            xw = r.x*r.w,
            yz = r.y*r.z,
            zz = r.z*r.z,
            zw = r.z*r.w;
    return {        2.0f*(xy - zw),
             1.0f - 2.0f*(xx + zz),
                    2.0f*(yz + xw) };
}

inline EXPORT_CUDA Vector3 ToNormal(const Quaternion& r)
{
    float    xx = r.x*r.x,
             xz = r.x*r.z,
             xw = r.x*r.w,
             yy = r.y*r.y,
             yz = r.y*r.z,
             yw = r.y*r.w;
    return {        2.0f*(xz + yw),
                    2.0f*(yz - xw),
             1.0f - 2.0f*(xx + yy) };
}

inline EXPORT_CUDA Vector3& ExtractVector(Quaternion& quat)
{
	static_assert(4*sizeof(float) == sizeof(Quaternion), "Invalid quaternion size");
	return quat.Vector;
}

inline EXPORT_CUDA float& ExtractScalar(Quaternion& quat)
{
	return quat.Scalar;
}

inline EXPORT_CUDA const Vector3& ExtractVector(const Quaternion& quat)
{
	return quat.Vector;
}

inline EXPORT_CUDA const float& ExtractScalar(const Quaternion& quat)
{
	return quat.Scalar;
}

#ifndef Array
#	define Array(x) x.Components
#endif

//! Sets the quaternion to the multiplication identity quaternion, i.e. q = 1
inline EXPORT_CUDA Quaternion IdentityQuaternion()
{
	return Quaternion{ 0.0f, 0.0f, 0.0f, 1.0f };
}

//! Returns the conjugate quaternion
inline EXPORT_CUDA Quaternion Conjugate(const Quaternion& quat)
{
	Quaternion result;
    result.x = -quat.x;
    result.y = -quat.y;
    result.z = -quat.z;
    result.w = quat.w;
    return result;
}

//! Conjugates this quaternion
inline EXPORT_CUDA void ConjugateSelf(Quaternion* quat)
{
	quat->x = -quat->x;
    quat->y = -quat->y;
    quat->z = -quat->z;
    quat->w = quat->w;
}

//! Returns the inverse quaternion
/*!
    \remarks It is the same as returning the conjugate
*/
inline EXPORT_CUDA Quaternion Inverse(const Quaternion& quat)
{
	return Conjugate(quat);
}
    
//! Inverts the quaternion
/*!
    \remarks It is the same as conjugating the quaternion
*/
inline EXPORT_CUDA void InvertSelf(Quaternion* quat)
{
	ConjugateSelf(quat);
}

//! Computes the length of the quaternion as a 4D vector
inline EXPORT_CUDA float Length(const Quaternion& quat)
{
	return sqrtf(quat.x*quat.x + quat.y*quat.y + quat.z*quat.z + quat.w*quat.w);
}

//! Normalizes the quaternion
inline EXPORT_CUDA Quaternion Normalize(const Quaternion& quat)
{
	float l = Length(quat);
    if(l == 0)
		return Quaternion{};
	Quaternion ret = quat;
    ret.x /= l;
    ret.y /= l;
    ret.z /= l;
    ret.w /= l;
	return ret;
}

inline EXPORT_CUDA Quaternion ToQuaternionNormal(const Vector3& rhs)
{
    Quaternion result;
    float dot_vec = rhs.z;
    ExtractVector(result) = { -rhs.y, rhs.x, 0.0f };
    float len = Length(rhs);
    ExtractScalar(result) = len + dot_vec;
    return Normalize(result);
}

inline EXPORT_CUDA Quaternion FastRotationBetweenVectorQuaternion(const Vector3& lhs, const Vector3& rhs)
{
    Quaternion result;
    float dot_vec = Dot(lhs, rhs);
    ExtractVector(result) = Cross(lhs, rhs);
    float len = sqrtf(Dot(lhs, lhs) * Dot(rhs, rhs));
    ExtractScalar(result) = len + dot_vec;
    return Normalize(result);
}

//! Computes the dot product between two quaternion
inline EXPORT_CUDA float Dot(const Quaternion& lhs, const Quaternion& rhs)
{
	return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w;
}

//! Rotates the coordinate system around the relative x-axis
/*!
    \param yaw rotation around the x-axis
*/
inline EXPORT_CUDA Quaternion RotateX(const Quaternion& quat, float pitch)
{
	float   rx, rw;
    FastSinCos(pitch*0.5f, &rx, &rw);
    Quaternion ret;
    ret.w =  quat.w*rw - quat.x*rx;
    ret.x =  quat.w*rx + quat.x*rw;
    ret.y =  quat.y*rw + quat.z*rx;
    ret.z = -quat.y*rx + quat.z*rw;
    return Normalize(ret);
}

//! Rotates the coordinate system around the relative y-axis
/*!
    \param yaw rotation around the y-axis
*/
inline EXPORT_CUDA Quaternion RotateY(const Quaternion& quat, float yaw)
{
	float   ry, rw;
    FastSinCos(yaw*0.5f, &ry, &rw);
    Quaternion ret;
    ret.w = quat.w*rw - quat.y*ry;
    ret.x = quat.x*rw - quat.z*ry;
    ret.y = quat.w*ry + quat.y*rw;
    ret.z = quat.x*ry + quat.z*rw;
    return Normalize(ret);
}

//! Rotates the coordinate system around the relative z-axis
/*!
    \param yaw rotation around the z-axis
*/
inline EXPORT_CUDA Quaternion RotateZ(const Quaternion& quat, float roll)
{
	float   rz, rw;
    FastSinCos(roll*0.5f, &rz, &rw);
    Quaternion ret;
    ret.w =  quat.w*rw - quat.z*rz;
    ret.x =  quat.x*rw + quat.y*rz;
    ret.y = -quat.x*rz + quat.y*rw;
    ret.z =  quat.w*rz + quat.z*rw;
    return Normalize(ret);
}

//! Transforms a 3-dimensional vector
inline EXPORT_CUDA Vector3 Transform(const Quaternion& quat, const Vector3& v)
{
	return v + 2.0f*Cross(ExtractVector(quat), Cross(ExtractVector(quat), v) + ExtractScalar(quat)*v);
}

//! Multiplies a quaternion with a floating-point variable and stores the result in the quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator*=(Quaternion& quat, float f)
{
	quat.x *= f;
    quat.y *= f;
    quat.z *= f;
    quat.w *= f;
    return quat;
}

//! Divides a quaternion with a floating-point variable and stores the result in the quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator/=(Quaternion& quat, float f)
{
	quat.x /= f;
    quat.y /= f;
    quat.z /= f;
    quat.w /= f;
    return quat;
}

//! Compares two quaternions
/*! \related Quaternion */
inline EXPORT_CUDA bool operator==(const Quaternion& lhs, const Quaternion& rhs)
{
	return ApproxEqual(lhs.x, rhs.x) &&
           ApproxEqual(lhs.y, rhs.y) &&
           ApproxEqual(lhs.z, rhs.z) &&
           ApproxEqual(lhs.w, rhs.w);
}

//! Multiplies two quaternions and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator*(const Quaternion& lhs, const Quaternion& rhs)
{
	Quaternion result;
    result.w = lhs.w*rhs.w - lhs.x*rhs.x - lhs.y*rhs.y - lhs.z*rhs.z;
    result.x = lhs.w*rhs.x + lhs.x*rhs.w + lhs.y*rhs.z - lhs.z*rhs.y;
    result.y = lhs.w*rhs.y - lhs.x*rhs.z + lhs.y*rhs.w + lhs.z*rhs.x;
    result.z = lhs.w*rhs.z + lhs.x*rhs.y - lhs.y*rhs.x + lhs.z*rhs.w;
    return result;
}

//! Divides two quaternions and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator/(const Quaternion& lhs, const Quaternion& rhs)
{
	return lhs * Inverse(rhs);
}

//! Sums two quaternions and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator+(const Quaternion& lhs, const Quaternion& rhs)
{
	Quaternion result;
    result.x = lhs.x + rhs.x;
    result.y = lhs.y + rhs.y;
    result.z = lhs.z + rhs.z;
    result.w = lhs.w + rhs.w;
    return result;
}

//! Subtracts two quaternions and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator-(const Quaternion& lhs, const Quaternion& rhs)
{
	Quaternion result;
    result.x = lhs.x - rhs.x;
    result.y = lhs.y - rhs.y;
    result.z = lhs.z - rhs.z;
    result.w = lhs.w - rhs.w;
    return result;
}

//! Multiplies a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator*(float f, const Quaternion& quat)
{
	return Quaternion{ quat.x*f, quat.y*f, quat.z*f, quat.w*f };
}

//! Multiplies a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator*(const Quaternion& quat, float f)
{
	return Quaternion{ quat.x*f, quat.y*f, quat.z*f, quat.w*f };
}

//! Divides a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator/(const Quaternion& quat, float f)
{
	float f_rcp = 1.0f / f;
	return Quaternion{ quat.x * f_rcp, quat.y * f_rcp, quat.z * f_rcp, quat.w * f_rcp };
}

//! Negates a quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion operator-(const Quaternion& quat)
{
	return Quaternion{ -quat.x, -quat.y, -quat.z, -quat.w };
}

//! Multiplies two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator*=(Quaternion& lhs, const Quaternion& rhs)
{
	lhs = lhs * rhs;
    return lhs;
}

//! Sums two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator+=(Quaternion& lhs, const Quaternion& rhs)
{
	lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

//! Subtracts two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator-=(Quaternion& lhs, const Quaternion& rhs)
{
	lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}

//! Divides two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
inline EXPORT_CUDA Quaternion& operator/=(Quaternion& lhs, const Quaternion& rhs)
{
	return lhs *= Inverse(rhs);
}

inline EXPORT_CUDA Quaternion RotationQuaternion(float angle, const Vector3& axis)
{
    Quaternion rotation;
    float sin_a, cos_a;
    FastSinCos(angle*0.5f, &sin_a, &cos_a);
    rotation.x = axis.x * sin_a;
    rotation.y = axis.y * sin_a;
    rotation.z = axis.z * sin_a;
    rotation.w = cos_a;
    return Normalize(rotation);
}

inline EXPORT_CUDA Quaternion ConservativeRotationBetweenVectorQuaternion(const Vector3& lhs, const Vector3& rhs)
{
    return RotationQuaternion(acosf(Dot(lhs, rhs)), Normalize(Cross(lhs, rhs)));
}

//! Rotates the coordinate system
/*!
    \param angle the angle of rotation
    \param axis the relative axis of rotation
*/
inline EXPORT_CUDA Quaternion Rotate(const Quaternion& quat, float angle, const Vector3& axis)
{
    return quat * RotationQuaternion(angle, axis);
}

inline EXPORT_CUDA Quaternion SlerpUnsafe(const Quaternion& lhs, const Quaternion& rhs, float t)
{
	float cos_half_theta = Dot(lhs, rhs);
	if (fabsf(cos_half_theta) >= 0.999f)
	{
		return lhs;
	}

	float sin_half_theta = sqrtf(1.0f - cos_half_theta*cos_half_theta);
	float half_theta = acosf(cos_half_theta);
	
	float ratio_a = sin((1 - t) * half_theta) / sin_half_theta;
	float ratio_b = sin(t * half_theta) / sin_half_theta;

	return lhs*ratio_a + rhs*ratio_b;
}

inline EXPORT_CUDA Quaternion Slerp(const Quaternion& lhs, const Quaternion& rhs, float t)
{
	float cos_half_theta = Dot(lhs, rhs);
	if (fabsf(cos_half_theta) >= 0.999f)
	{
		return lhs;
	}

    float sign = 1.0f;
    if(cos_half_theta < 0.0f)
    {
        cos_half_theta = -cos_half_theta;
        sign = -1.0f;
    }

	float sin_half_theta = sqrtf(1.0f - cos_half_theta*cos_half_theta);
	float half_theta = acosf(cos_half_theta);
	
	float ratio_a = sinf((1 - t) * half_theta);
	float ratio_b = sinf(t * half_theta);

	return (lhs*ratio_a + rhs*ratio_b*sign) / sin_half_theta;
}

inline EXPORT_CUDA Quaternion Slerp(const Quaternion& quat, float t)
{
	float cos_half_theta = quat.w;
	if (fabsf(cos_half_theta) >= 0.999f)
	{
		return quat;
	}

	float sin_half_theta = sqrtf(1.0f - cos_half_theta*cos_half_theta);
	float half_theta = acosf(cos_half_theta);
	
	float ratio_a = sin((1 - t) * half_theta) / sin_half_theta;
	float ratio_b = sin(t * half_theta) / sin_half_theta;

	auto vec = ExtractVector(quat)*ratio_b;
	auto scalar = ratio_a + ExtractScalar(quat)*ratio_b;
	return Quaternion{ vec.x, vec.y, vec.z, scalar };
}

inline EXPORT_CUDA bool ApproxEqual(const Quaternion& lhs, const Quaternion& rhs, float epsilon)
{
	return ApproxEqual(lhs.x, rhs.x, epsilon) &&
           ApproxEqual(lhs.y, rhs.y, epsilon) &&
           ApproxEqual(lhs.z, rhs.z, epsilon) &&
           ApproxEqual(lhs.w, rhs.w, epsilon);
}
}

#endif // _TEMPEST_QUATERNION_HH_
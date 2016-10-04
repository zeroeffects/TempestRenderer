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

#ifndef _TEMPEST_VECTOR3_HH_
#define _TEMPEST_VECTOR3_HH_

#include "tempest/math/vector2.hh"
#include "tempest/math/functions.hh"

#include <algorithm>

#ifndef EXPORT_CUDA
#	ifdef __CUDACC__
#		define EXPORT_CUDA __device__ __host__
#	else
#		define EXPORT_CUDA
#	endif
#endif

namespace Tempest
{
//! 3-dimensional vector
/*! \ingroup TempestMath
*/
union Vector3
{
    struct
    {
        float x, /*!< x-coordinate component */
              y, /*!< y-coordinate component */
              z; /*!< z-coordinate component */
    };
    struct
    {
        Vector2 xy;
        float _z;
    };
    float Components[3];
};

inline EXPORT_CUDA Vector3 ToVector3(float* f) { return Vector3{ f[0], f[1], f[2] }; }

#ifndef Array
#	define Array(x) x.Components
#endif

//! Computes the length of the vector
/*!
    \return the length of the vector as a floating-point number
*/
inline EXPORT_CUDA float Length(const Vector3& vec) { return sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z); }

//! Normalizes the vector
/*!
    \remarks the function doesn't do anything if length is zero
*/
inline EXPORT_CUDA void NormalizeSelf(Vector3* vec)
{
    float l = Length(*vec);
    if(l != 0.0f)
    {
		float rcp_l = 1.0f/l;
        vec->x *= rcp_l;
        vec->y *= rcp_l;
        vec->z *= rcp_l;
    }
}

//! Computes the dot product between two vectors
/*! 
    \param vec a 3-dimesional vector
    \return the dot product between the vectors
*/
inline EXPORT_CUDA float Dot(const Vector3& lhs, const Vector3& rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; }

//! Computes the cross product between two vectors
/*!
    \param vec a 3-dimensional vector
    \return the cross product between the vectors
*/
inline EXPORT_CUDA Vector3 Cross(const Vector3& lhs, const Vector3& rhs) { return Vector3{lhs.y*rhs.z - lhs.z*rhs.y,
																		      lhs.z*rhs.x - lhs.x*rhs.z,
																		      lhs.x*rhs.y - lhs.y*rhs.x}; }

inline EXPORT_CUDA bool operator==(const Vector3& lhs, const Vector3& rhs) { return ApproxEqual(lhs.x, rhs.x) && ApproxEqual(lhs.y, rhs.y) && ApproxEqual(lhs.z, rhs.z); }

inline EXPORT_CUDA bool IsZero(const Vector3& vec)
{
    return vec.x && vec.y && vec.z;
}

inline EXPORT_CUDA bool ApproxEqual(const Vector3& lhs, const Vector3& rhs, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return ApproxEqual(lhs.x, rhs.x, epsilon) &&
           ApproxEqual(lhs.y, rhs.y, epsilon) &&
           ApproxEqual(lhs.z, rhs.z, epsilon);
}

inline EXPORT_CUDA bool ApproxEqual(const Vector3& vec, float scalar, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return ApproxEqual(vec.x, scalar, epsilon) &&
           ApproxEqual(vec.y, scalar, epsilon) &&
           ApproxEqual(vec.z, scalar, epsilon);
}

inline EXPORT_CUDA bool operator!=(const Vector3& lhs, const Vector3& rhs) { return ApproxNotEqual(lhs.x, rhs.x) || ApproxNotEqual(lhs.y, rhs.y) || ApproxNotEqual(lhs.z, rhs.z); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator+(const Vector3& lhs, const Vector3& rhs) { return Vector3{lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z}; }

inline EXPORT_CUDA Vector3 operator+(const Vector3& vec, float scalar) { return Vector3{vec.x + scalar, vec.y + scalar, vec.z + scalar}; }

inline EXPORT_CUDA Vector3 operator+(float scalar, const Vector3& vec) { return Vector3{vec.x + scalar, vec.y + scalar, vec.z + scalar}; }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator-(const Vector3& lhs, const Vector3& rhs) { return Vector3{lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z}; }

inline EXPORT_CUDA Vector3 operator-(const Vector3& vec, float scalar) { return Vector3{vec.x - scalar, vec.y - scalar, vec.z - scalar}; }

inline EXPORT_CUDA Vector3 operator-(float scalar, const Vector3& vec) { return Vector3{scalar - vec.x, scalar - vec.y, scalar - vec.z}; }

//! Negates a vector and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator-(const Vector3& vec) { return Vector3{-vec.x, -vec.y, -vec.z}; }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3& operator+=(Vector3& lhs, const Vector3& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3& operator-=(Vector3& lhs, const Vector3& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator*(const Vector3& vec, float a) { return Vector3{vec.x * a, vec.y * a, vec.z * a}; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator*(float a, const Vector3& vec) { return Vector3{vec.x * a, vec.y * a, vec.z * a}; }

//! Divides a float-pointing variable by vector component-wise and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator/(float a, const Vector3& vec) { return Vector3{a / vec.x, a / vec.y, a / vec.z}; }

//! Multiplies component-wise two vectors
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator*(const Vector3& lhs, const Vector3& rhs) { return Vector3{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z}; }

//! Divides component-wise two vectors
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator/(const Vector3& lhs, const Vector3& rhs) { return Vector3{lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z}; }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3& operator*=(Vector3& vec, float a) { vec.x *= a; vec.y *= a; vec.z *= a; return vec; }

inline EXPORT_CUDA Vector3& operator*=(Vector3& lhs, const Vector3& rhs) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3 operator/(const Vector3& vec, float a)
{
	static_assert(sizeof(Vector3) == 3*sizeof(float), "Vector3 has the wrong size"); 
	float rcp_a = 1.0f / a;
	return Vector3{vec.x * rcp_a, vec.y * rcp_a, vec.z * rcp_a};
}

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector3 */
inline EXPORT_CUDA Vector3& operator/=(Vector3& vec, float a)
{
	float rcp_a = 1.0f / a;
	vec.x *= rcp_a;
	vec.y *= rcp_a;
	vec.z *= rcp_a;
	return vec;
}

inline EXPORT_CUDA bool operator<(Vector3& lhs, const Vector3& rhs) { return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z; }

inline EXPORT_CUDA bool operator<(Vector3& lhs, float scalar) { return lhs.x < scalar && lhs.y < scalar && lhs.z < scalar; }

inline EXPORT_CUDA bool operator<=(const Vector3& lhs, const Vector3& rhs) { return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z; }

inline EXPORT_CUDA bool operator>(Vector3& lhs, const Vector3& rhs) { return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z; }

inline EXPORT_CUDA bool operator>(Vector3& lhs, float scalar) { return lhs.x > scalar && lhs.y > scalar && lhs.z > scalar; }

inline std::ostream& operator<<(std::ostream& os, const Vector3& val) { return os << "<" << val.x << ", " << val.y << ", " << val.z << ">"; }

inline EXPORT_CUDA Vector3 Normalize(const Vector3& vec)
{
    float l = Length(vec);
	return l != 0.0f ? (vec / l) : vec;
}

inline EXPORT_CUDA Vector3 Wedge(const Vector2& lhs, const Vector2& rhs) { return Vector3{ 0.0f, 0.0f, lhs.x*rhs.y - lhs.y*rhs.x }; }

//! Returns the component-wise absolute value of a 3-dimensional vector
inline EXPORT_CUDA Vector3 Vector3Abs(const Vector3& v)
{
	return Vector3{fabsf(v.x), fabsf(v.y), fabsf(v.z)};
}

inline EXPORT_CUDA Vector3 Vector3Pow(const Vector3& v, float p)
{
	return Vector3{powf(v.x, p), powf(v.y, p), powf(v.z, p)};
}

inline EXPORT_CUDA Vector3 Vector3Floor(const Vector3& v)
{
	return Vector3{FastFloor(v.x), FastFloor(v.y), FastFloor(v.z)};
}

inline EXPORT_CUDA Vector3 Vector3Ceil(const Vector3& v)
{
	return Vector3{FastCeil(v.x), FastCeil(v.y), FastCeil(v.z)};
}

inline EXPORT_CUDA Vector3 Vector3Clamp(const Vector3& val, const Vector3& vmin, const Vector3& vmax)
{
	return Vector3{Clamp(val.x, vmin.x, vmax.x),
				   Clamp(val.y, vmin.y, vmax.y),
				   Clamp(val.z, vmin.z, vmax.z)};
}

inline EXPORT_CUDA Vector3 Vector3Clamp(const Vector3& val, float smin, float smax)
{
	return Vector3{Clamp(val.x, smin, smax),
				   Clamp(val.y, smin, smax),
				   Clamp(val.z, smin, smax)};
}

inline EXPORT_CUDA Vector3 Vector3Saturate(const Vector3& val)
{
	return Vector3{Clamp(val.x, 0.0f, 1.0f),
				   Clamp(val.y, 0.0f, 1.0f),
				   Clamp(val.z, 0.0f, 1.0f)};
}

inline EXPORT_CUDA Vector3 Vector3Sqrt(const Vector3& val)
{
    return Vector3{sqrtf(val.x),
                   sqrtf(val.y),
                   sqrtf(val.z)};
}

inline EXPORT_CUDA Vector3 GenericSqrt(const Vector3& val)
{
    return Vector3Sqrt(val);
}

inline EXPORT_CUDA Vector3 Vector3Min(const Vector3& val, const Vector3& vmax)
{
	return Vector3{Minf(val.x, vmax.x),
				   Minf(val.y, vmax.y),
				   Minf(val.z, vmax.z)};
}

inline EXPORT_CUDA Vector3 Vector3Max(const Vector3& val, float scalar)
{
	return Vector3{Maxf(val.x, scalar),
				   Maxf(val.y, scalar),
				   Maxf(val.z, scalar)};
}

inline EXPORT_CUDA Vector3 Vector3Max(const Vector3& lhs, const Vector3& rhs)
{
	return Vector3{Maxf(lhs.x, rhs.x),
				   Maxf(lhs.y, rhs.y),
				   Maxf(lhs.z, rhs.z)};
}

inline EXPORT_CUDA Vector3 GenericMin(const Vector3& lhs, const Vector3& rhs) { return Vector3Min(lhs, rhs); }

inline EXPORT_CUDA Vector3 GenericMax(const Vector3& lhs, const Vector3& rhs) { return Vector3Max(lhs, rhs); }

inline EXPORT_CUDA Vector3 Vector3Exp(const Vector3& val)
{
    return Vector3{expf(val.x), expf(val.y), expf(val.z)};
}

inline EXPORT_CUDA Vector3 Vector3Log(const Vector3& val)
{
    return Vector3{ logf(val.x), logf(val.y), logf(val.z) };
}

inline EXPORT_CUDA Vector3 Vector3Log10(const Vector3& val)
{
    return Vector3{ log10f(val.x), log10f(val.y), log10f(val.z) };
}

inline EXPORT_CUDA Vector3 GenericLog10(const Vector3& val)
{
    return Vector3Log10(val);
}

inline EXPORT_CUDA Vector3 Vector3Cos(const Vector3& val)
{
    return Vector3{cosf(val.x), cosf(val.y), cosf(val.z)};
}

inline EXPORT_CUDA Vector3 Vector3Sinc(const Vector3& val)
{
    return Vector3{Sinc(val.x), Sinc(val.y), Sinc(val.z)};
}

inline EXPORT_CUDA void CopyVec3ToFloatArray(const Vector3& vec, float* arr)
{
    arr[0] = vec.x;
    arr[1] = vec.y;
    arr[2] = vec.z;
}

inline EXPORT_CUDA Vector3 ToDegress(const Vector3& vec) { return Vector3{ToDegress(vec.x), ToDegress(vec.y), ToDegress(vec.z)}; }

inline EXPORT_CUDA Vector3 ToRadians(const Vector3& vec) { return Vector3{ToRadians(vec.x), ToRadians(vec.y), ToRadians(vec.z)}; }

inline EXPORT_CUDA Vector3 ToVector3(float value)
{
    return Vector3{value, value, value};
}

inline EXPORT_CUDA Vector3 SelectGE(const Vector3& vec, float value, const Vector3& true_vec, const Vector3& false_vec)
{
    return Vector3{ vec.x >= value ? true_vec.x : false_vec.x,
                    vec.y >= value ? true_vec.y : false_vec.y,
                    vec.z >= value ? true_vec.z : false_vec.z };
}

inline EXPORT_CUDA float MaxValue(const Vector3& vec)
{
    return Maxf(Maxf(vec.x, vec.y), vec.z);
}

inline EXPORT_CUDA float MinValue(const Vector3& vec)
{
    return Minf(Minf(vec.x, vec.y), vec.z);
}

inline EXPORT_CUDA Vector2 ToVector2Trunc(const Vector3& vec)
{
    return Vector2{ vec.x, vec.y };
}

inline EXPORT_CUDA Vector3 Interpolate(const Vector3& lhs, const Vector3& rhs, float t)
{
    return lhs*(1 - t) + rhs*t;
}

inline EXPORT_CUDA Vector3 SphereToCartesianCoordinates(const Vector2& angles)
{
    float cos_theta, sin_theta, cos_phi, sin_phi;
    FastSinCos(angles.x, &sin_theta, &cos_theta);
    FastSinCos(angles.y, &sin_phi, &cos_phi);

    return { cos_phi*sin_theta, sin_phi*sin_theta, cos_theta };
}

inline EXPORT_CUDA Vector2 CartesianToParabolicCoordinates(const Vector3& coordinates)
{
    return { coordinates.x/(coordinates.z + 1.0f), coordinates.y/(coordinates.z + 1.0f) };
}

inline EXPORT_CUDA Vector3 ParabolicToCartesianCoordinates(const Vector2& coordinates)
{
    float len_sq = Dot(coordinates, coordinates);
    return { 2.0f*coordinates.x/(len_sq + 1.0f), 2.0f*coordinates.y/(len_sq + 1.0f), (1 - len_sq) / (1 + len_sq) };
}

inline EXPORT_CUDA Tempest::Vector3 ParabolicMapToCartesianCoordinates(const Tempest::Vector2& tc)
{
    return Tempest::ParabolicToCartesianCoordinates(2.0f*tc - Tempest::ToVector2(1.0f));
}

inline EXPORT_CUDA Tempest::Vector2 CartesianToParabolicMapCoordinates(const Tempest::Vector3& dir)
{
    return Tempest::CartesianToParabolicCoordinates(dir)*0.5f + Tempest::ToVector2(0.5f);
}

inline EXPORT_CUDA Vector2 CartesianToLambertEqualAreaCoordinates(Vector3 normal)
{
    float f = sqrt(8*normal.z + 8);
    return { normal.x/f + 0.5f, normal.y/f + 0.5f };
}

inline EXPORT_CUDA Vector3 LambertEqualAreaToCartesianCoordinates(Vector2 enc)
{
    Vector2 fenc = enc*4 - 2;
    float f = Dot(fenc, fenc);
    float g = sqrtf(1.0f - f/4);
    auto interm = fenc * g;
    return { interm.x, interm.y, 1.0f - f/2 };
}

inline EXPORT_CUDA Vector3 Reflect(const Vector3& inc, const Vector3& norm) { return 2.0f*Dot(norm, inc)*norm - inc; }

// Refer to the Reshetov10 paper for explanation what all of this means
inline EXPORT_CUDA Vector3 ComputeConsistentNormal(Vector3 dir, Vector3 norm, float a_coef)
{
	float q_numer = (1.0f - (2.0f/MathPi)*a_coef);
	float q_coef = q_numer*q_numer/(1.0f + 2.0f*(1.0f - 2.0f/MathPi)*a_coef);

	float b_coef = Dot(dir, norm);
	float g_coef = 1.0f + q_coef*(b_coef - 1.0f);
	float ratio = sqrtf(q_coef*(1.0f + g_coef)/(1.0f + b_coef));
	Vector3 refl = (g_coef + ratio*b_coef)*norm - ratio*dir;
			
	return Normalize(dir + refl);
}
}

#endif // _TEMPEST_VECTOR3_HH_

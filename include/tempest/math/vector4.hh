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

#ifndef _TEMPEST_VECTOR4_HH_
#define _TEMPEST_VECTOR4_HH_

#include "tempest/math/vector2.hh"
#include "tempest/math/vector3.hh"

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
//! 4-dimensional vector
/*!
    \ingroup TempestMath
*/
union Vector4
{
    struct
    {
        float x, /*!< x-coordinate component */
		      y, /*!< y-coordinate component */
		      z, /*!< z-coordinate component */
		      w; /*!< w-coordinate component */
    };
    float Components[4];
};

#ifndef Array
#	define Array(x) x.Components
#endif

inline EXPORT_CUDA float Element(const Vector4& vec, size_t i) { return Array(vec)[i]; }

//! Computes the length of the vector
/*! This function uses the following equation:
    \f[\|\vec{v}\| = \frac{\sqrt{v_{x}^{2} + v_{y}^{2} + v_{z}^{2}}}{v_w}\f]
    \return the length of the vector as a floating-point number
    \warning the function doesn't check for \f$v_w \neq 0\f$
*/
inline EXPORT_CUDA float Length(const Vector4& vec) { return sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z)/vec.w; }

//! Computes the length of the 4D vector
/*! This function uses the following equation:
    \f[\|\vec{v}\| = \sqrt{v_{x}^{2} + v_{y}^{2} + v_{z}^{2} + v_{w}^{2}}\f]
    \return the length of the vector as a floating-point number
    \warning the function doesn't check for \f$v_w \neq 0\f$
*/
inline EXPORT_CUDA float Length4D(const Vector4& vec) { return sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z + vec.w*vec.w); }

//! Partially normalizes the component of the vector
/*! The function divides all of the coordinate component by:
    \f[\|\vec{v_{xyz}}\| = \sqrt{v_{x}^{2} + v_{y}^{2} + v_{z}^{2}}\f]
    \warning \f$\vec{v_w} = \frac{\vec{v_w}}{|\vec{v_{xyz}}\|}\f$
*/
inline EXPORT_CUDA void NormalizePartialSelf(Vector4* vec)
{
    float l = sqrt(vec->x*vec->x+vec->y*vec->y+vec->z*vec->z);
    if(l != 0.0f)
    {
        vec->x /= l;
        vec->y /= l;
        vec->z /= l;
        vec->w /= l;
    }
}

//! Computes the dot product between two vectors
/*! The formula used for this function is as follows: 
    \f[a_x b_x + a_y b_y + a_z b_z + a_w b_w\f]
    \param vec a 4-dimensional vector
    \returns the dot product between the two vectors
*/
inline EXPORT_CUDA float Dot(const Vector4& lhs, const Vector4& rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w; }

//! Computes the dot product between two vectors
/*! The formula used for this function is as follows: 
    \f[a_x b_x + a_y b_y + a_z b_z + a_w\f]
    Where \f$\vec{a}\f$ is this vector.
    \param vec a 3-dimensional vector
    \returns the dot product between the two vectors
*/
inline EXPORT_CUDA float Dot(const Vector4& lhs, const Vector3& rhs) { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w; }

inline EXPORT_CUDA Vector4 ToVector4(const Vector3& vec) { return Vector4{vec.x, vec.y, vec.z, 1.0f}; }

inline EXPORT_CUDA Vector4 ToVector4(float x) { return Vector4{x, x, x, x}; }

inline EXPORT_CUDA Vector4 ToVector4(uint32_t color)
{
    float coef = (1.0f/255.0f);
    return Vector4{ coef*rgbaR(color), coef*rgbaG(color), coef*rgbaB(color), (float)coef*rgbaA(color) };
}

inline EXPORT_CUDA Vector3 ToVector3(const Vector4& vec)
{
	static_assert(sizeof(Vector4) == 4*sizeof(float), "Vector4 has the wrong size");
    auto w_rcp = 1.0f / vec.w;
	return Vector3{vec.x * w_rcp, vec.y * w_rcp, vec.z * w_rcp};
}

inline EXPORT_CUDA Vector4 Vector4Log(const Vector4& vec)
{
    return { logf(vec.x), logf(vec.y), logf(vec.z), logf(vec.w) };
}

inline EXPORT_CUDA Vector3 ToVector3Trunc(const Vector4& vec)
{
	return Vector3{vec.x, vec.y, vec.z};
}

inline EXPORT_CUDA Vector2 ToVector2Trunc(const Vector4& vec)
{
    return Vector2{ vec.x, vec.y };
}

inline EXPORT_CUDA bool ApproxEqual(const Vector4& lhs, const Vector4& rhs, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return ApproxEqual(lhs.x, rhs.x, epsilon) &&
           ApproxEqual(lhs.y, rhs.y, epsilon) &&
           ApproxEqual(lhs.z, rhs.z, epsilon) &&
           ApproxEqual(lhs.w, rhs.w, epsilon);
}

inline EXPORT_CUDA bool ApproxEqual(const Vector4& vec, float scalar, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return ApproxEqual(vec.x, scalar, epsilon) &&
           ApproxEqual(vec.y, scalar, epsilon) &&
           ApproxEqual(vec.z, scalar, epsilon) &&
           ApproxEqual(vec.w, scalar, epsilon);
}

inline EXPORT_CUDA bool operator==(const Vector4& lhs, const Vector4& rhs) { return ApproxEqual(lhs.x, rhs.x) && ApproxEqual(lhs.y, rhs.y) && ApproxEqual(lhs.z, rhs.z) && ApproxEqual(lhs.w, rhs.w); }

inline EXPORT_CUDA bool operator!=(const Vector4& lhs, const Vector4& rhs) { return ApproxNotEqual(lhs.x, rhs.x) || ApproxNotEqual(lhs.y, rhs.y) || ApproxNotEqual(lhs.z, rhs.z) || ApproxNotEqual(lhs.w, rhs.w); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator+(const Vector4& lhs, const Vector4& rhs) { return Vector4{lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z, lhs.w+rhs.w}; }

inline EXPORT_CUDA Vector4 operator+(const Vector4& vec, float scalar) { return Vector4{vec.x + scalar, vec.y + scalar, vec.z + scalar, vec.w + scalar}; }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator-(const Vector4& lhs, const Vector4& rhs) { return Vector4{lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z, lhs.w-rhs.w}; }

inline EXPORT_CUDA Vector4 operator-(const Vector4& vec, float scalar) { return Vector4{vec.x - scalar, vec.y - scalar, vec.z - scalar, vec.w - scalar}; }

//! Negates a vector and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator-(const Vector4& vec) { return Vector4{-vec.x, -vec.y, -vec.z, -vec.w}; }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4& operator+=(Vector4& lhs, const Vector4& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }

inline EXPORT_CUDA Vector4& operator+=(Vector4& vec, float scalar) { vec.x += scalar; vec.y += scalar; vec.z += scalar; vec.w += scalar; return vec; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4& operator-=(Vector4& lhs, const Vector4& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }

inline EXPORT_CUDA Vector4& operator-=(Vector4& vec, float scalar) { vec.x -= scalar; vec.y -= scalar; vec.z -= scalar; vec.w -= scalar; return vec; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator*(const Vector4& vec, float a) { return Vector4{vec.x * a, vec.y * a, vec.z * a, vec.w * a}; }

inline EXPORT_CUDA Vector4 operator*(const Vector4& lhs, const Vector4& rhs) { return Vector4{lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w}; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator*(float a, const Vector4& vec) { return Vector4{vec.x * a, vec.y * a, vec.z * a, vec.w * a}; }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4& operator*=(Vector4& vec, float a) { vec.x *= a; vec.y *= a; vec.z *= a, vec.w *= a; return vec; }

inline EXPORT_CUDA Vector4& operator*=(Vector4& lhs, const Vector4& rhs) { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; lhs.w *= rhs.w; return lhs; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4 operator/(const Vector4& vec, float a)
{
	float rcp_a = 1.0f / a;
	return Vector4{vec.x * rcp_a, vec.y * rcp_a, vec.z * rcp_a, vec.w * rcp_a};
}

inline EXPORT_CUDA Vector4 operator/(const Vector4& lhs, const Vector4& rhs) { return Vector4{lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w}; }

inline EXPORT_CUDA Vector4& operator/=(Vector4& lhs, const Vector4& rhs) { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs; }

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector4 */
inline EXPORT_CUDA Vector4& operator/=(Vector4& vec, float a)
{
	float rcp_a = 1.0f / a;
	vec.x *= rcp_a;
	vec.y *= rcp_a;
	vec.z *= rcp_a;
	vec.w *= rcp_a;
	return vec;
}

inline EXPORT_CUDA float MinValue(const Vector4& vec)
{
    return Minf(Minf(Minf(vec.x, vec.y), vec.z), vec.w);
}

inline EXPORT_CUDA float MaxValue(const Vector4& vec)
{
    return Maxf(Maxf(Maxf(vec.x, vec.y), vec.z), vec.w);
}
}

#endif // _TEMPEST_VECTOR4_HH_

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

#ifndef _TEMPEST_VECTOR2_HH_
#define _TEMPEST_VECTOR2_HH_

#include "tempest/utils/macros.hh"
#include "tempest/utils/assert.hh"
#include "tempest/math/functions.hh"

#include <cstddef>
#include <cmath>

namespace Tempest
{
//! 2-dimensional vector
/*! \ingroup TempestMath
*/
union Vector2
{
    struct
    {
    float x, /*!< x-coordinate component */
          y; /*!< y-coordinate component */
    };
    float Components[2];
};

#ifndef Array
#	define Array(x) x.Components
#endif

inline EXPORT_CUDA float Element(const Vector2& vec, size_t i) { return Array(vec)[i]; }

//! Computes the length of the vector
/*!
    \return the length of the vector as a floating-point number
*/
inline EXPORT_CUDA float Length(const Vector2& vec) { return sqrt(vec.x*vec.x + vec.y*vec.y); }
    
//! Normalizes the vector
/*!
    \remarks the function doesn't do anything if length is zero
*/
inline EXPORT_CUDA void NormalizeSelf(Vector2* vec)
{
    float l = Length(*vec);
    if(l != 0.0f)
    {
        vec->x /= l;
        vec->y /= l;
    }
}

inline EXPORT_CUDA Vector2 Normalize(const Vector2& vec)
{
    float l = Length(vec);
	return l != 0.0f ? Vector2{ vec.x / l, vec.y / l } : vec;
}

inline EXPORT_CUDA Vector2 Vector2Abs(const Vector2& vec)
{
    return Vector2{fabsf(vec.x), fabsf(vec.y)};
}

inline EXPORT_CUDA Vector2 Vector2Floor(const Vector2& vec)
{
    return Vector2{FastFloor(vec.x), FastFloor(vec.y)};
}

inline EXPORT_CUDA Vector2 Vector2Exp(const Vector2& vec)
{
    return Vector2{ expf(vec.x), expf(vec.y) };
}

inline EXPORT_CUDA Vector2 Vector2Log(const Vector2& vec)
{
    return Vector2{ logf(vec.x), logf(vec.y) };
}

inline EXPORT_CUDA Vector2 Vector2Ceil(const Vector2& vec)
{
    return Vector2{ceilf(vec.x), ceilf(vec.y)};
}

inline EXPORT_CUDA Vector2 Vector2Sqrt(const Vector2& vec)
{
    return Vector2{sqrtf(vec.x), sqrtf(vec.y)};
}

inline EXPORT_CUDA Vector2 ToVector2(float scalar)
{
    return Vector2{scalar, scalar};
}

inline EXPORT_CUDA Vector2 Vector2Pow(const Vector2& lhs, const Vector2& rhs)
{
    return Vector2{powf(lhs.x, rhs.x),
                   powf(lhs.y, rhs.y)};
}

inline EXPORT_CUDA Vector2 Vector2Pow(const Vector2& vec, float scalar)
{
    return Vector2{powf(vec.x, scalar),
                   powf(vec.y, scalar)};
}

inline EXPORT_CUDA Vector2 Vector2Min(const Vector2& lhs, const Vector2& rhs)
{
    return Vector2{Minf(lhs.x, rhs.x),
                   Minf(lhs.y, rhs.y)};
}

inline EXPORT_CUDA Vector2 Vector2Min(const Vector2& lhs, float scalar)
{
    return Vector2{Minf(lhs.x, scalar),
                   Minf(lhs.y, scalar)};
}

inline EXPORT_CUDA Vector2 Vector2Max(const Vector2& lhs, const Vector2& rhs)
{
    return Vector2{Maxf(lhs.x, rhs.x),
                   Maxf(lhs.y, rhs.y)};
}

inline EXPORT_CUDA Vector2 Vector2Max(const Vector2& lhs, float scalar)
{
    return Vector2{Maxf(lhs.x, scalar),
                   Maxf(lhs.y, scalar)};
}

inline EXPORT_CUDA Vector2 Vector2Clamp(const Vector2& val, const Vector2& vmin, const Vector2& vmax)
{
	return Vector2{Clamp(val.x, vmin.x, vmax.x),
				   Clamp(val.y, vmin.y, vmax.y)};
}

inline EXPORT_CUDA Vector2 Vector2Clamp(const Vector2& val, float smin, float smax)
{
	return Vector2{Clamp(val.x, smin, smax),
				   Clamp(val.y, smin, smax)};
}

// For template purposes
inline EXPORT_CUDA Vector2 GenericMin(const Vector2& lhs, const Vector2& rhs) { return Vector2Min(lhs, rhs); }
inline EXPORT_CUDA Vector2 GenericMax(const Vector2& lhs, const Vector2& rhs) { return Vector2Max(lhs, rhs); }

inline EXPORT_CUDA float MaxValue(const Vector2& vec)
{
    return Maxf(vec.x, vec.y);
}

inline EXPORT_CUDA float MinValue(const Vector2& vec)
{
    return Minf(vec.x, vec.y);
}

//! Computes the dot product between two vectors
/*!
    \param vec a 2-dimesional vector
    \return the dot product between the vectors
*/
inline EXPORT_CUDA float Dot(const Vector2& lhs, const Vector2& rhs) { return lhs.x*rhs.x + lhs.y*rhs.y; }

inline EXPORT_CUDA float WedgeZ(const Vector2& lhs, const Vector2& rhs) { return lhs.x*rhs.y - lhs.y*rhs.x; }

inline EXPORT_CUDA bool operator==(const Vector2& lhs, const Vector2& rhs) { return ApproxEqual(lhs.x, rhs.x) && ApproxEqual(lhs.y, rhs.y); }

inline EXPORT_CUDA bool operator!=(const Vector2& lhs, const Vector2& rhs) { return ApproxNotEqual(lhs.x, rhs.x) || ApproxNotEqual(lhs.y, rhs.y); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator+(const Vector2& lhs, const Vector2& rhs) { return Vector2{lhs.x+rhs.x, lhs.y+rhs.y}; }

inline EXPORT_CUDA Vector2 operator+(const Vector2& vec, float scalar) { return Vector2{vec.x + scalar, vec.y + scalar}; }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator-(const Vector2& lhs, const Vector2& rhs) { return Vector2{lhs.x-rhs.x, lhs.y-rhs.y}; }

inline EXPORT_CUDA Vector2 operator-(const Vector2& vec, float scalar) { return Vector2{vec.x - scalar, vec.y - scalar}; }

//! Negates a vector and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator-(const Vector2& vec) { return Vector2{-vec.x, -vec.y}; }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2& operator+=(Vector2& lhs, const Vector2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }

inline EXPORT_CUDA Vector2& operator+=(Vector2& vec, float scalar) { vec.x += scalar; vec.y += scalar; return vec; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2& operator-=(Vector2& lhs, const Vector2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }

inline EXPORT_CUDA Vector2& operator-=(Vector2& vec, float scalar) { vec.x -= scalar; vec.y -= scalar; return vec; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator*(const Vector2& vec, float a) { return Vector2{vec.x * a, vec.y * a}; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator*(float a, const Vector2& vec) { return Vector2{vec.x * a, vec.y * a}; }

//! Multiplies two vectors in component-wise fashion and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator*(const Vector2& lhs, const Vector2& rhs) { return Vector2{ lhs.x*rhs.x, lhs.y*rhs.y }; }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2& operator*=(Vector2& vec, float a) { vec.x *= a; vec.y *= a; return vec; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2 operator/(const Vector2& vec, float a) { return Vector2{vec.x / a, vec.y / a}; }

inline EXPORT_CUDA Vector2 operator/(const Vector2& lhs, const Vector2& rhs) { return Vector2{lhs.x/rhs.x, lhs.y/rhs.y}; }

inline EXPORT_CUDA Vector2 operator/(float a, const Vector2& vec) { return Vector2{ a / vec.x, a / vec.y }; }

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector2 */
inline EXPORT_CUDA Vector2& operator/=(Vector2& vec, float a) { vec.x /= a; vec.y /= a; return vec; }

inline EXPORT_CUDA bool operator<=(const Vector2& vec, float a) { return vec.x <= a && vec.y <= a; }

inline EXPORT_CUDA bool operator<=(float a, const Vector2& vec) { return a <= vec.x && a <= vec.y; }

inline EXPORT_CUDA bool operator<=(const Vector2& lhs, const Vector2& rhs) { return lhs.x <= rhs.x && lhs.y <= rhs.y; }

inline EXPORT_CUDA bool ApproxEqual(const Vector2& lhs, const Vector2& rhs, float epsilon)
{
    return ApproxEqual(lhs.x, rhs.x, epsilon) &&
           ApproxEqual(lhs.y, rhs.y, epsilon);
}
}

#endif // _TEMPEST_VECTOR2_HH_
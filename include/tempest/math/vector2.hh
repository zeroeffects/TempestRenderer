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

//! 2-dimensional vector
/*! \ingroup TempestMath
*/
struct Vector2
{
    struct coord
    {
        float x, /*!< x-coordinate component */
              y; /*!< y-coordinate component */
    };

    union
    {
        coord coordinate;
        float elem[2]; /*!< coordinate array */
    };

    //! Default constructor
    explicit Vector2() { static_assert(sizeof(Vector2) == 2*sizeof(float), "Vector2 has the wrong size"); }

    //! Constructor
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
    */
    explicit Vector2(float _x, float _y)
    {
        set(_x, _y);
    }

    Vector2(const Vector2& v)
    {
        coordinate = v.coordinate;
    }
    
    inline float& x() { return coordinate.x; }
    inline float& y() { return coordinate.y; }
    inline float& r() { return coordinate.x; }
    inline float& g() { return coordinate.y; }
    inline float& s() { return coordinate.x; }
    inline float& t() { return coordinate.y; }

    inline float x() const { return coordinate.x; }
    inline float y() const { return coordinate.y; }
    inline float r() const { return coordinate.x; }
    inline float g() const { return coordinate.y; }
    inline float s() const { return coordinate.x; }
    inline float t() const { return coordinate.y; }

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float& operator[](size_t i)
    {
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element coordex");
        return elem[i];
    }

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float operator[](size_t i) const
    {
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element coordex");
        return elem[i];
    }
    //! Computes the length of the vector
    /*!
        \return the length of the vector as a floating-point number
    */
    float length() const { return sqrt(coordinate.x*coordinate.x+coordinate.y*coordinate.y); }
    
    //! Normalizes the vector
    /*!
        \remarks the function doesn't do anything if length is zero
    */
    void normalize()
    {
        float l = length();
        if(l != 0.0f)
        {
            coordinate.x /= l;
            coordinate.y /= l;
        }
    }

    //! Computes the dot product between two vectors
    /*!
        \param vec a 2-dimesional vector
        \return the dot product between the vectors
    */
    float dot(const Vector2& vec) const { return coordinate.x*vec.coordinate.x + coordinate.y*vec.coordinate.y; }

    //! Sets the values of the coordinate component
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
    */
    inline void set(float _x, float _y) { coordinate.x = _x; coordinate.y = _y; }
};

inline bool operator==(const Vector2& lhs, const Vector2& rhs) { return approx_eq(lhs.coordinate.x, rhs.coordinate.x) && approx_eq(lhs.coordinate.y, rhs.coordinate.y); }

inline bool operator!=(const Vector2& lhs, const Vector2& rhs) { return approx_neq(lhs.coordinate.x, rhs.coordinate.x) || approx_neq(lhs.coordinate.y, rhs.coordinate.y); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator+(const Vector2& lhs, const Vector2& rhs) { return Vector2(lhs.coordinate.x+rhs.coordinate.x, lhs.coordinate.y+rhs.coordinate.y); }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator-(const Vector2& lhs, const Vector2& rhs) { return Vector2(lhs.coordinate.x-rhs.coordinate.x, lhs.coordinate.y-rhs.coordinate.y); }

//! Negates a vector and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator-(const Vector2& vec) { return Vector2(-vec.coordinate.x, -vec.coordinate.y); }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector2 */
inline Vector2& operator+=(Vector2& lhs, const Vector2& rhs) { lhs.coordinate.x += rhs.coordinate.x; lhs.coordinate.y += rhs.coordinate.y; return lhs; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector2 */
inline Vector2& operator-=(Vector2& lhs, const Vector2& rhs) { lhs.coordinate.x -= rhs.coordinate.x; lhs.coordinate.y -= rhs.coordinate.y; return lhs; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator*(const Vector2& vec, float a) { return Vector2(vec.coordinate.x * a, vec.coordinate.y * a); }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator*(float a, const Vector2& vec) { return Vector2(vec.coordinate.x * a, vec.coordinate.y * a); }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector2 */
inline Vector2& operator*=(Vector2& vec, float a) { vec.coordinate.x *= a; vec.coordinate.y *= a; return vec; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector2 */
inline Vector2 operator/(const Vector2& vec, float a) { return Vector2(vec.coordinate.x / a, vec.coordinate.y / a); }

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector2 */
inline Vector2& operator/=(Vector2& vec, float a) { vec.coordinate.x /= a; vec.coordinate.y /= a; return vec; }

#endif // _TEMPEST_VECTOR2_HH_
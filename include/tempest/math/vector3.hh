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

namespace Tempest
{
//! 3-dimensional vector
/*! \ingroup TempestMath
*/
struct Vector3
{
    struct coord
    {
        float x, /*!< x-coordinate component */
              y, /*!< y-coordinate component */
              z; /*!< z-coordinate component */
    };

    union
    {
        coord coordinate;
        float elem[3]; /*!< coordinate array */
    };

    //! Default constructor
    explicit Vector3() { static_assert(sizeof(Vector3) == 3*sizeof(float), "Vector3 has the wrong size"); }

    Vector3(const Vector3& v)
    {
        coordinate.x = v.coordinate.x, coordinate.y = v.coordinate.y, coordinate.z = v.coordinate.z;
    }
    
    //! Constructor
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
        \param _z z-coordinate component
    */
    Vector3(float _x, float _y, float _z)
    {
        set(_x, _y, _z);
    }

    inline float& x() { return coordinate.x; }
    inline float& y() { return coordinate.y; }
    inline float& z() { return coordinate.z; }
    inline float& r() { return coordinate.x; }
    inline float& g() { return coordinate.y; }
    inline float& b() { return coordinate.z; }
    inline float& s() { return coordinate.x; }
    inline float& t() { return coordinate.y; }
    inline float& p() { return coordinate.z; }

    inline float x() const { return coordinate.x; }
    inline float y() const { return coordinate.y; }
    inline float z() const { return coordinate.z; }
    inline float r() const { return coordinate.x; }
    inline float g() const { return coordinate.y; }
    inline float b() const { return coordinate.z; }
    inline float s() const { return coordinate.x; }
    inline float t() const { return coordinate.y; }
    inline float p() const { return coordinate.z; }

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
    float length() const { return sqrt(coordinate.x*coordinate.x+coordinate.y*coordinate.y+coordinate.z*coordinate.z); }

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
            coordinate.z /= l;
        }
    }

    //! Computes the dot product between two vectors
    /*! 
        \param vec a 3-dimesional vector
        \return the dot product between the vectors
    */
    float dot(const Vector3& vec) const { return coordinate.x*vec.coordinate.x + coordinate.y*vec.coordinate.y + coordinate.z*vec.coordinate.z; }

    //! Computes the cross product between two vectors
    /*!
        \param vec a 3-dimensional vector
        \return the cross product between the vectors
    */
    Vector3 cross(const Vector3& vec) const { return Vector3(coordinate.y*vec.coordinate.z - coordinate.z*vec.coordinate.y,
                                                             coordinate.z*vec.coordinate.x - coordinate.x*vec.coordinate.z,
                                                             coordinate.x*vec.coordinate.y - coordinate.y*vec.coordinate.x); }

    //! Sets the values of the coordinate component
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
        \param _z z-coordinate component
    */
    inline void set(float _x, float _y, float _z) { coordinate.x = _x; coordinate.y = _y; coordinate.z = _z; }
};

inline bool operator==(const Vector3& lhs, const Vector3& rhs) { return approx_eq(lhs.coordinate.x, rhs.coordinate.x) && approx_eq(lhs.coordinate.y, rhs.coordinate.y) && approx_eq(lhs.coordinate.z, rhs.coordinate.z); }

inline bool operator!=(const Vector3& lhs, const Vector3& rhs) { return approx_neq(lhs.coordinate.x, rhs.coordinate.x) || approx_neq(lhs.coordinate.y, rhs.coordinate.y) || approx_neq(lhs.coordinate.z, rhs.coordinate.z); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator+(const Vector3& lhs, const Vector3& rhs) { return Vector3(lhs.coordinate.x+rhs.coordinate.x, lhs.coordinate.y+rhs.coordinate.y, lhs.coordinate.z+rhs.coordinate.z); }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator-(const Vector3& lhs, const Vector3& rhs) { return Vector3(lhs.coordinate.x-rhs.coordinate.x, lhs.coordinate.y-rhs.coordinate.y, lhs.coordinate.z-rhs.coordinate.z); }

//! Negates a vector and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator-(const Vector3& vec) { return Vector3(-vec.coordinate.x, -vec.coordinate.y, -vec.coordinate.z); }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector3 */
inline Vector3& operator+=(Vector3& lhs, const Vector3& rhs) { lhs.coordinate.x += rhs.coordinate.x; lhs.coordinate.y += rhs.coordinate.y; lhs.coordinate.z += rhs.coordinate.z; return lhs; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector3 */
inline Vector3& operator-=(Vector3& lhs, const Vector3& rhs) { lhs.coordinate.x -= rhs.coordinate.x; lhs.coordinate.y -= rhs.coordinate.y; lhs.coordinate.z -= rhs.coordinate.z; return lhs; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator*(const Vector3& vec, float a) { return Vector3(vec.coordinate.x * a, vec.coordinate.y * a, vec.coordinate.z * a); }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator*(float a, const Vector3& vec) { return Vector3(vec.coordinate.x * a, vec.coordinate.y * a, vec.coordinate.z * a); }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector3 */
inline Vector3& operator*=(Vector3& vec, float a) { vec.coordinate.x *= a; vec.coordinate.y *= a; vec.coordinate.z *= a; return vec; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector3 */
inline Vector3 operator/(const Vector3& vec, float a)
{
	float rcp_a = 1.0f / a;
	return Vector3(vec.coordinate.x * rcp_a, vec.coordinate.y * rcp_a, vec.coordinate.z * rcp_a);
}

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector3 */
inline Vector3& operator/=(Vector3& vec, float a)
{
	float rcp_a = 1.0f / a;
	vec.coordinate.x *= rcp_a;
	vec.coordinate.y *= rcp_a;
	vec.coordinate.z *= rcp_a;
	return vec;
}

//! Returns the component-wise absolute value of a 3-dimensional vector
inline Vector3 v3abs(Vector3& v)
{
    return Vector3(fabs(v.coordinate.x), fabs(v.coordinate.y), fabs(v.coordinate.z));
}

inline Vector3 to_degrees(const Vector3& vec) { return Vector3(to_degrees(vec.coordinate.x), to_degrees(vec.coordinate.y), to_degrees(vec.coordinate.z)); }

inline Vector3 to_radians(const Vector3& vec) { return Vector3(to_radians(vec.coordinate.x), to_radians(vec.coordinate.y), to_radians(vec.coordinate.z)); }
}

#endif // _TEMPEST_VECTOR3_HH_
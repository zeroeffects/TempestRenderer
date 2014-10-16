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

namespace Tempest
{
//! 4-dimensional vector
/*!
    \ingroup TempestMath
*/
struct Vector4
{
    struct coord
    {
        float x, /*!< x-coordinate component */
              y, /*!< y-coordinate component */
              z, /*!< z-coordinate component */
              w; /*!< w-coordinate component */
    };
    struct comb0
    {
        Vector2 xy, /*!< xy-component subvector */
                zw; /*!< zw-component subvector */
    };
    struct comb1
    {
        float   x;  /*!< x-coordinate component */
        Vector2 yz; /*!< yz-component subvector */
        float   w;  /*!< w-coordinate component */
    };
    struct comb2
    {
        Vector3 xyz; /*!< xyz-component subvector */
        float   w;   /*!< w-coordinate component subvector */
    };
    struct comb3
    {
        float   x;   /*!< x-coordinate component */
        Vector3 yzw; /*!< yzw-component subvector */
    };

    union
    {
        coord coordinate;
        comb0 combined0;
        comb1 combined1;
        comb2 combined2;
        comb3 combined3;
        float elem[4]; /*!< coordinate array */
    };

    //! Default constructor
    explicit Vector4() { static_assert(sizeof(Vector4) == 4*sizeof(float), "Vector4 has the wrong size"); }

    Vector4(const Vector4& vec)
    {
        set(vec.coordinate.x, vec.coordinate.y, vec.coordinate.z, vec.coordinate.w);
    }
    
    //! Conversion constructor
    explicit Vector4(const Vector3& vec)
    {
        set(vec.coordinate.x, vec.coordinate.y, vec.coordinate.z, 1.0f);
    }

    //! Constructor
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
        \param _z z-coordinate component
        \param _w w-coordinate component
    */
    Vector4(float _x, float _y, float _z, float _w)
    {
        set(_x, _y, _z, _w);
    }

    inline float x() const { return coordinate.x; }
    inline float y() const { return coordinate.y; }
    inline float z() const { return coordinate.z; }
    inline float w() const { return coordinate.w; }
    inline float r() const { return coordinate.x; }
    inline float g() const { return coordinate.y; }
    inline float b() const { return coordinate.z; }
    inline float a() const { return coordinate.w; }
    inline float s() const { return coordinate.x; }
    inline float t() const { return coordinate.y; }
    inline float p() const { return coordinate.z; }
    inline float q() const { return coordinate.w; }

    inline Vector2 xy() const { return combined0.xy; }
    inline Vector2 yz() const { return combined1.yz; }
    inline Vector2 zw() const { return combined0.zw; }
    inline Vector3 xyz() const { return combined2.xyz; }
    inline Vector3 yzw() const { return combined3.yzw; }
    inline Vector2 rg() const { return combined0.xy; }
    inline Vector2 gb() const { return combined1.yz; }
    inline Vector2 ba() const { return combined0.zw; }
    inline Vector3 rgb() const { return combined2.xyz; }
    inline Vector3 gba() const { return combined3.yzw; }
    inline Vector2 st() const { return combined0.xy; }
    inline Vector2 tp() const { return combined1.yz; }
    inline Vector2 pq() const { return combined0.zw; }
    inline Vector3 stp() const { return combined2.xyz; }
    inline Vector3 tpq() const { return combined3.yzw; }

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float& operator[](size_t i) 
    {
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element index");
        return elem[i];
    }

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float operator[](size_t i) const
    {
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element index");
        return elem[i];
    }

    //! Computes the length of the vector
    /*! This function uses the following equation:
        \f[\|\vec{v}\| = \frac{\sqrt{v_{x}^{2} + v_{y}^{2} + v_{z}^{2}}}{v_w}\f]
        \return the length of the vector as a floating-point number
        \warning the function doesn't check for \f$v_w \neq 0\f$
    */
    float length() const { return sqrt(coordinate.x*coordinate.x+coordinate.y*coordinate.y+coordinate.z*coordinate.z)/coordinate.w; }

    //! Partially normalizes the component of the vector
    /*! The function divides all of the coordinate component by:
        \f[\|\vec{v_{xyz}}\| = \sqrt{v_{x}^{2} + v_{y}^{2} + v_{z}^{2}}\f]
        \warning \f$\vec{v_w} = \frac{\vec{v_w}}{|\vec{v_{xyz}}\|}\f$
    */
    void normalizePartial()
    {
        float l = sqrt(coordinate.x*coordinate.x+coordinate.y*coordinate.y+coordinate.z*coordinate.z);
        if(l != 0.0f)
        {
            coordinate.x /= l;
            coordinate.y /= l;
            coordinate.z /= l;
            coordinate.w /= l;
        }
    }

    //! Computes the dot product between two vectors
    /*! The formula used for this function is as follows: 
        \f[a_x b_x + a_y b_y + a_z b_z + a_w b_w\f]
        \param vec a 4-dimensional vector
        \returns the dot product between the two vectors
    */
    float dot(const Vector4& vec) const { return coordinate.x*vec.coordinate.x + coordinate.y*vec.coordinate.y + coordinate.z*vec.coordinate.z + coordinate.w*vec.coordinate.w; }

    //! Computes the dot product between two vectors
    /*! The formula used for this function is as follows: 
        \f[a_x b_x + a_y b_y + a_z b_z + a_w\f]
        Where \f$\vec{a}\f$ is this vector.
        \param vec a 3-dimensional vector
        \returns the dot product between the two vectors
    */
    float dot(const Vector3& vec) const { return coordinate.x*vec.coordinate.x + coordinate.y*vec.coordinate.y + coordinate.z*vec.coordinate.z + coordinate.w; }

    //! Sets the values of the coordinate component
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
        \param _z z-coordinate component
        \param _w w-coordinate component
    */
    inline void set(float _x, float _y, float _z, float _w = 1.0f) { coordinate.x = _x; coordinate.y = _y; coordinate.z = _z; coordinate.w = _w; }
};

inline bool operator==(const Vector4& lhs, const Vector4& rhs) { return approx_eq(lhs.coordinate.x, rhs.coordinate.x) && approx_eq(lhs.coordinate.y, rhs.coordinate.y) && approx_eq(lhs.coordinate.z, rhs.coordinate.z) && approx_eq(lhs.coordinate.w, rhs.coordinate.w); }

inline bool operator!=(const Vector4& lhs, const Vector4& rhs) { return approx_neq(lhs.coordinate.x, rhs.coordinate.x) || approx_neq(lhs.coordinate.y, rhs.coordinate.y) || approx_neq(lhs.coordinate.z, rhs.coordinate.z) || approx_neq(lhs.coordinate.w, rhs.coordinate.w); }

//! Sums two vectors and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator+(const Vector4& lhs, const Vector4& rhs) { return Vector4(lhs.coordinate.x+rhs.coordinate.x, lhs.coordinate.y+rhs.coordinate.y, lhs.coordinate.z+rhs.coordinate.z, lhs.coordinate.w+rhs.coordinate.w); }

//! Subtracts two vectors and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator-(const Vector4& lhs, const Vector4& rhs) { return Vector4(lhs.coordinate.x-rhs.coordinate.x, lhs.coordinate.y-rhs.coordinate.y, lhs.coordinate.z-rhs.coordinate.z, lhs.coordinate.w-rhs.coordinate.w); }

//! Negates a vector and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator-(const Vector4& vec) { return Vector4(-vec.coordinate.x, -vec.coordinate.y, -vec.coordinate.z, -vec.coordinate.w); }

//! Sums two vectors and stores the result in the first vector
/*! \related Vector4 */
inline Vector4& operator+=(Vector4& lhs, const Vector4& rhs) { lhs.coordinate.x += rhs.coordinate.x; lhs.coordinate.y += rhs.coordinate.y; lhs.coordinate.z += rhs.coordinate.z; lhs.coordinate.w += rhs.coordinate.w; return lhs; }

//! Subtracts two vectors and stores the result in the first vector
/*! \related Vector4 */
inline Vector4& operator-=(Vector4& lhs, const Vector4& rhs) { lhs.coordinate.x -= rhs.coordinate.x; lhs.coordinate.y -= rhs.coordinate.y; lhs.coordinate.z -= rhs.coordinate.z; lhs.coordinate.w -= rhs.coordinate.w; return lhs; }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator*(const Vector4& vec, float a) { return Vector4(vec.coordinate.x * a, vec.coordinate.y * a, vec.coordinate.z * a, vec.coordinate.w * a); }

//! Multiplies a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator*(float a, const Vector4& vec) { return Vector4(vec.coordinate.x * a, vec.coordinate.y * a, vec.coordinate.z * a, vec.coordinate.w * a); }

//! Multiplies a vector with a float-pointing variable and replaces the vector
/*! \related Vector4 */
inline Vector4& operator*=(Vector4& vec, float a) { vec.coordinate.x *= a; vec.coordinate.y *= a; vec.coordinate.z *= a, vec.coordinate.w *= a; return vec; }

//! Divides a vector with a float-pointing variable and returns the resulting vector
/*! \related Vector4 */
inline Vector4 operator/(const Vector4& vec, float a) { return Vector4(vec.coordinate.x / a, vec.coordinate.y / a, vec.coordinate.z / a, vec.coordinate.w / a); }

//! Divides a vector with a float-pointing variable and replaces the vector
/*! \related Vector4 */
inline Vector4& operator/=(Vector4& vec, float a) { vec.coordinate.x /= a; vec.coordinate.y /= a; vec.coordinate.z /= a; vec.coordinate.w /= a; return vec; }
}

#endif // _TEMPEST_VECTOR4_HH_
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

namespace Tempest
{
class Matrix4;

//! Quaternion representation of spatial rotation
/*! \ingroup CarinaMath
    Quaternions are used for representing spatial rotation. In the case of dual quaternions
    and some other mathematical structures they could be used to represent other types of
    linear transfromations, e.g. translation.
*/
struct Quaternion
{
    struct coord
    {
        float x, /*!< x-coordinate component */
              y, /*!< y-coordinate component */
              z, /*!< z-coordinate component */
              w; /*!< w-coordinate component */
    };
    struct comp
    {
        Vector3 vector; /*!< the vector part of the quaternion */
        float scalar;   /*!< the scalar part of the quaternion */
    };
    union
    {
        coord coordinate;
#ifdef _CXX11_SUPPORT
        comp component;
#endif
        float elem[4]; /*!< coordinate array */
    };

    //! Default constructor
    Quaternion() {}

    //! Constructor
    /*!
        \param _x x-coordinate component
        \param _y y-coordinate component
        \param _z z-coordinate component
        \param _w w-coordinate component
    */
    Quaternion(float _x, float _y, float _z, float _w)
    {
        coordinate.x = _x;
        coordinate.y = _y;
        coordinate.z = _z;
        coordinate.w = _w;
    }

    //! Conversion constructor
    /*! Converts the rotation part of a 4x4 homogeneous matrix to quaternion */
    explicit Quaternion(const Matrix4& matrix);

    //! Copy constructor
    Quaternion(const Quaternion& quat);

    //! Assignment operator
    Quaternion& operator=(const Quaternion& quat);

    inline float& x() { return coordinate.x; }
    inline float& y() { return coordinate.y; }
    inline float& z() { return coordinate.z; }
    inline float& w() { return coordinate.w; }
#ifdef _CXX11_SUPPORT
    inline Vector3& vector() { return component.vector; }
    inline float& scalar() { return component.scalar; }
#else
    inline Vector3& vector() { return reinterpret_cast<comp*>(this)->vector; }
    inline float& scalar() { return reinterpret_cast<comp*>(this)->scalar; }
#endif

    inline float x() const { return coordinate.x; }
    inline float y() const { return coordinate.y; }
    inline float z() const { return coordinate.z; }
    inline float w() const { return coordinate.w; }
#ifdef _CXX11_SUPPORT
    inline Vector3 vector() const { return component.vector; }
    inline float scalar() const { return component.scalar; }
#else
    inline Vector3 vector() const { return reinterpret_cast<const comp*>(this)->vector; }
    inline float scalar() const { return reinterpret_cast<const comp*>(this)->scalar; }
#endif

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float& operator[](size_t i)
    {
    #ifndef CARINA_UNSAFE
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element coordex");
    #endif
        return elem[i];
    }

    //! Array style coordinate component referencing
    /*!
        \param i the coordex of the coordinate component
        \return reference to a coordinate component
    */
    inline float operator[](size_t i) const
    {
    #ifndef CARINA_UNSAFE
        TGE_ASSERT(i < TGE_FIXED_ARRAY_SIZE(elem), "Bad element coordex");
    #endif
        return elem[i];
    }

    //! Sets the quaternion to the multiplication identity quaternion, i.e. q = 1
    void identity();

    //! Returns the conjugate quaternion
    Quaternion conjugate() const;

    //! Conjugates this quaternion
    void conjugateSelf();

    //! Returns the inverse quaternion
    /*!
        \remarks It is the same as returning the conjugate
    */
    Quaternion inverse() const;
    
    //! Inverts the quaternion
    /*!
        \remarks It is the same as conjugating the quaternion
    */
    void invertSelf();

    //! Computes the length of the quaternion as a 4D vector
    float length() const;

    //! Normalizes the quaternion
    void normalize();

    //! Computes the dot product between two quaternion
    float dot(const Quaternion& quat) const;

    //! Rotates the coordinate system around the relative x-axis
    /*!
        \param yaw rotation around the x-axis
    */
    void rotateX(float pitch);

    //! Rotates the coordinate system around the relative y-axis
    /*!
        \param yaw rotation around the y-axis
    */
    void rotateY(float yaw);

    //! Rotates the coordinate system around the relative z-axis
    /*!
        \param yaw rotation around the z-axis
    */
    void rotateZ(float roll);

    //! Rotates the coordinate system
    /*!
        \param angle the angle of rotation
        \param axis the relative axis of rotation
    */
    void rotate(float angle, const Vector3& axis);

    //! Transforms a 3-dimensional vector
    Vector3 transform(const Vector3& v) const;
};

//! Multiplies two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
Quaternion& operator*=(Quaternion& lhs, const Quaternion& rhs);

//! Sums two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
Quaternion& operator+=(Quaternion& lhs, const Quaternion& rhs);

//! Subtracts two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
Quaternion& operator-=(Quaternion& lhs, const Quaternion& rhs);

//! Divides two quaternions and stores the result in the first quaternion
/*! \related Quaternion */
Quaternion& operator/=(Quaternion& lhs, const Quaternion& rhs);

//! Multiplies a quaternion with a floating-point variable and stores the result in the quaternion
/*! \related Quaternion */
Quaternion& operator*=(Quaternion& quat, float f);

//! Divides a quaternion with a floating-point variable and stores the result in the quaternion
/*! \related Quaternion */
Quaternion& operator/=(Quaternion& quat, float f);

//! Compares two quaternions
/*! \related Quaternion */
bool operator==(const Quaternion& lhs, const Quaternion& rhs);

//! Divides two quaternions and returns the result
/*! \related Quaternion */
Quaternion operator/(const Quaternion& lhs, const Quaternion& rhs);

//! Multiplies two quaternions and returns the result
/*! \related Quaternion */
Quaternion operator*(const Quaternion& lhs, const Quaternion& rhs);

//! Sums two quaternions and returns the result
/*! \related Quaternion */
Quaternion operator+(const Quaternion& lhs, const Quaternion& rhs);

//! Subtracts two quaternions and returns the result
/*! \related Quaternion */
Quaternion operator-(const Quaternion& lhs, const Quaternion& rhs);

//! Multiplies a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
Quaternion operator*(float f, const Quaternion& quat);

//! Multiplies a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
Quaternion operator*(const Quaternion& quat, float f);

//! Divides a quaternion with a floating-point variable and returns the result
/*! \related Quaternion */
Quaternion operator/(const Quaternion& quat, float f);

//! Negates a quaternion
/*! \related Quaternion */
Quaternion operator-(const Quaternion& quat);
}

#endif // _TEMPEST_QUATERNION_HH_
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

#ifndef _TEMPEST_DUAL_QUATERNION_HH_
#define _TEMPEST_DUAL_QUATERNION_HH_

#include "tempest/math/quaternion.hh"

namespace Tempest
{
//! Dual quaternion representation of spatial translation and rotation
/*! \ingroup CarinaMath
*/
struct DualQuaternion
{
    Quaternion  non_dual, /*!< the non-dual part */
                dual;     /*!< the dual part */

    //! Default constructor
    DualQuaternion();

    //! Constructor
    /*!
        \param ndp the non-dual part
        \param dp the dual part
    */
    DualQuaternion(const Quaternion& ndp, const Quaternion& dp);

    //! Copy constructor
    DualQuaternion(const DualQuaternion& d);

    //! Assignment operator
    DualQuaternion& operator=(const DualQuaternion& d);

    //! Conversion constructor
    /*! Convert the rotation and translation part of a 4x4 homogeneous matrix to dual quaternion
        \param mat a 4x4 homogeneous matrix
    */
    explicit DualQuaternion(const Matrix4& mat);

    //! Sets the dual quaternion to identity, i.e. q = 1
    void identity();

    //! Returns the dual conjugate dual quaternion
    DualQuaternion conjugate() const;
    
    //! Dual conjugates the dual quaternion
    void conjugateSelf();

    //! Returns the quaternion conjugate dual quaternion
    DualQuaternion conjugateQuaternion() const;

    //! Quaternion conjugates the quaternion
    void conjugateQuaternionSelf();

    //! Returns the inverse dual quaternion
    /*!
        \remarks It is the same as returning the quaternion conjugate
    */
    DualQuaternion inverse() const;

    //! Inverts the dual quaternion
    /*!
        \remarks It is the same as quaternion conjugating the dual quaternion
    */
    void invertSelf();

    //! Returns the length of the non-dual part
    float length() const;

    //! Normalizes the dual quaternion
    void normalize();

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

    //! Translates the coordinate system
    /*!
        \param vec a 3-dimensional vector representing the relative translation
    */
    void translate(const Vector3& vec);

    //! Translates the coordinate system by the relative x-axis
    /*!
        \param x a floating-point variable representing the relative translation by the x-axis
    */
    void translateX(float x);

    //! Translates the coordinate system by the relative y-axis
    /*!
        \param y a floating-point variable representing the relative translation by the y-axis
    */
    void translateY(float y);

    //! Translates the coordinate system by the relative z-axis
    /*!
        \param z a floating-point variable representing the relative translation by the z-axis
    */
    void translateZ(float z);

    //! Transforms a 3-dimensional vector
    Vector3 transform(const Vector3& v) const;
};

//! Multiplies two dual quaternions and stores the result in the first dual quaternion
/*! \related DualQuaternion */
DualQuaternion& operator*=(DualQuaternion& lhs, const DualQuaternion& rhs);

//! Sums component-wise two dual quaternions and stores the result in the first dual quaternion
/*! \related DualQuaternion */
DualQuaternion& operator+=(DualQuaternion& lhs, const DualQuaternion& rhs);

//! Subtracts component-wise two dual quaternions and stores the result in the first dual quaternion
/*! \related DualQuaternion */
DualQuaternion& operator-=(DualQuaternion& lhs, const DualQuaternion& rhs);

//! Multiplies a dual quaternion with a floating-point variable and stores the result in the dual quaternion
/*! \related DualQuaternion */
DualQuaternion& operator*=(DualQuaternion& d, float f);

//! Compares two dual quaternions
/*! \related DualQuaternion */
bool operator==(const DualQuaternion& lhs, const DualQuaternion& rhs);

//! Multiplies two dual quaternions and returns the result
/*! \related DualQuaternion */
DualQuaternion operator*(const DualQuaternion& lhs, const DualQuaternion& rhs);

//! Sums component-wise two dual quaternion and returns the result
/*! \related DualQuaternion */
DualQuaternion operator+(const DualQuaternion& lhs, const DualQuaternion& rhs);

//! Subtracts component-wise two dual quaternion and returns the result
/*! \related DualQuaternion */
DualQuaternion operator-(const DualQuaternion& lhs, const DualQuaternion& rhs);

//! Multiplies a dual quaternion with a floating-point variable and returns the result
/*! \related DualQuaternion */
DualQuaternion operator*(const DualQuaternion& d, float f);

//! Multiplies a dual quaternion with a floating-point variable and returns the result
/*! \related DualQuaternion */
DualQuaternion operator*(float f, const DualQuaternion& d);

//! Negates component-wise a dual quaternion and returns the result
/*! \related DualQuaternion */
DualQuaternion operator-(const DualQuaternion& q);

//! Calculates the linear interpolation between two dual quaternion
/*! This function uses the following equation: \f$(1-t)*q_1 + t*q_2\f$
    \param q1 a dual quaternion
    \param q1 a dual quaternion
    \param t the distance between the two quaternions (\f$t \in [0, 1]\f$)
    \related DualQuaternion 
*/
DualQuaternion interpolate(float t, const DualQuaternion& q1, const DualQuaternion& q2);
}

#endif // _TEMPEST_DUAL_QUATERNION_HH_
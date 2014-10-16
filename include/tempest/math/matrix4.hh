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

#ifndef _TEMPEST_MATRIX_HH_
#define _TEMPEST_MATRIX_HH_

#include "tempest/utils/config.hh"
#include "tempest/utils/assert.hh"

#if defined(HAS_SSE) || defined(HAS_ARM_NEON)
#   include <xmmintrin.h>
#   ifdef _MSC_VER
#       define SIMD128_ALIGNED __declspec(align(16))
#   else
#       define SIMD128_ALIGNED __attribute__((aligned(16)))
#   endif
#else
#   define SIMD128_ALIGNED
#endif

namespace Tempest
{
class Vector2;
class Vector3;
class Vector4;
class Quaternion;
class DualQuaternion;

/*! \defgroup sse_ext SIMD Convenience functions
    @{
*/

#ifdef HAS_SSE
//! Selects only the first value of p for the four floating-point values
#define SHUFFLE_0(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 0, 0, 0))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_0(p) p[0]
#endif

#ifdef HAS_SSE
//! Selects only the second value of p for the four floating-point values
#define SHUFFLE_1(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(1, 1, 1, 1))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_1(p) p[1]
#endif

#ifdef HAS_SSE
//! Selects only the third value of p for the four floating-point values
#define SHUFFLE_2(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 2, 2))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_2(p) p[2]
#endif

#ifdef HAS_SSE
//! Selects only the fourth value of p for the four floating-point values
#define SHUFFLE_3(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 3, 3, 3))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_3(p) p[3]
#endif

union SIMD128_ALIGNED simd128
{
#ifdef HAS_SSE
    typedef __m128 simd_type;
    simd_type   m128;
#elif defined(HAS_ARM_NEON)
    typedef float32x4_t simd_type;
    simd_type   m128;
#endif
    float       m128_f32[4];

    simd128() {}
    
    simd128(float f0, float f1, float f2, float f3)
#ifdef HAS_SSE
        :   m128(_mm_setr_ps(f0, f1, f2, f3)) {}
#else
        { m128_f32[0] = f0; m128_f32[1] = f1; m128_f32[2] = f2; m128_f32[3] = f3; }
#endif

#if defined(HAS_SSE) || defined(HAS_ARM_NEON)
    simd128(const simd_type& m)
        :   m128(m) {}

    simd128& operator=(const simd_type& m) { m128 = m; return *this; }

    inline operator simd_type() const { return m128; }
#endif
    
    float& operator[](size_t idx) { return m128_f32[idx]; }
    float operator[](size_t idx) const { return m128_f32[idx]; }
};

#ifdef HAS_SSE
//! Multiplies the four floating-point values by a single floating-point value. (SSE composite)
inline simd128 operator*(const simd128& lhs, float rhs) { return _mm_mul_ps(lhs, _mm_set_ps1(rhs)); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator*(const simd128& lhs, float rhs) { return vmulq_n_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Multiplies the four floating-point values of lhs and rhs
inline simd128 operator*(const simd128& lhs, const simd128& rhs) { return _mm_mul_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator*(const simd128& lhs, const simd128& rhs) { return vmulq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Sums the four floating-point values of lhs and rhs
inline simd128 operator+(const simd128& lhs, const simd128& rhs) { return _mm_add_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator+(const simd128& lhs, const simd128& rhs) { return vaddq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Subtracts the four floating-point values of lhs and rhs
inline simd128 operator-(const simd128& lhs, const simd128& rhs) { return _mm_sub_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator-(const simd128& lhs, const simd128& rhs) { return vsubq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Divides the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128 operator/(const simd128& lhs, const simd128& rhs) { return _mm_div_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator/(const simd128& lhs, const simd128& rhs) { return vmulq_f32(lhs, vrecpeq_f32(rhs)); }
#endif

#ifdef HAS_SSE
//! Multiplies in-place the four floating-point values by a single floating-point value. (SSE composite)
inline simd128& operator*=(simd128& lhs, float rhs) { return lhs = _mm_mul_ps(lhs, _mm_set_ps1(rhs)); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator*=(simd128& lhs, float rhs) { return lhs = vmulq_n_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Multiplies in-place the four floating-point values of lhs and rhs
inline simd128& operator*=(simd128& lhs, const simd128& rhs) { return lhs = _mm_mul_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator*=(simd128& lhs, const simd128& rhs) { return lhs = vmulq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Sums in-place the four floating-point values of lhs and rhs
inline simd128& operator+=(simd128& lhs, const simd128& rhs) { return lhs = _mm_add_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator+=(simd128& lhs, const simd128& rhs) { return lhs = vaddq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Subtracts in-place the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128& operator-=(simd128& lhs, const simd128& rhs) { return lhs = _mm_sub_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator-=(simd128& lhs, const simd128& rhs) { return lhs = vsubq_f32(lhs, rhs); }
#endif

#ifdef HAS_SSE
//! Divides in-place the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128& operator/=(simd128& lhs, const simd128& rhs) { return lhs = _mm_div_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator/=(simd128& lhs, const simd128& rhs) { return lhs = vmulq_f32(lhs, vrecpeq_f32(rhs)); }
#endif

#ifdef HAS_SSE
inline void i_mad(simd128& res, const simd128& op_a, const simd128& op_b) { res += op_a * op_b; }
#elif defined(HAS_ARM_NEON)
inline void i_mad(simd128& res, const simd128& op_a, const simd128& op_b) { vmlaq_f32(res, op_a, op_b); }
#endif

/*! @} */

//! 4x4 homogeneous column major matrix
/*!
    \ingroup CarinaMath
*/
class SIMD128_ALIGNED Matrix4
{
    simd128 matrix[4];      /*!< the matrix data as a SIMD array */
public:
    //! Default constructor
    Matrix4() { static_assert(sizeof(Matrix4) == 16*sizeof(float), "Matrix4 has the wrong size"); }

    //! Constructor
    Matrix4(float* _mat);

    //! Constructor
    /*!
        \warning The constructor treats the data as column-major.
    */
    Matrix4(float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33);

    //! Conversion constructor
    /*! Sets the value of the matrix to the linear transformation
        represented by the rotation quaternion and the translation
        vector
        \param rotation a rotation quaternion
        \param translation a translation vector
    */
    Matrix4(const Quaternion& rotation, const Vector3& translation);
    
    //! Conversion constructor
    explicit Matrix4(const DualQuaternion& dq);

    /*! Sets the value of the matrix to the linear transformation
        represented by the rotation quaternion and the translation
        vector
    */
    void set(const Quaternion& rotation, const Vector3& translation);

    //! Assignment operator
    Matrix4& operator=(const Matrix4& mat);

    //! Multiplies this matrix by another (in-place)
    Matrix4& operator*=(const Matrix4& mat);

    //! Sums component wise this matrix with another (in-place)
    Matrix4& operator+=(const Matrix4& mat);

    //! Subtracts component wise this matrix with another (in-place)
    Matrix4& operator-=(const Matrix4& mat);

    Matrix4& operator/=(float f);
    
    //! Compares this matrix with another
    bool operator==(const Matrix4& mat) const;

    //! Gets the \f$a_{ji}\f$ element of the matrix
    float& get(unsigned i, unsigned j) { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    float get(unsigned i, unsigned j) const { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    float& operator()(unsigned i, unsigned j) { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    float operator()(unsigned i, unsigned j) const { return matrix->m128_f32[i*4+j]; }

    //! Get the \f$a_i\f$ column of the matrix
    float* operator[](unsigned i)
    {
        TGE_ASSERT(i < 4, "Bad element coordex");
        return matrix[i].m128_f32;
    }

    //! Get the \f$a_i\f$ column of the matrix
    const float* operator[](unsigned i) const
    {
        TGE_ASSERT(i < 4, "Bad element coordex");
        return matrix[i].m128_f32;
    }

    //! Subtracts component wise this matrix with another
    Matrix4 operator-(const Matrix4& mat) const;

    //! Sums component wise this matrix with another
    Matrix4 operator+(const Matrix4& mat) const;

    Matrix4 operator/(float f) const;
    
    //! Multiplies this matrix with another
    Matrix4 operator*(const Matrix4& mat2) const;

    //! Transforms a 3-dimensional vector
    Vector3 operator*(const Vector3& vec) const;

    //! Transforms a 4-dimensional vector
    Vector4 operator*(const Vector4& vec) const;

    //! Returns the transposed matrix
    Matrix4 transpose() const;

    //! Transposes this matrix
    void transposeSelf();

    //! Calculates the inverse matrix
    Matrix4 inverse() const;

    //! Inverts this matrix
    void invertSelf();

    //! Sets this matrix to identity matrix
    void identity();

    //! Scales the coordinate system
    /*!
        \param vec a 3-dimensional vector representing the scaling
    */
    void scale(const Vector3& vec);
    
    //! Scales the coordinate system
    /*!
        \param s scaling factor
    */
    void scale(float s);
    
    //! Rotates the coordinate system around the relative x-axis
    /*!
        \param pitch rotation around the x-axis
    */
    void rotateX(float pitch);
    
    //! Rotates the coordinate system around the relative y-axis
    /*!
        \param yaw rotation around the y-axis
    */
    void rotateY(float yaw);

    //! Rotates the coordinate system around the relative z-axis
    /*!
        \param roll rotation around the z-axis
    */
    void rotateZ(float roll);

    //! Rotates the coordinate system
    /*!
        \param angle the angle of rotation
        \param axis the relative axis of rotation
    */
    void rotate(float angle, const Vector3& axis);

    //! Rotates the coordinate system
    /*!
        \param euler a 3-dimensional vector representing the euler angles
    */
    void rotate(const Vector3& euler);

    //! Translates the coordinate system
    /*!
        \param vec a 3-dimensional vector representing the relative translation
    */
    void translate(const Vector3& vec);
    
    //! Translates the coordinate system
    /*!
        \param vec a 2-dimensional vector representing the relative translation
    */
    void translate(const Vector2& vec);

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

    //! Rotates a 3-dimensional vector
    Vector3 transform_rotate(const Vector3& v) const;

    Vector3 relativeX() const;
    Vector3 relativeY() const;
    Vector3 relativeZ() const;

    Vector3 translation() const;
    Vector3 scaling() const;
    void decompose(Vector3& translation, Vector3& scaling, Vector3& euler);
};
}

#endif // _TEMPEST_MATRIX_HH_
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
#include "tempest/math/functions.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/matrix3.hh"
#include <cstring>

#ifndef EXPORT_CUDA
#	ifdef __CUDACC__
#		define EXPORT_CUDA __device__ __host__
#	else
#		define EXPORT_CUDA
#	endif
#endif

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
union Vector2;
union Quaternion;
struct DualQuaternion;

/*! \defgroup sse_ext SIMD Convenience functions
    @{
*/

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Selects only the first value of p for the four floating-point values
#define SHUFFLE_0(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(0, 0, 0, 0))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_0(p) p[0]
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Selects only the second value of p for the four floating-point values
#define SHUFFLE_1(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(1, 1, 1, 1))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_1(p) p[1]
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Selects only the third value of p for the four floating-point values
#define SHUFFLE_2(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(2, 2, 2, 2))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_2(p) p[2]
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Selects only the fourth value of p for the four floating-point values
#define SHUFFLE_3(p) _mm_shuffle_ps(p, p, _MM_SHUFFLE(3, 3, 3, 3))
#elif defined(HAS_ARM_NEON)
#define SHUFFLE_3(p) p[3]
#endif

union SIMD128_ALIGNED simd128
{
#if defined(HAS_SSE) && !defined(__CUDACC__)
    typedef __m128 simd_type;
    simd_type   m128;
#elif defined(HAS_ARM_NEON)
    typedef float32x4_t simd_type;
    simd_type   m128;
#endif
    float       m128_f32[4];

    EXPORT_CUDA simd128() {}
    
    EXPORT_CUDA simd128(float f0, float f1, float f2, float f3)
#if defined(HAS_SSE) && !defined(__CUDACC__)
        :   m128(_mm_setr_ps(f0, f1, f2, f3)) {}
#else
        { m128_f32[0] = f0; m128_f32[1] = f1; m128_f32[2] = f2; m128_f32[3] = f3; }
#endif

#if (defined(HAS_SSE) || defined(HAS_ARM_NEON)) && !defined(__CUDACC__)
    EXPORT_CUDA  simd128(const simd_type& m)
        :   m128(m) {}

    EXPORT_CUDA simd128& operator=(const simd_type& m) { m128 = m; return *this; }

    inline EXPORT_CUDA operator simd_type() const { return m128; }
#endif
    
    EXPORT_CUDA float& operator[](size_t idx) { return m128_f32[idx]; }
    EXPORT_CUDA float operator[](size_t idx) const { return m128_f32[idx]; }
};

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Multiplies the four floating-point values by a single floating-point value. (SSE composite)
inline simd128 operator*(const simd128& lhs, float rhs) { return _mm_mul_ps(lhs, _mm_set_ps1(rhs)); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator*(const simd128& lhs, float rhs) { return vmulq_n_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Multiplies the four floating-point values of lhs and rhs
inline simd128 operator*(const simd128& lhs, const simd128& rhs) { return _mm_mul_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator*(const simd128& lhs, const simd128& rhs) { return vmulq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Sums the four floating-point values of lhs and rhs
inline simd128 operator+(const simd128& lhs, const simd128& rhs) { return _mm_add_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator+(const simd128& lhs, const simd128& rhs) { return vaddq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Subtracts the four floating-point values of lhs and rhs
inline simd128 operator-(const simd128& lhs, const simd128& rhs) { return _mm_sub_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator-(const simd128& lhs, const simd128& rhs) { return vsubq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Divides the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128 operator/(const simd128& lhs, const simd128& rhs) { return _mm_div_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128 operator/(const simd128& lhs, const simd128& rhs) { return vmulq_f32(lhs, vrecpeq_f32(rhs)); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Multiplies in-place the four floating-point values by a single floating-point value. (SSE composite)
inline simd128& operator*=(simd128& lhs, float rhs) { return lhs = _mm_mul_ps(lhs, _mm_set_ps1(rhs)); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator*=(simd128& lhs, float rhs) { return lhs = vmulq_n_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Multiplies in-place the four floating-point values of lhs and rhs
inline simd128& operator*=(simd128& lhs, const simd128& rhs) { return lhs = _mm_mul_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator*=(simd128& lhs, const simd128& rhs) { return lhs = vmulq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Sums in-place the four floating-point values of lhs and rhs
inline simd128& operator+=(simd128& lhs, const simd128& rhs) { return lhs = _mm_add_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator+=(simd128& lhs, const simd128& rhs) { return lhs = vaddq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Subtracts in-place the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128& operator-=(simd128& lhs, const simd128& rhs) { return lhs = _mm_sub_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator-=(simd128& lhs, const simd128& rhs) { return lhs = vsubq_f32(lhs, rhs); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
//! Divides in-place the four floating-point values of lhs and rhs. (ARM Neon composite)
inline simd128& operator/=(simd128& lhs, const simd128& rhs) { return lhs = _mm_div_ps(lhs, rhs); }
#elif defined(HAS_ARM_NEON)
inline simd128& operator/=(simd128& lhs, const simd128& rhs) { return lhs = vmulq_f32(lhs, vrecpeq_f32(rhs)); }
#endif

#if defined(HAS_SSE) && !defined(__CUDACC__)
inline void i_mad(simd128& res, const simd128& op_a, const simd128& op_b) { res += op_a * op_b; }
#elif defined(HAS_ARM_NEON)
inline void i_mad(simd128& res, const simd128& op_a, const simd128& op_b) { vmlaq_f32(res, op_a, op_b); }
#endif

/*! @} */

struct Matrix3;

//! 4x4 homogeneous column major matrix
/*!
    \ingroup CarinaMath
*/
class SIMD128_ALIGNED Matrix4
{
    simd128 matrix[4];      /*!< the matrix data as a SIMD array */
public:
    //! Default constructor
    EXPORT_CUDA explicit Matrix4() { static_assert(sizeof(Matrix4) == 16*sizeof(float), "Matrix4 has the wrong size"); }

    //! Constructor
    EXPORT_CUDA explicit Matrix4(float* _mat);

    EXPORT_CUDA explicit Matrix4(const Vector3& c0,
                                 const Vector3& c1,
                                 const Vector3& c2)
    {
        matrix[0] = simd128(c0.x, c0.y, c0.z, 0.0f);
        matrix[1] = simd128(c1.x, c1.y, c1.z, 0.0f);
        matrix[2] = simd128(c2.x, c2.y, c2.z, 0.0f);
        matrix[3] = simd128(0.0f, 0.0f, 0.0f, 1.0f);
    }

    EXPORT_CUDA explicit Matrix4(const Matrix3& mat)
    {
        auto& c0 = mat.column(0),
            & c1 = mat.column(1),
            & c2 = mat.column(2);
        matrix[0] = simd128(c0.x, c0.y, c0.z, 0.0f);
        matrix[1] = simd128(c1.x, c1.y, c1.z, 0.0f);
        matrix[2] = simd128(c2.x, c2.y, c2.z, 0.0f);
        matrix[3] = simd128(0.0f, 0.0f, 0.0f, 1.0f);
    }

    //! Constructor
    /*!
        \warning Automatic transpose is applied.
    */
    EXPORT_CUDA explicit Matrix4(float m00, float m10, float m20, float m30,
                                 float m01, float m11, float m21, float m31,
                                 float m02, float m12, float m22, float m32,
                                 float m03, float m13, float m23, float m33);

    //! Conversion constructor
    /*! Sets the value of the matrix to the linear transformation
        represented by the rotation quaternion and the translation
        vector
        \param rotation a rotation quaternion
        \param translation a translation vector
    */
    explicit Matrix4(const Quaternion& rotation, const Vector3& translation);
    
    //! Conversion constructor
    explicit Matrix4(const DualQuaternion& dq);

    /*! Sets the value of the matrix to the linear transformation
        represented by the rotation quaternion and the translation
        vector
    */
    void set(const Quaternion& rotation, const Vector3& translation);

    //! Assignment operator
    EXPORT_CUDA Matrix4& operator=(const Matrix4& mat)
    {
        memcpy(matrix, mat.matrix, sizeof(matrix));
        return *this;
    }

    //! Multiplies this matrix by another (in-place)
    EXPORT_CUDA Matrix4& operator*=(const Matrix4& mat)
    {
        *this = *this * mat;
        return *this;
    }

    //! Sums component wise this matrix with another (in-place)
    EXPORT_CUDA Matrix4& operator+=(const Matrix4& mat);

    //! Subtracts component wise this matrix with another (in-place)
    EXPORT_CUDA Matrix4& operator-=(const Matrix4& mat);

    EXPORT_CUDA Matrix4& operator/=(float f);
    
    //! Compares this matrix with another
    EXPORT_CUDA bool operator==(const Matrix4& mat) const
    {
        return ApproxEqual((*this)(0, 0), mat(0, 0)) && ApproxEqual((*this)(0, 1), mat(0, 1)) && ApproxEqual((*this)(0, 2), mat(0, 2)) && ApproxEqual((*this)(0, 3), mat(0, 3)) &&
               ApproxEqual((*this)(1, 0), mat(1, 0)) && ApproxEqual((*this)(1, 1), mat(1, 1)) && ApproxEqual((*this)(1, 2), mat(1, 2)) && ApproxEqual((*this)(1, 3), mat(1, 3)) &&
               ApproxEqual((*this)(2, 0), mat(2, 0)) && ApproxEqual((*this)(2, 1), mat(2, 1)) && ApproxEqual((*this)(2, 2), mat(2, 2)) && ApproxEqual((*this)(2, 3), mat(2, 3)) &&
               ApproxEqual((*this)(3, 0), mat(3, 0)) && ApproxEqual((*this)(3, 1), mat(3, 1)) && ApproxEqual((*this)(3, 2), mat(3, 2)) && ApproxEqual((*this)(3, 3), mat(3, 3));
    }

	EXPORT_CUDA bool operator!=(const Matrix4& mat) const
    {
        return ApproxNotEqual((*this)(0, 0), mat(0, 0)) || ApproxNotEqual((*this)(0, 1), mat(0, 1)) || ApproxNotEqual((*this)(0, 2), mat(0, 2)) || ApproxNotEqual((*this)(0, 3), mat(0, 3)) ||
               ApproxNotEqual((*this)(1, 0), mat(1, 0)) || ApproxNotEqual((*this)(1, 1), mat(1, 1)) || ApproxNotEqual((*this)(1, 2), mat(1, 2)) || ApproxNotEqual((*this)(1, 3), mat(1, 3)) ||
               ApproxNotEqual((*this)(2, 0), mat(2, 0)) || ApproxNotEqual((*this)(2, 1), mat(2, 1)) || ApproxNotEqual((*this)(2, 2), mat(2, 2)) || ApproxNotEqual((*this)(2, 3), mat(2, 3)) ||
               ApproxNotEqual((*this)(3, 0), mat(3, 0)) || ApproxNotEqual((*this)(3, 1), mat(3, 1)) || ApproxNotEqual((*this)(3, 2), mat(3, 2)) || ApproxNotEqual((*this)(3, 3), mat(3, 3));
    }

    static EXPORT_CUDA Matrix4 identityMatrix() { return Matrix4(1, 0, 0, 0,
                                                                 0, 1, 0, 0,
                                                                 0, 0, 1, 0,
                                                                 0, 0, 0, 1); }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    EXPORT_CUDA float& get(unsigned i, unsigned j) { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    EXPORT_CUDA float get(unsigned i, unsigned j) const { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    EXPORT_CUDA float& operator()(unsigned i, unsigned j) { return matrix->m128_f32[i*4+j]; }

    //! Gets the \f$a_{ji}\f$ element of the matrix
    EXPORT_CUDA float operator()(unsigned i, unsigned j) const { return matrix->m128_f32[i*4+j]; }

    //! Get the \f$a_i\f$ column of the matrix
    EXPORT_CUDA float* operator[](unsigned i)
    {
        TGE_ASSERT(i < 4, "Bad element coordex");
        return matrix[i].m128_f32;
    }

    //! Get the \f$a_i\f$ column of the matrix
    EXPORT_CUDA const float* operator[](unsigned i) const
    {
        TGE_ASSERT(i < 4, "Bad element coordex");
        return matrix[i].m128_f32;
    }

    //! Subtracts component wise this matrix with another
    EXPORT_CUDA Matrix4 operator-(const Matrix4& mat) const;

    //! Sums component wise this matrix with another
    EXPORT_CUDA Matrix4 operator+(const Matrix4& mat) const;

    EXPORT_CUDA Matrix4 operator/(float f) const;
    
    //! Multiplies this matrix with another
    EXPORT_CUDA Matrix4 operator*(const Matrix4& mat2) const;

    //! Transforms a 3-dimensional vector
    EXPORT_CUDA Vector3 operator*(const Vector3& vec) const;

    //! Transforms a 4-dimensional vector
    EXPORT_CUDA Vector4 operator*(const Vector4& vec) const;

    //! Returns the transposed matrix
    EXPORT_CUDA Matrix4 transpose() const
    {
        Matrix4 mat(*this);
        mat.transposeSelf();

        return mat;
    }

    //! Transposes this matrix
    EXPORT_CUDA void transposeSelf();

    //! Calculates the inverse matrix
    EXPORT_CUDA Matrix4 inverse() const;

    //! Inverts this matrix
    EXPORT_CUDA void invertSelf()
    {
        *this = inverse();
    }

    //! Sets this matrix to identity matrix
    EXPORT_CUDA void identity();

    //! Scales the coordinate system
    /*!
        \param vec a 3-dimensional vector representing the scaling
    */
    EXPORT_CUDA void scale(const Vector3& vec);
    
    //! Scales the coordinate system
    /*!
        \param s scaling factor
    */
    EXPORT_CUDA void scale(float s);
    
    //! Rotates the coordinate system around the relative x-axis
    /*!
        \param pitch rotation around the x-axis
    */
    EXPORT_CUDA void rotateX(float pitch);
    
    //! Rotates the coordinate system around the relative y-axis
    /*!
        \param yaw rotation around the y-axis
    */
    EXPORT_CUDA void rotateY(float yaw);

    //! Rotates the coordinate system around the relative z-axis
    /*!
        \param roll rotation around the z-axis
    */
    EXPORT_CUDA void rotateZ(float roll);

    //! Rotates the coordinate system
    /*!
        \param angle the angle of rotation
        \param axis the relative axis of rotation
    */
    EXPORT_CUDA void rotate(float angle, const Vector3& axis);

    //! Rotates the coordinate system
    /*!
        \param euler a 3-dimensional vector representing the euler angles
    */
    EXPORT_CUDA void rotate(const Vector3& euler);

    //! Translates the coordinate system
    /*!
        \param vec a 3-dimensional vector representing the relative translation
    */
    EXPORT_CUDA void translate(const Vector3& vec);
    
    //! Translates the coordinate system
    /*!
        \param vec a 2-dimensional vector representing the relative translation
    */
    EXPORT_CUDA void translate(const Vector2& vec);

    //! Translates the coordinate system by the relative x-axis
    /*!
        \param x a floating-point variable representing the relative translation by the x-axis
    */
    EXPORT_CUDA void translateX(float x);

    //! Translates the coordinate system by the relative y-axis
    /*!
        \param y a floating-point variable representing the relative translation by the y-axis
    */
    EXPORT_CUDA void translateY(float y);

    //! Translates the coordinate system by the relative z-axis
    /*!
        \param z a floating-point variable representing the relative translation by the z-axis
    */
    EXPORT_CUDA void translateZ(float z);

    EXPORT_CUDA void lookAtUpTarget(const Vector3& origin, const Vector3& target, const Vector3& up_target)
    {
        Vector3 forward = Normalize(origin - target);
		float up_axis_proj = Dot(up_target, forward);
		Vector3 up = up_target*Dot(up_target, forward) - forward;
		if(Dot(up, up) < 1e-3f)
		{
			Array(up)[0] = Array(up_target)[1];
			Array(up)[1] = Array(up_target)[2];
			Array(up)[2] = Array(up_target)[0];
		}
        Vector3 left = Normalize(Cross(up, forward));
        Vector3 up_final = Cross(forward, left);
        matrix[0][0] = left.x;
        matrix[0][1] = up_final.x;
        matrix[0][2] = forward.x;
        matrix[0][3] = 0.0f;
        matrix[1][0] = left.y;
        matrix[1][1] = up_final.y;
        matrix[1][2] = forward.y;
        matrix[1][3] = 0.0f;
        matrix[2][0] = left.z;
        matrix[2][1] = up_final.z;
        matrix[2][2] = forward.z;
        matrix[2][3] = 0.0f;
        matrix[3][0] = -Dot(origin, left);
        matrix[3][1] = -Dot(origin, up_final);
        matrix[3][2] = -Dot(origin, forward);
        matrix[3][3] = 1.0f;
    }

	EXPORT_CUDA void lookAt(const Vector3& origin, const Vector3& target, const Vector3& up)
    {
        Vector3 forward = Normalize(origin - target);
        Vector3 left = Normalize(Cross(up, forward));
        Vector3 up_final = Cross(forward, left);
        matrix[0][0] = left.x;
        matrix[0][1] = up_final.x;
        matrix[0][2] = forward.x;
        matrix[0][3] = 0.0f;
        matrix[1][0] = left.y;
        matrix[1][1] = up_final.y;
        matrix[1][2] = forward.y;
        matrix[1][3] = 0.0f;
        matrix[2][0] = left.z;
        matrix[2][1] = up_final.z;
        matrix[2][2] = forward.z;
        matrix[2][3] = 0.0f;
        matrix[3][0] = -Dot(origin, left);
        matrix[3][1] = -Dot(origin, up_final);
        matrix[3][2] = -Dot(origin, forward);
        matrix[3][3] = 1.0f;
    }

    //! Rotates a 3-dimensional vector
    EXPORT_CUDA Vector3 transformRotate(const Vector3& v) const;

    EXPORT_CUDA Vector3 relativeX() const
    {
	    return Vector3{(*this)(0, 0), (*this)(0, 1), (*this)(0, 2)};
    }

    EXPORT_CUDA Vector3 relativeY() const
    {
	    return Vector3{(*this)(1, 0), (*this)(1, 1), (*this)(1, 2)};
    }

    EXPORT_CUDA Vector3 relativeZ() const
    {
	    return Vector3{(*this)(2, 0), (*this)(2, 1), (*this)(2, 2)};
    }

    EXPORT_CUDA Vector3 transposeRelativeX()
    {
        return Vector3{(*this)(0, 0), (*this)(1, 0), (*this)(2, 0)};
    }

    EXPORT_CUDA Vector3 transposeRelativeY()
    {
        return Vector3{(*this)(0, 1), (*this)(1, 1), (*this)(2, 1)};
    }

    EXPORT_CUDA Vector3 transposeRelativeZ()
    {
        return Vector3{(*this)(0, 2), (*this)(1, 2), (*this)(2, 2)};
    }

    EXPORT_CUDA Matrix4 rotationMatrix() const
    {
        Matrix4 rot;
        rot.matrix[0] = matrix[0];
        rot.matrix[1] = matrix[1];
        rot.matrix[2] = matrix[2];
        rot.matrix[3] = simd128(0.0f, 0.0f, 0.0f, 1.0f);

        return rot;
    }

    EXPORT_CUDA Matrix4 normalTransform() const
    {
        return inverse().transpose();
    }

    EXPORT_CUDA Matrix3 rotationMatrix3() const
    {
        return Matrix3(Vector3{ matrix[0].m128_f32[0], matrix[0].m128_f32[1], matrix[0].m128_f32[2] },
                       Vector3{ matrix[1].m128_f32[0], matrix[1].m128_f32[1], matrix[1].m128_f32[2] },
                       Vector3{ matrix[2].m128_f32[0], matrix[2].m128_f32[1], matrix[2].m128_f32[2] });
    }

    EXPORT_CUDA Matrix4 rotationFromPerspectiveInverseMatrix() const
    {
        // TODO: Manually inline functions and look at disassembly
        auto origin = (*this) * Vector3{ 0.0f, 0.0f, -1.0f };
        auto far_forward = (*this) * Vector3{ 0.0f, 0.0f, 1.0f };
        auto near_up = (*this) * Vector3{ 0.0f, 1.0f, -1.0f };

        auto forward = Normalize(origin - far_forward);
        auto up = Normalize(near_up - origin);
        auto left = Normalize(Cross(up, forward));

        return Matrix4(left, up, forward).transpose();
    }

    EXPORT_CUDA Vector3 translation() const
    {
	    return Vector3{(*this)(3, 0)/(*this)(3, 3), (*this)(3, 1)/(*this)(3, 3), (*this)(3, 2)/(*this)(3, 3)};
    }

    EXPORT_CUDA Vector3 scaling() const
    {
        Vector3 s;
        s.x = sqrtf((*this)(0, 0)*(*this)(0, 0) + (*this)(0, 1)*(*this)(0, 1) + (*this)(0, 2)*(*this)(0, 2));
        s.y = sqrtf((*this)(1, 0)*(*this)(1, 0) + (*this)(1, 1)*(*this)(1, 1) + (*this)(1, 2)*(*this)(1, 2));
        s.z = sqrtf((*this)(2, 0)*(*this)(2, 0) + (*this)(2, 1)*(*this)(2, 1) + (*this)(2, 2)*(*this)(2, 2));
        return s;
    }

    EXPORT_CUDA void decompose(Vector3& translation, Vector3& scaling, Vector3& euler)
    {
        translation = this->translation();
        scaling = this->scaling();
        float m20 = (*this)(2, 0)/scaling.z;
        if(m20 >= 0.999)
        {
            euler.x = 0.0f;
            euler.y = MathPi*0.5f;
            euler.z = atan2((*this)(0, 1), (*this)(1, 1));
        }
        else if(m20 <= -0.999)
        {
            euler.x = 0.0f;
            euler.y = -MathPi*0.5f;
            euler.z = atan2((*this)(0, 1), (*this)(1, 1));
        }
        else
        {
            euler.x = atan2(-(*this)(2, 1), (*this)(2, 2));
            euler.y = asin(m20);
            euler.z = atan2(-(*this)(1, 0)/scaling.y, (*this)(0, 0)/scaling.x);
        }
    }
};

inline Matrix4 PerspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
    float f = 1.0f / tan(ToRadians(fovy)*0.5f);
    return Matrix4(f / aspect, 0.0f, 0.0f, 0.0f,
                   0.0f, f, 0.0f, 0.0f,
                   0.0f, 0.0f, (zNear + zFar) / (zNear - zFar), (2 * zNear*zFar) / (zNear - zFar),
                   0.0f, 0.0f, -1.0f, 0.0f);
}

inline Matrix4 OrthoMatrix(float left, float right, float bottom, float top, float _near, float _far)
{
    return Matrix4(2.0f / (right - left), 0.0f, 0.0f, (right + left) / (left - right),
                   0.0f, 2.0f / (top - bottom), 0.0f, (top + bottom) / (bottom - top),
                   0.0f, 0.0f, -2.0f / (_far - _near), (_far + _near) / (_near - _far),
                   0.0f, 0.0f, 0.0f, 1.0f);
}

#if (!defined(HAS_SSE) && !defined(HAS_ARM_NEON)) || defined(__CUDA_ARCH__)
inline Matrix4::Matrix4(float* _mat)
{
    memcpy(matrix, _mat, sizeof(matrix));
}

inline Matrix4::Matrix4(float m00, float m01, float m02, float m03,
                        float m10, float m11, float m12, float m13,
                        float m20, float m21, float m22, float m23,
                        float m30, float m31, float m32, float m33)
{
    matrix[0][0] = m00;
    matrix[0][1] = m01;
    matrix[0][2] = m02;
    matrix[0][3] = m03;
    matrix[1][0] = m10;
    matrix[1][1] = m11;
    matrix[1][2] = m12;
    matrix[1][3] = m13;
    matrix[2][0] = m20;
    matrix[2][1] = m21;
    matrix[2][2] = m22;
    matrix[2][3] = m23;
    matrix[3][0] = m30;
    matrix[3][1] = m31;
    matrix[3][2] = m32;
    matrix[3][3] = m33;
}

inline void Matrix4::transposeSelf()
{
    matrix[0][1] = matrix[1][0], matrix[0][2] = matrix[2][0], matrix[0][3] = matrix[3][0];
    matrix[1][0] = matrix[0][1], matrix[1][2] = matrix[2][1], matrix[1][3] = matrix[3][1];
    matrix[2][0] = matrix[0][2], matrix[2][1] = matrix[1][2], matrix[2][3] = matrix[3][2];
    matrix[3][0] = matrix[0][3], matrix[3][1] = matrix[1][3], matrix[3][2] = matrix[2][3];
}

inline void Matrix4::identity()
{
    matrix[0][0] = 1.0f, matrix[0][1] = 0.0f, matrix[0][2] = 0.0f, matrix[0][3] = 0.0f;
    matrix[1][0] = 0.0f, matrix[1][1] = 1.0f, matrix[1][2] = 0.0f, matrix[1][3] = 0.0f;
    matrix[2][0] = 0.0f, matrix[2][1] = 0.0f, matrix[2][2] = 1.0f, matrix[2][3] = 0.0f;
    matrix[3][0] = 0.0f, matrix[3][1] = 0.0f, matrix[3][2] = 0.0f, matrix[3][3] = 1.0f;
}

inline Matrix4 Matrix4::inverse() const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[1][1]*matrix[2][2]*matrix[3][3] + matrix[2][1]*matrix[3][2]*matrix[1][3] + matrix[3][1]*matrix[1][2]*matrix[2][3] - matrix[1][1]*matrix[3][2]*matrix[2][3] - matrix[2][1]*matrix[1][2]*matrix[3][3] - matrix[3][1]*matrix[2][2]*matrix[1][3];
    r.matrix[1][0] = matrix[1][0]*matrix[3][2]*matrix[2][3] + matrix[2][0]*matrix[1][2]*matrix[3][3] + matrix[3][0]*matrix[2][2]*matrix[1][3] - matrix[1][0]*matrix[2][2]*matrix[3][3] - matrix[2][0]*matrix[3][2]*matrix[1][3] - matrix[3][0]*matrix[1][2]*matrix[2][3];
    r.matrix[2][0] = matrix[1][0]*matrix[2][1]*matrix[3][3] + matrix[2][0]*matrix[3][1]*matrix[1][3] + matrix[3][0]*matrix[1][1]*matrix[2][3] - matrix[1][0]*matrix[3][1]*matrix[2][3] - matrix[2][0]*matrix[1][1]*matrix[3][3] - matrix[3][0]*matrix[2][1]*matrix[1][3];
    r.matrix[3][0] = matrix[1][0]*matrix[3][1]*matrix[2][2] + matrix[2][0]*matrix[1][1]*matrix[3][2] + matrix[3][0]*matrix[2][1]*matrix[1][2] - matrix[1][0]*matrix[2][1]*matrix[3][2] - matrix[2][0]*matrix[3][1]*matrix[1][2] - matrix[3][0]*matrix[1][1]*matrix[2][2];
    r.matrix[0][1] = matrix[0][1]*matrix[3][2]*matrix[2][3] + matrix[2][1]*matrix[0][2]*matrix[3][3] + matrix[3][1]*matrix[2][2]*matrix[0][3] - matrix[0][1]*matrix[2][2]*matrix[3][3] - matrix[2][1]*matrix[3][2]*matrix[0][3] - matrix[3][1]*matrix[0][2]*matrix[2][3];
    r.matrix[1][1] = matrix[0][0]*matrix[2][2]*matrix[3][3] + matrix[2][0]*matrix[3][2]*matrix[0][3] + matrix[3][0]*matrix[0][2]*matrix[2][3] - matrix[0][0]*matrix[3][2]*matrix[2][3] - matrix[2][0]*matrix[0][2]*matrix[3][3] - matrix[3][0]*matrix[2][2]*matrix[0][3];
    r.matrix[2][1] = matrix[0][0]*matrix[3][1]*matrix[2][3] + matrix[2][0]*matrix[0][1]*matrix[3][3] + matrix[3][0]*matrix[2][1]*matrix[0][3] - matrix[0][0]*matrix[2][1]*matrix[3][3] - matrix[2][0]*matrix[3][1]*matrix[0][3] - matrix[3][0]*matrix[0][1]*matrix[2][3];
    r.matrix[3][1] = matrix[0][0]*matrix[2][1]*matrix[3][2] + matrix[2][0]*matrix[3][1]*matrix[0][2] + matrix[3][0]*matrix[0][1]*matrix[2][2] - matrix[0][0]*matrix[3][1]*matrix[2][2] - matrix[2][0]*matrix[0][1]*matrix[3][2] - matrix[3][0]*matrix[2][1]*matrix[0][2];
    r.matrix[0][2] = matrix[0][1]*matrix[1][2]*matrix[3][3] + matrix[1][1]*matrix[3][2]*matrix[0][3] + matrix[3][1]*matrix[0][2]*matrix[1][3] - matrix[0][1]*matrix[3][2]*matrix[1][3] - matrix[1][1]*matrix[0][2]*matrix[3][3] - matrix[3][1]*matrix[1][2]*matrix[0][3];
    r.matrix[1][2] = matrix[0][0]*matrix[3][2]*matrix[1][3] + matrix[1][0]*matrix[0][2]*matrix[3][3] + matrix[3][0]*matrix[1][2]*matrix[0][3] - matrix[0][0]*matrix[1][2]*matrix[3][3] - matrix[1][0]*matrix[3][2]*matrix[0][3] - matrix[3][0]*matrix[0][2]*matrix[1][3];
    r.matrix[2][2] = matrix[0][0]*matrix[1][1]*matrix[3][3] + matrix[1][0]*matrix[3][1]*matrix[0][3] + matrix[3][0]*matrix[0][1]*matrix[1][3] - matrix[0][0]*matrix[3][1]*matrix[1][3] - matrix[1][0]*matrix[0][1]*matrix[3][3] - matrix[3][0]*matrix[1][1]*matrix[0][3];
    r.matrix[3][2] = matrix[0][0]*matrix[3][1]*matrix[1][2] + matrix[1][0]*matrix[0][1]*matrix[3][2] + matrix[3][0]*matrix[1][1]*matrix[0][2] - matrix[0][0]*matrix[1][1]*matrix[3][2] - matrix[1][0]*matrix[3][1]*matrix[0][2] - matrix[3][0]*matrix[0][1]*matrix[1][2];
    r.matrix[0][3] = matrix[0][1]*matrix[2][2]*matrix[1][3] + matrix[1][1]*matrix[0][2]*matrix[2][3] + matrix[2][1]*matrix[1][2]*matrix[0][3] - matrix[0][1]*matrix[1][2]*matrix[2][3] - matrix[1][1]*matrix[2][2]*matrix[0][3] - matrix[2][1]*matrix[0][2]*matrix[1][3];
    r.matrix[1][3] = matrix[0][0]*matrix[1][2]*matrix[2][3] + matrix[1][0]*matrix[2][2]*matrix[0][3] + matrix[2][0]*matrix[0][2]*matrix[1][3] - matrix[0][0]*matrix[2][2]*matrix[1][3] - matrix[1][0]*matrix[0][2]*matrix[2][3] - matrix[2][0]*matrix[1][2]*matrix[0][3];
    r.matrix[2][3] = matrix[0][0]*matrix[2][1]*matrix[1][3] + matrix[1][0]*matrix[0][1]*matrix[2][3] + matrix[2][0]*matrix[1][1]*matrix[0][3] - matrix[0][0]*matrix[1][1]*matrix[2][3] - matrix[1][0]*matrix[2][1]*matrix[0][3] - matrix[2][0]*matrix[0][1]*matrix[1][3];
    r.matrix[3][3] = matrix[0][0]*matrix[1][1]*matrix[2][2] + matrix[1][0]*matrix[2][1]*matrix[0][2] + matrix[2][0]*matrix[0][1]*matrix[1][2] - matrix[0][0]*matrix[2][1]*matrix[1][2] - matrix[1][0]*matrix[0][1]*matrix[2][2] - matrix[2][0]*matrix[1][1]*matrix[0][2];
    float det = matrix[0][0]*r.matrix[0][0] + matrix[1][0]*r.matrix[0][1] + matrix[2][0]*r.matrix[0][2] + matrix[3][0]*r.matrix[0][3];
    r /= det;
    
    return r;
}

inline EXPORT_CUDA Matrix4& Matrix4::operator+=(const Matrix4& mat)
{
    matrix[0][0] += mat.matrix[0][0];
    matrix[0][1] += mat.matrix[0][1];
    matrix[0][2] += mat.matrix[0][2];
    matrix[0][3] += mat.matrix[0][3];
    matrix[1][0] += mat.matrix[1][0];
    matrix[1][1] += mat.matrix[1][1];
    matrix[1][2] += mat.matrix[1][2];
    matrix[1][3] += mat.matrix[1][3];
    matrix[2][0] += mat.matrix[2][0];
    matrix[2][1] += mat.matrix[2][1];
    matrix[2][2] += mat.matrix[2][2];
    matrix[2][3] += mat.matrix[2][3];
    matrix[3][0] += mat.matrix[3][0];
    matrix[3][1] += mat.matrix[3][1];
    matrix[3][2] += mat.matrix[3][2];
    matrix[3][3] += mat.matrix[3][3];
    return *this;
}

inline EXPORT_CUDA Matrix4& Matrix4::operator-=(const Matrix4& mat)
{
    matrix[0][0] -= mat.matrix[0][0];
    matrix[0][1] -= mat.matrix[0][1];
    matrix[0][2] -= mat.matrix[0][2];
    matrix[0][3] -= mat.matrix[0][3];
    matrix[1][0] -= mat.matrix[1][0];
    matrix[1][1] -= mat.matrix[1][1];
    matrix[1][2] -= mat.matrix[1][2];
    matrix[1][3] -= mat.matrix[1][3];
    matrix[2][0] -= mat.matrix[2][0];
    matrix[2][1] -= mat.matrix[2][1];
    matrix[2][2] -= mat.matrix[2][2];
    matrix[2][3] -= mat.matrix[2][3];
    matrix[3][0] -= mat.matrix[3][0];
    matrix[3][1] -= mat.matrix[3][1];
    matrix[3][2] -= mat.matrix[3][2];
    matrix[3][3] -= mat.matrix[3][3];
    return *this;
}

inline EXPORT_CUDA Matrix4& Matrix4::operator/=(float f)
{
    matrix[0][0] /= f;
    matrix[0][1] /= f;
    matrix[0][2] /= f;
    matrix[0][3] /= f;
    matrix[1][0] /= f;
    matrix[1][1] /= f;
    matrix[1][2] /= f;
    matrix[1][3] /= f;
    matrix[2][0] /= f;
    matrix[2][1] /= f;
    matrix[2][2] /= f;
    matrix[2][3] /= f;
    matrix[3][0] /= f;
    matrix[3][1] /= f;
    matrix[3][2] /= f;
    matrix[3][3] /= f;
    return *this;
}

inline EXPORT_CUDA Matrix4 Matrix4::operator-(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] - mat.matrix[0][0];
    r.matrix[0][1] = matrix[0][1] - mat.matrix[0][1];
    r.matrix[0][2] = matrix[0][2] - mat.matrix[0][2];
    r.matrix[0][3] = matrix[0][3] - mat.matrix[0][3];
    r.matrix[1][0] = matrix[1][0] - mat.matrix[1][0];
    r.matrix[1][1] = matrix[1][1] - mat.matrix[1][1];
    r.matrix[1][2] = matrix[1][2] - mat.matrix[1][2];
    r.matrix[1][3] = matrix[1][3] - mat.matrix[1][3];
    r.matrix[2][0] = matrix[2][0] - mat.matrix[2][0];
    r.matrix[2][1] = matrix[2][1] - mat.matrix[2][1];
    r.matrix[2][2] = matrix[2][2] - mat.matrix[2][2];
    r.matrix[2][3] = matrix[2][3] - mat.matrix[2][3];
    r.matrix[3][0] = matrix[3][0] - mat.matrix[3][0];
    r.matrix[3][1] = matrix[3][1] - mat.matrix[3][1];
    r.matrix[3][2] = matrix[3][2] - mat.matrix[3][2];
    r.matrix[3][3] = matrix[3][3] - mat.matrix[3][3];
    return r;
}

inline EXPORT_CUDA Matrix4 Matrix4::operator+(const Matrix4& mat) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] + mat.matrix[0][0];
    r.matrix[0][1] = matrix[0][1] + mat.matrix[0][1];
    r.matrix[0][2] = matrix[0][2] + mat.matrix[0][2];
    r.matrix[0][3] = matrix[0][3] + mat.matrix[0][3];
    r.matrix[1][0] = matrix[1][0] + mat.matrix[1][0];
    r.matrix[1][1] = matrix[1][1] + mat.matrix[1][1];
    r.matrix[1][2] = matrix[1][2] + mat.matrix[1][2];
    r.matrix[1][3] = matrix[1][3] + mat.matrix[1][3];
    r.matrix[2][0] = matrix[2][0] + mat.matrix[2][0];
    r.matrix[2][1] = matrix[2][1] + mat.matrix[2][1];
    r.matrix[2][2] = matrix[2][2] + mat.matrix[2][2];
    r.matrix[2][3] = matrix[2][3] + mat.matrix[2][3];
    r.matrix[3][0] = matrix[3][0] + mat.matrix[3][0];
    r.matrix[3][1] = matrix[3][1] + mat.matrix[3][1];
    r.matrix[3][2] = matrix[3][2] + mat.matrix[3][2];
    r.matrix[3][3] = matrix[3][3] + mat.matrix[3][3];
    return r;
}

inline EXPORT_CUDA Matrix4 Matrix4::operator/(float f) const
{
    Matrix4 r;
    r.matrix[0][0] = matrix[0][0] / f;
    r.matrix[0][1] = matrix[0][1] / f;
    r.matrix[0][2] = matrix[0][2] / f;
    r.matrix[0][3] = matrix[0][3] / f;
    r.matrix[1][0] = matrix[1][0] / f;
    r.matrix[1][1] = matrix[1][1] / f;
    r.matrix[1][2] = matrix[1][2] / f;
    r.matrix[1][3] = matrix[1][3] / f;
    r.matrix[2][0] = matrix[2][0] / f;
    r.matrix[2][1] = matrix[2][1] / f;
    r.matrix[2][2] = matrix[2][2] / f;
    r.matrix[2][3] = matrix[2][3] / f;
    r.matrix[3][0] = matrix[3][0] / f;
    r.matrix[3][1] = matrix[3][1] / f;
    r.matrix[3][2] = matrix[3][2] / f;
    r.matrix[3][3] = matrix[3][3] / f;
    return r;
}

inline EXPORT_CUDA Vector3 Matrix4::operator*(const Vector3& vec) const
{
    Vector4 res;
    res.x  = matrix[0][0] * vec.x;
    res.y  = matrix[0][1] * vec.x;
    res.z  = matrix[0][2] * vec.x;
    res.w  = matrix[0][3] * vec.x;
    res.x += matrix[1][0] * vec.y;
    res.y += matrix[1][1] * vec.y;
    res.z += matrix[1][2] * vec.y;
    res.w += matrix[1][3] * vec.y;
    res.x += matrix[2][0] * vec.z;
    res.y += matrix[2][1] * vec.z;
    res.z += matrix[2][2] * vec.z;
    res.w += matrix[2][3] * vec.z;
    res.x += matrix[3][0];
    res.y += matrix[3][1];
    res.z += matrix[3][2];
    res.w += matrix[3][3];   
    res.x /= res.w;
    res.y /= res.w;
    res.z /= res.w;

    return Vector3{ res.x, res.y, res.z };
}

inline EXPORT_CUDA Vector4 Matrix4::operator*(const Vector4& vec) const
{
    Vector4 res;
    res.x  = matrix[0][0] * vec.x;
    res.y  = matrix[0][1] * vec.x;
    res.z  = matrix[0][2] * vec.x;
    res.w  = matrix[0][3] * vec.x;
    res.x += matrix[1][0] * vec.y;
    res.y += matrix[1][1] * vec.y;
    res.z += matrix[1][2] * vec.y;
    res.w += matrix[1][3] * vec.y;
    res.x += matrix[2][0] * vec.z;
    res.y += matrix[2][1] * vec.z;
    res.z += matrix[2][2] * vec.z;
    res.w += matrix[2][3] * vec.z;
    res.x += matrix[3][0] * vec.w;
    res.y += matrix[3][1] * vec.w;
    res.z += matrix[3][2] * vec.w;
    res.w += matrix[3][3] * vec.w;    

    return res;
}

inline EXPORT_CUDA void Matrix4::rotateX(float pitch)
{
    float   s, c;
    FastSinCos(pitch, &s, &c);
    float tmp0 = matrix[1][0],
          tmp1 = matrix[1][1],
          tmp2 = matrix[1][2],
          tmp3 = matrix[1][3];

    matrix[1][0] = tmp0*c    + matrix[2][0]*s;
    matrix[1][1] = tmp1*c    + matrix[2][1]*s;
    matrix[1][2] = tmp2*c    + matrix[2][2]*s;
    matrix[1][3] = tmp3*c    + matrix[2][3]*s;
    matrix[2][0] = tmp0*(-s) + matrix[2][0]*c;
    matrix[2][1] = tmp1*(-s) + matrix[2][1]*c;
    matrix[2][2] = tmp2*(-s) + matrix[2][2]*c;
    matrix[2][3] = tmp3*(-s) + matrix[2][3]*c;
}

inline EXPORT_CUDA void Matrix4::rotateY(float yaw)
{
    float   s, c;
    FastSinCos(yaw, &s, &c);
    float   tmp0 = matrix[0][0],
            tmp1 = matrix[0][1],
            tmp2 = matrix[0][2],
            tmp3 = matrix[0][3];

    matrix[0][0] = tmp0*c + matrix[2][0]*(-s);
    matrix[0][1] = tmp1*c + matrix[2][1]*(-s);
    matrix[0][2] = tmp2*c + matrix[2][2]*(-s);
    matrix[0][3] = tmp3*c + matrix[2][3]*(-s);
    matrix[2][0] = tmp0*s + matrix[2][0]*c;
    matrix[2][1] = tmp1*s + matrix[2][1]*c;
    matrix[2][2] = tmp2*s + matrix[2][2]*c;
    matrix[2][3] = tmp3*s + matrix[2][3]*c;
}

inline EXPORT_CUDA void Matrix4::rotateZ(float roll)
{
    float   s, c;
    FastSinCos(roll, &s, &c);
    float   tmp0 = matrix[0][0],
            tmp1 = matrix[0][1],
            tmp2 = matrix[0][2],
            tmp3 = matrix[0][3];

    matrix[0][0] = tmp0*c    + matrix[1][0]*s;
    matrix[0][1] = tmp1*c    + matrix[1][1]*s;
    matrix[0][2] = tmp2*c    + matrix[1][2]*s;
    matrix[0][3] = tmp3*c    + matrix[1][3]*s;
    matrix[1][0] = tmp0*(-s) + matrix[1][0]*c;
    matrix[1][1] = tmp1*(-s) + matrix[1][1]*c;
    matrix[1][2] = tmp2*(-s) + matrix[1][2]*c;
    matrix[1][3] = tmp3*(-s) + matrix[1][3]*c;
}

inline EXPORT_CUDA void Matrix4::rotate(float angle, const Vector3& axis)
{
    float   x = axis.x, y = axis.y, z = axis.z,
            c = cosf(angle),
            s = sinf(angle),
            xc = (1 - c) * x,
            yc = (1 - c) * y;

    float m00 = xc*x + c,   m01 = xc*y - z*s,   m02 = xc*z + y*s,
          m10 = xc*y + z*s, m11 = yc*y + c,     m12 = yc*z -x*s,
          m20 = xc*z - y*s, m21 = yc*z + x*s,   m22 = z*z*(1-c) + c;
          
    float tmp0 = matrix[0][0], tmp1 = matrix[1][0], tmp2 = matrix[2][0];
    matrix[0][0] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][0] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][0] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][1], tmp1 = matrix[1][1], tmp2 = matrix[2][1];
    matrix[0][1] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][1] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][1] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][2], tmp1 = matrix[1][2], tmp2 = matrix[2][2];
    matrix[0][2] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][2] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][2] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][3], tmp1 = matrix[1][3], tmp2 = matrix[2][3];
    matrix[0][3] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][3] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][3] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
}

inline EXPORT_CUDA void Matrix4::rotate(const Vector3& euler)
{
    float sx, cx, sy, cy, sz, cz;

    FastSinCos(euler.x, &sx, &cx);
    FastSinCos(euler.y, &sy, &cy);
    FastSinCos(euler.z, &sz, &cz);

    float sx_sy = sx*sy;
    float cx_sy = cx*sy;

    float   m00 =  cy*cz,            m01 = -cy*sz,            m02 =  sy,
            m10 =  sx_sy*cz + cx*sz, m11 = -sx_sy*sz + cx*cz, m12 = -sx*cy,
            m20 = -cx_sy*cz + sx*sz, m21 =  cx_sy*sz + sx*cz, m22 =  cx*cy;
    
    float tmp0 = matrix[0][0], tmp1 = matrix[1][0], tmp2 = matrix[2][0];
    matrix[0][0] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][0] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][0] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][1], tmp1 = matrix[1][1], tmp2 = matrix[2][1];
    matrix[0][1] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][1] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][1] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][2], tmp1 = matrix[1][2], tmp2 = matrix[2][2];
    matrix[0][2] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][2] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][2] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
    tmp0 = matrix[0][3], tmp1 = matrix[1][3], tmp2 = matrix[2][3];
    matrix[0][3] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
    matrix[1][3] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
    matrix[2][3] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
}

inline EXPORT_CUDA void Matrix4::scale(const Vector3& s)
{
    matrix[0][0] *= s.x;
    matrix[0][1] *= s.x;
    matrix[0][2] *= s.x;
    matrix[0][3] *= s.x;
    matrix[1][0] *= s.y;
    matrix[1][1] *= s.y;
    matrix[1][2] *= s.y;
    matrix[1][3] *= s.y;
    matrix[2][0] *= s.z;
    matrix[2][1] *= s.z;
    matrix[2][2] *= s.z;
    matrix[2][3] *= s.z;
}

inline EXPORT_CUDA void Matrix4::scale(float s)
{
    matrix[0][0] *= s;
    matrix[0][1] *= s;
    matrix[0][2] *= s;
    matrix[0][3] *= s;
    matrix[1][0] *= s;
    matrix[1][1] *= s;
    matrix[1][2] *= s;
    matrix[1][3] *= s;
    matrix[2][0] *= s;
    matrix[2][1] *= s;
    matrix[2][2] *= s;
    matrix[2][3] *= s;
}

inline EXPORT_CUDA Vector3 Matrix4::transformRotate(const Vector3& vec) const
{
    Vector4 vec4f;
    vec4f.x  = matrix[0][0] * vec.x;
    vec4f.y  = matrix[0][1] * vec.x;
    vec4f.z  = matrix[0][2] * vec.x;
    vec4f.w  = matrix[0][3] * vec.x;
    vec4f.x += matrix[1][0] * vec.y;
    vec4f.y += matrix[1][1] * vec.y;
    vec4f.z += matrix[1][2] * vec.y;
    vec4f.w += matrix[1][3] * vec.y;
    vec4f.x += matrix[2][0] * vec.z;
    vec4f.y += matrix[2][1] * vec.z;
    vec4f.z += matrix[2][2] * vec.z;
    vec4f.w += matrix[2][3] * vec.z;
    vec4f.x /= matrix[3][3];
    vec4f.y /= matrix[3][3];
    vec4f.z /= matrix[3][3];

    return Vector3{ vec4f.x, vec4f.y, vec4f.z };
}

inline EXPORT_CUDA void Matrix4::translate(const Vector3& vec)
{
    matrix[3][0] = matrix[0][0] * vec.x + matrix[1][0] * vec.y + matrix[2][0] * vec.z + matrix[3][0];
    matrix[3][1] = matrix[0][1] * vec.x + matrix[1][1] * vec.y + matrix[2][1] * vec.z + matrix[3][1];
    matrix[3][2] = matrix[0][2] * vec.x + matrix[1][2] * vec.y + matrix[2][2] * vec.z + matrix[3][2];
    matrix[3][3] = matrix[0][3] * vec.x + matrix[1][3] * vec.y + matrix[2][3] * vec.z + matrix[3][3];
}

inline EXPORT_CUDA void Matrix4::translate(const Vector2& vec)
{
    matrix[3][0] = matrix[0][0] * vec.x + matrix[1][0] * vec.y + matrix[3][0];
    matrix[3][1] = matrix[0][1] * vec.x + matrix[1][1] * vec.y + matrix[3][1];
    matrix[3][2] = matrix[0][2] * vec.x + matrix[1][2] * vec.y + matrix[3][2];
    matrix[3][3] = matrix[0][3] * vec.x + matrix[1][3] * vec.y + matrix[3][3];
}

inline EXPORT_CUDA void Matrix4::translateX(float x)
{
    matrix[3][0] = matrix[0][0] * x + matrix[3][0];
    matrix[3][1] = matrix[0][1] * x + matrix[3][1];
    matrix[3][2] = matrix[0][2] * x + matrix[3][2];
    matrix[3][3] = matrix[0][3] * x + matrix[3][3];
}

inline EXPORT_CUDA void Matrix4::translateY(float y)
{
    matrix[3][0] = matrix[1][0] * y + matrix[3][0];
    matrix[3][1] = matrix[1][1] * y + matrix[3][1];
    matrix[3][2] = matrix[1][2] * y + matrix[3][2];
    matrix[3][3] = matrix[1][3] * y + matrix[3][3];
}

inline EXPORT_CUDA void Matrix4::translateZ(float z)
{
    matrix[3][0] = matrix[2][0] * z + matrix[3][0];
    matrix[3][1] = matrix[2][1] * z + matrix[3][1];
    matrix[3][2] = matrix[2][2] * z + matrix[3][2];
    matrix[3][3] = matrix[2][3] * z + matrix[3][3];
}

inline EXPORT_CUDA Matrix4 Matrix4::operator*(const Matrix4& mat2) const 
{
    Matrix4 r;
    TGE_ASSERT(&mat2 != this, "Multiplication of matrix by itself is not supported");
    r.matrix[0][0]  = matrix[0][0] * mat2.matrix[0][0];
    r.matrix[0][1]  = matrix[0][1] * mat2.matrix[0][0];
    r.matrix[0][2]  = matrix[0][2] * mat2.matrix[0][0];
    r.matrix[0][3]  = matrix[0][3] * mat2.matrix[0][0];
    r.matrix[0][0] += matrix[1][0] * mat2.matrix[0][1];
    r.matrix[0][1] += matrix[1][1] * mat2.matrix[0][1];
    r.matrix[0][2] += matrix[1][2] * mat2.matrix[0][1];
    r.matrix[0][3] += matrix[1][3] * mat2.matrix[0][1];
    r.matrix[0][0] += matrix[2][0] * mat2.matrix[0][2];
    r.matrix[0][1] += matrix[2][1] * mat2.matrix[0][2];
    r.matrix[0][2] += matrix[2][2] * mat2.matrix[0][2];
    r.matrix[0][3] += matrix[2][3] * mat2.matrix[0][2];
    r.matrix[0][0] += matrix[3][0] * mat2.matrix[0][3];
    r.matrix[0][1] += matrix[3][1] * mat2.matrix[0][3];
    r.matrix[0][2] += matrix[3][2] * mat2.matrix[0][3];
    r.matrix[0][3] += matrix[3][3] * mat2.matrix[0][3];

    r.matrix[1][0]  = matrix[0][0] * mat2.matrix[1][0];
    r.matrix[1][1]  = matrix[0][1] * mat2.matrix[1][0];
    r.matrix[1][2]  = matrix[0][2] * mat2.matrix[1][0];
    r.matrix[1][3]  = matrix[0][3] * mat2.matrix[1][0];
    r.matrix[1][0] += matrix[1][0] * mat2.matrix[1][1];
    r.matrix[1][1] += matrix[1][1] * mat2.matrix[1][1];
    r.matrix[1][2] += matrix[1][2] * mat2.matrix[1][1];
    r.matrix[1][3] += matrix[1][3] * mat2.matrix[1][1];
    r.matrix[1][0] += matrix[2][0] * mat2.matrix[1][2];
    r.matrix[1][1] += matrix[2][1] * mat2.matrix[1][2];
    r.matrix[1][2] += matrix[2][2] * mat2.matrix[1][2];
    r.matrix[1][3] += matrix[2][3] * mat2.matrix[1][2];
    r.matrix[1][0] += matrix[3][0] * mat2.matrix[1][3];
    r.matrix[1][1] += matrix[3][1] * mat2.matrix[1][3];
    r.matrix[1][2] += matrix[3][2] * mat2.matrix[1][3];
    r.matrix[1][3] += matrix[3][3] * mat2.matrix[1][3];
    
    r.matrix[2][0]  = matrix[0][0] * mat2.matrix[2][0];
    r.matrix[2][1]  = matrix[0][1] * mat2.matrix[2][0];
    r.matrix[2][2]  = matrix[0][2] * mat2.matrix[2][0];
    r.matrix[2][3]  = matrix[0][3] * mat2.matrix[2][0];
    r.matrix[2][0] += matrix[1][0] * mat2.matrix[2][1];
    r.matrix[2][1] += matrix[1][1] * mat2.matrix[2][1];
    r.matrix[2][2] += matrix[1][2] * mat2.matrix[2][1];
    r.matrix[2][3] += matrix[1][3] * mat2.matrix[2][1];
    r.matrix[2][0] += matrix[2][0] * mat2.matrix[2][2];
    r.matrix[2][1] += matrix[2][1] * mat2.matrix[2][2];
    r.matrix[2][2] += matrix[2][2] * mat2.matrix[2][2];
    r.matrix[2][3] += matrix[2][3] * mat2.matrix[2][2];
    r.matrix[2][0] += matrix[3][0] * mat2.matrix[2][3];
    r.matrix[2][1] += matrix[3][1] * mat2.matrix[2][3];
    r.matrix[2][2] += matrix[3][2] * mat2.matrix[2][3];
    r.matrix[2][3] += matrix[3][3] * mat2.matrix[2][3];
    
    r.matrix[3][0]  = matrix[0][0] * mat2.matrix[3][0];
    r.matrix[3][1]  = matrix[0][1] * mat2.matrix[3][0];
    r.matrix[3][2]  = matrix[0][2] * mat2.matrix[3][0];
    r.matrix[3][3]  = matrix[0][3] * mat2.matrix[3][0];
    r.matrix[3][0] += matrix[1][0] * mat2.matrix[3][1];
    r.matrix[3][1] += matrix[1][1] * mat2.matrix[3][1];
    r.matrix[3][2] += matrix[1][2] * mat2.matrix[3][1];
    r.matrix[3][3] += matrix[1][3] * mat2.matrix[3][1];
    r.matrix[3][0] += matrix[2][0] * mat2.matrix[3][2];
    r.matrix[3][1] += matrix[2][1] * mat2.matrix[3][2];
    r.matrix[3][2] += matrix[2][2] * mat2.matrix[3][2];
    r.matrix[3][3] += matrix[2][3] * mat2.matrix[3][2];
    r.matrix[3][0] += matrix[3][0] * mat2.matrix[3][3];
    r.matrix[3][1] += matrix[3][1] * mat2.matrix[3][3];
    r.matrix[3][2] += matrix[3][2] * mat2.matrix[3][3];
    r.matrix[3][3] += matrix[3][3] * mat2.matrix[3][3];
    
    return r;
}
#endif
}

#endif // _TEMPEST_MATRIX_HH_

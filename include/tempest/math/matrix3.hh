/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_MATRIX3_HH_
#define _TEMPEST_MATRIX3_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/matrix2.hh"

namespace Tempest
{
struct Matrix3;

inline EXPORT_CUDA Matrix3 operator*(const Matrix3& mat, float scalar);
inline EXPORT_CUDA Matrix3 operator/(const Matrix3& mat, float scalar);
inline EXPORT_CUDA Matrix3 operator*(float scalar, const Matrix3& mat);
inline EXPORT_CUDA Matrix3 operator*(const Matrix3& lhs, const Matrix3& rhs);

struct Matrix3
{
    union
    {
        Vector3 Columns[3];
        float   Elements[3][3];
    };

    EXPORT_CUDA Matrix3()=default;
    EXPORT_CUDA Matrix3(const Vector3& r0,
                        const Vector3& r1,
                        const Vector3& r2)
    {
        Columns[0] = r0;
        Columns[1] = r1;
        Columns[2] = r2;
    }

    EXPORT_CUDA Matrix3(float angle, const Vector3& axis)
    {
        float   s, c;
        FastSinCos(angle, &s, &c);

        float   x = axis.x, y = axis.y, z = axis.z,
                xc = (1 - c) * x,
                yc = (1 - c) * y;

        Columns[0] = Vector3{ xc*x + c,    xc*y + z*s, xc*z - y*s };
        Columns[1] = Vector3{ xc*y - z*s,  yc*y + c,   yc*z + x*s };
        Columns[2] = Vector3{ xc*z + y*s,  yc*z -x*s,  z*z*(1-c) + c };
    }

    EXPORT_CUDA float operator()(size_t i, size_t j) const { return Elements[i][j]; }

    EXPORT_CUDA const Vector3& column(size_t i) const { return Columns[i]; }

    EXPORT_CUDA Vector3& column(size_t i) { return Columns[i]; }

    EXPORT_CUDA void identity()
    {
        Columns[0] = Vector3{ 1.0f, 0.0f, 0.0f };
        Columns[1] = Vector3{ 0.0f, 1.0f, 0.0f };
        Columns[2] = Vector3{ 0.0f, 0.0f, 1.0f };
    }

    static EXPORT_CUDA Matrix3 identityMatrix()
    {
        return Matrix3({ 1.0f, 0.0f, 0.0f },
                       { 0.0f, 1.0f, 0.0f },
                       { 0.0f, 0.0f, 1.0f });
    }

    EXPORT_CUDA Matrix3 transpose()
    {
        return Matrix3({ Columns[0].x, Columns[1].x, Columns[2].x },
                       { Columns[0].y, Columns[1].y, Columns[2].y },
                       { Columns[0].z, Columns[1].z, Columns[2].z });
    }

    EXPORT_CUDA void makeBasis(const Vector3& norm)
    {
        Vector3 d0 = Cross(Vector3{1.0f, 0.0f, 0.0f}, norm),
			    d1 = Cross(Vector3{0.0f, 1.0f, 0.0f}, norm);
        float d0_len = Dot(d0, d0);
        float d1_len = Dot(d1, d1);
	    Columns[0] = d0_len > d1_len ? d0 / sqrtf(d0_len) : d1 / sqrtf(d1_len);
	    Columns[1] = Cross(norm, Columns[0]);
        Columns[2] = norm;
    }

    EXPORT_CUDA void makeBasisTangent(const Vector3& tan)
    {
        Vector3 d0 = Cross(Vector3{0.0f, 1.0f, 0.0f}, tan),
			    d1 = Cross(Vector3{0.0f, 0.0f, 1.0f}, tan);
        float d0_len = Dot(d0, d0);
        float d1_len = Dot(d1, d1);

	    Columns[0] = tan;
	    Columns[1] = d0_len > d1_len ? d0 / sqrtf(d0_len) : d1 / sqrtf(d1_len);
        Columns[2] = Cross(tan, Columns[1]);
    }

    EXPORT_CUDA void makeBasisOrthogonalize(const Vector3& tan, const Vector3& norm)
    {
        Vector3 binorm = Normalize(Cross(norm, tan));
        Vector3 ortho_tan = Cross(binorm, norm);
        TGE_ASSERT(Dot(tan, ortho_tan) > 1e-3f, "Invalid tangent");

        Columns[0] = ortho_tan;
        Columns[1] = binorm;
        Columns[2] = norm;
    }

	EXPORT_CUDA void rotateTangentPlane(const Matrix2& mat)
	{
		Columns[0] = mat.m_Cols[0].x*Columns[0] + mat.m_Cols[0].y*Columns[1];
		Columns[1] = mat.m_Cols[1].x*Columns[0] + mat.m_Cols[1].y*Columns[1];
	}

    EXPORT_CUDA Vector3 transformRotationInverse(const Vector3& vec) const
    {
        return Vector3{ Dot(vec, Columns[0]), Dot(vec, Columns[1]), Dot(vec, Columns[2]) };
    }

    EXPORT_CUDA Vector3 transform(const Vector3& vec) const
    {
        return vec.x*Columns[0] + vec.y*Columns[1] + vec.z*Columns[2];
    }

    // Produces what is considered across the code as inverse matrix
    // Works properly for orthonormal matrices
    EXPORT_CUDA Matrix3 transformCovariance(const Vector3& variance) const
    {
		auto tan  = Columns[0]*variance.x*Columns[0].x + Columns[1]*variance.y*Columns[1].x + Columns[2]*variance.z*Columns[2].x;
		auto bin  = Columns[0]*variance.x*Columns[0].y + Columns[1]*variance.y*Columns[1].y + Columns[2]*variance.z*Columns[2].y;
		auto norm = Columns[0]*variance.x*Columns[0].z + Columns[1]*variance.y*Columns[1].z + Columns[2]*variance.z*Columns[2].z;

        return Matrix3(tan, bin, norm);
    }

    EXPORT_CUDA Matrix3& operator*=(const Matrix3& matrix)
    {
        return *this = *this * matrix;
    }

    EXPORT_CUDA void scale(const Vector3& scale)
	{
		Columns[0] *= scale.x;
        Columns[1] *= scale.y;
        Columns[2] *= scale.z;
	}

    // Generalized square root
    EXPORT_CUDA Matrix3 choleskyDecomposition() const
    {
        float l00_sq = Columns[0].x;
        float l00 = sqrtf(l00_sq);
        float rcp_l00 = 1.0f/l00;

        float l10 = rcp_l00*Columns[0].y;
        float l20 = rcp_l00*Columns[0].z;

        float l11_sq = Columns[1].y - l10*l10;
        float l11 = sqrtf(l11_sq);
        float rcp_l11 = 1.0f/l11;
        float l21 = rcp_l11*(Columns[1].z - l20*l10);

        float l22_sq = Columns[2].z - l21*l21 - l20*l20;
        float l22 = sqrtf(l22_sq);

        return Matrix3({ l00, l10, l20}, { 0.0f, l11, l21 }, { 0.0f, 0.0f, l22 });
    }

    EXPORT_CUDA float determinant() const
    {
        return Elements[0][0]*(Elements[1][1]*Elements[2][2] -
                               Elements[1][2]*Elements[2][1]) -
               Elements[1][0]*(Elements[0][1]*Elements[2][2] -
                               Elements[0][2]*Elements[2][1]) +
               Elements[2][0]*(Elements[0][1]*Elements[1][2] -
                               Elements[0][2]*Elements[1][1]);
    }

    EXPORT_CUDA Matrix3 adjugate() const
    {
        float cofact00 =   (Elements[1][1]*Elements[2][2] - Elements[1][2]*Elements[2][1]);
        float cofact10 = - (Elements[0][1]*Elements[2][2] - Elements[0][2]*Elements[2][1]);
        float cofact20 =   (Elements[0][1]*Elements[1][2] - Elements[0][2]*Elements[1][1]);

        float cofact01 = - (Elements[1][0]*Elements[2][2] - Elements[1][2]*Elements[2][0]);
        float cofact11 =   (Elements[0][0]*Elements[2][2] - Elements[0][2]*Elements[2][0]);
        float cofact21 = - (Elements[0][0]*Elements[1][2] - Elements[0][2]*Elements[1][0]);

        float cofact02 =   (Elements[1][0]*Elements[2][1] - Elements[1][1]*Elements[2][0]);
        float cofact12 = - (Elements[0][0]*Elements[2][1] - Elements[0][1]*Elements[2][0]);
        float cofact22 =   (Elements[0][0]*Elements[1][1] - Elements[0][1]*Elements[1][0]);

        // Transpose
        return Matrix3({ cofact00, cofact10, cofact20 },
                       { cofact01, cofact11, cofact21 },
                       { cofact02, cofact12, cofact22 });
    }

    EXPORT_CUDA Matrix3 inverse() const
    {
        float det = determinant();
        return adjugate()/det;
    }

    EXPORT_CUDA Vector3& tangent() { return Columns[0]; }
    EXPORT_CUDA Vector3& binormal() { return Columns[1]; }
    EXPORT_CUDA Vector3& normal() { return Columns[2]; }

    EXPORT_CUDA const Vector3& tangent() const { return Columns[0]; }
    EXPORT_CUDA const Vector3& binormal() const { return Columns[1]; }
    EXPORT_CUDA const Vector3& normal() const { return Columns[2]; }

    EXPORT_CUDA void rotateX(float pitch)
    {

        float   s, c;
        FastSinCos(pitch, &s, &c);

        auto tmp = Columns[1];

        Columns[1] = tmp*c    + Columns[2]*s;
        Columns[2] = tmp*(-s) + Columns[2]*c;
    }

    EXPORT_CUDA void rotateY(float yaw)
    {
        float   s, c;
        FastSinCos(yaw, &s, &c);

        auto tmp = Columns[0];

        Columns[0] = tmp*c + Columns[2]*(-s);
        Columns[2] = tmp*s + Columns[2]*c;
    }

    EXPORT_CUDA void rotateZ(float roll)
    {
        float   s, c;
        FastSinCos(roll, &s, &c);

        auto tmp = Columns[0];

        Columns[0] = tmp*c    + Columns[1]*s;
        Columns[1] = tmp*(-s) + Columns[1]*c;
    }

	EXPORT_CUDA void rotate(const Vector3& euler)
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
    
		float tmp0 = Elements[0][0], tmp1 = Elements[1][0], tmp2 = Elements[2][0];
		Elements[0][0] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
		Elements[1][0] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
		Elements[2][0] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
		tmp0 = Elements[0][1], tmp1 = Elements[1][1], tmp2 = Elements[2][1];
		Elements[0][1] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
		Elements[1][1] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
		Elements[2][1] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
    
		tmp0 = Elements[0][2], tmp1 = Elements[1][2], tmp2 = Elements[2][2];
		Elements[0][2] = tmp0 * m00 + tmp1 * m10 + tmp2 * m20;
		Elements[1][2] = tmp0 * m01 + tmp1 * m11 + tmp2 * m21;
		Elements[2][2] = tmp0 * m02 + tmp1 * m12 + tmp2 * m22;
	}

	EXPORT_CUDA Vector3 scaling() const
    {
        Vector3 s;
        s.x = Length(Columns[0]);
        s.y = Length(Columns[1]);
        s.z = Length(Columns[2]);
        return s;
    }

	EXPORT_CUDA void decompose(Vector3* scaling, Vector3* euler)
	{
		*scaling = this->scaling();
        float m20 = Elements[2][0]/scaling->z;
        if(m20 >= 0.9999f)
        {
            euler->x = 0.0f;
            euler->y = MathPi*0.5f;
            euler->z = atan2f(Elements[0][1], Elements[1][1]);
        }
        else if(m20 <= -0.9999f)
        {
            euler->x = 0.0f;
            euler->y = -MathPi*0.5f;
            euler->z = atan2f(Elements[0][1], Elements[1][1]);
        }
        else
        {
            euler->x = atan2f(-Elements[2][1], Elements[2][2]);
            euler->y = asinf(m20);
            euler->z = atan2f(-Elements[1][0]/scaling->y, Elements[0][0]/scaling->x);
        }
	}
};

inline EXPORT_CUDA Matrix3 OuterProduct(const Vector3& lhs, const Vector3& rhs)
{
    return Matrix3({ lhs.x*rhs.x, lhs.y*rhs.x, lhs.z*rhs.x },
                   { lhs.x*rhs.y, lhs.y*rhs.y, lhs.z*rhs.y },
                   { lhs.x*rhs.z, lhs.y*rhs.z, lhs.z*rhs.z });
}

inline EXPORT_CUDA Matrix3 operator*(const Matrix3& mat, float scalar)
{
    return Matrix3(mat.Columns[0]*scalar,
                   mat.Columns[1]*scalar,
                   mat.Columns[2]*scalar);
}

inline EXPORT_CUDA Matrix3 operator/(const Matrix3& mat, float scalar)
{
    return mat * (1/scalar);
}

inline EXPORT_CUDA Matrix3 operator*(const Matrix3& lhs, const Matrix3& rhs)
{
    return Matrix3(rhs.Columns[0].x*lhs.Columns[0] + rhs.Columns[0].y*lhs.Columns[1] + rhs.Columns[0].z*lhs.Columns[2],
                   rhs.Columns[1].x*lhs.Columns[0] + rhs.Columns[1].y*lhs.Columns[1] + rhs.Columns[1].z*lhs.Columns[2],
                   rhs.Columns[2].x*lhs.Columns[0] + rhs.Columns[2].y*lhs.Columns[1] + rhs.Columns[2].z*lhs.Columns[2]);
}

inline EXPORT_CUDA Matrix3 operator*(float scalar, const Matrix3& mat)
{
    return mat * scalar;
}

inline EXPORT_CUDA Matrix3 operator+(const Matrix3& lhs, const Matrix3& rhs)
{
    return Matrix3(lhs.Columns[0] + rhs.Columns[0],
                   lhs.Columns[1] + rhs.Columns[1],
                   lhs.Columns[2] + rhs.Columns[2]);
}

inline EXPORT_CUDA Matrix3& operator+=(Matrix3& lhs, const Matrix3& rhs)
{
    lhs.Columns[0] += rhs.Columns[0];
    lhs.Columns[1] += rhs.Columns[1];
    lhs.Columns[2] += rhs.Columns[2];
    return lhs;
}

inline EXPORT_CUDA bool operator==(const Matrix3& lhs, const Matrix3& rhs)
{
    return lhs.column(0) == rhs.column(0) &&
           lhs.column(1) == rhs.column(1) &&
           lhs.column(2) == rhs.column(2);
}

inline EXPORT_CUDA bool ApproxEqual(const Matrix3& lhs, const Matrix3& rhs, float epsilon = TEMPEST_WEAK_FLOAT_EPSILON)
{
    return ApproxEqual(lhs.column(0), rhs.column(0), epsilon) &&
           ApproxEqual(lhs.column(1), rhs.column(1), epsilon) &&
           ApproxEqual(lhs.column(2), rhs.column(2), epsilon);
}
}

#endif // _TEMPEST_MATRIX3_HH_
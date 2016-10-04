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

#ifndef _TEMPEST_MATRIX2_HH_
#define _TEMPEST_MATRIX2_HH_

#include "tempest/math/vector2.hh"

namespace Tempest
{
struct Matrix2
{
    union
    {
        Vector2 m_Cols[2];
        float   m_Elements[2][2];
    };

    EXPORT_CUDA Matrix2()=default;

    EXPORT_CUDA Matrix2(const Vector2& local_x,
						const Vector2& local_y)
    {
        m_Cols[0] = local_x;
        m_Cols[1] = local_y;
    }

    EXPORT_CUDA Matrix2(float m00, float m01,
						float m10, float m11)
    {
        m_Cols[0].x = m00;
        m_Cols[0].y = m01;
        m_Cols[1].x = m10;
        m_Cols[1].y = m11;
    }

    float operator()(size_t row, size_t col) const { return m_Elements[col][row]; }

    EXPORT_CUDA void identity()
    {
        m_Cols[0].x = 1.0f; m_Cols[0].y = 0.0f;
        m_Cols[1].x = 0.0f; m_Cols[1].y = 1.0f;
    }

    static EXPORT_CUDA Matrix2 rotation(float angle)
    {
        float s, c;
        FastSinCos(angle, &s, &c);

        return Matrix2(Vector2{  c, s },
                       Vector2{ -s, c });
    }

    static EXPORT_CUDA Matrix2 rotation(const Vector2& start, const Vector2& end)
    {
        float c = Dot(start, end);
        float s = WedgeZ(start, end);

        return Matrix2(Vector2{  c, s },
                       Vector2{ -s, c });
    }

    static EXPORT_CUDA Matrix2 slerpLerpMatrix(const Vector2& start, const Vector2& end, float t)
    {
        float c0_len = Length(start);
        float c1_len = Length(end);

        float c = Dot(start, end)/(c0_len*c1_len);
        float s = WedgeZ(start, end)/(c0_len*c1_len);

        float scale = (1 - t) + c1_len*t/c0_len;

        float angle = atan2f(s, c);

        FastSinCos(angle*t, &s, &c);

        return Matrix2(Vector2{  c*scale, s*scale },
                       Vector2{ -s*scale, c*scale });
    }

    EXPORT_CUDA Matrix2 slerp(float t)
    {
        float c, s;
        float angle = atan2f(m_Cols[0].y, m_Cols[0].x);

        FastSinCos(angle*t, &s, &c);

        return Matrix2(Vector2{  c, s },
                       Vector2{ -s, c });
    }

    EXPORT_CUDA Vector2 transformRotationInverse(const Vector2& vec) const { return Vector2{Dot(vec, m_Cols[0]),
																							Dot(vec, m_Cols[1])}; }

    EXPORT_CUDA Vector2 transform(const Vector2& vec) const
    {
        return vec.x*m_Cols[0] + vec.y*m_Cols[1];
    }

    EXPORT_CUDA Matrix2 inverse() const
    {
        float det = m_Cols[0].x*m_Cols[1].y - m_Cols[0].y*m_Cols[1].x;
        Matrix2 inv_matrix( m_Cols[1].y, -m_Cols[0].y,
                           -m_Cols[1].x,  m_Cols[0].x);
        return inv_matrix/det;
    }

    EXPORT_CUDA Matrix2 operator/(float a)
    {
        return Matrix2(m_Cols[0]/a, m_Cols[1]/a);
    }

    EXPORT_CUDA Vector2& column(size_t idx) { return m_Cols[idx]; }

    EXPORT_CUDA const Vector2& column(size_t idx) const { return m_Cols[idx]; }

    EXPORT_CUDA void rotate(float angle)
    {
        float s, c;
        FastSinCos(angle, &s, &c);
        auto tmp = m_Cols[0];

        m_Cols[0] = tmp*c    + m_Cols[1]*s;
        m_Cols[1] = tmp*(-s) + m_Cols[1]*c;
    }

	EXPORT_CUDA void scale(const Vector2& scale)
	{
		m_Cols[0] *= scale.x;
        m_Cols[1] *= scale.y;
	}

    EXPORT_CUDA void scaleX(float scale)
    {
        m_Cols[0] *= scale;
    }

    EXPORT_CUDA void scaleY(float scale)
    {
        m_Cols[1] *= scale;
    }

    EXPORT_CUDA void scale(float scale)
    {
        m_Cols[0] *= scale;
        m_Cols[1] *= scale;
    }

    EXPORT_CUDA Vector2 scaling() const
    {
        return { Length(m_Cols[0]), Length(m_Cols[1]) };
    }

    EXPORT_CUDA Vector2 relativeX() const { return m_Cols[0]; }
    EXPORT_CUDA Vector2 relativeY() const { return m_Cols[1]; }

    bool operator==(const Matrix2& cmp)
    {
        return m_Cols[0] == cmp.m_Cols[0] &&
               m_Cols[1] == cmp.m_Cols[1];
    }
};

}

#endif // _TEMPEST_MATRIX2_HH_
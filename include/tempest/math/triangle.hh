/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
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

#ifndef _TEMPEST_TRIANGLE_HH_
#define _TEMPEST_TRIANGLE_HH_

#include <cstdint>

#include "tempest/math/functions.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/matrix2.hh"

namespace Tempest
{
inline EXPORT_CUDA float TriangleArea(const Vector2& v0, const Vector2& v1, const Vector2& v2)
{
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    return 0.5f*WedgeZ(e1, e2);
}

inline EXPORT_CUDA float TriangleArea(const Vector3& v0, const Vector3& v1, const Vector3& v2)
{
    auto e1 = v1 - v0;
    auto e2 = v2 - v0;
    return 0.5f*Length(Cross(e1, e2));
}

// Lengyel's Method
inline EXPORT_CUDA void GenerateTangentSpace(const Vector3& p0, const Vector3& p1, const Vector3& p2,
                                             const Vector2& tc0, const Vector2& tc1, const Vector2& tc2,
                                             Vector3* tangent_unorm, Vector3* binormal_unorm)
{
    Vector3 edge0 = p1 - p0,
            edge1 = p2 - p0;

    Matrix2 tex_space(tc1 - tc0, tc2 - tc0);
    auto tex_space_inv = tex_space.inverse();
    // try to fix singularities
#ifndef __CUDACC__
    using std::isfinite;
#endif

    if(!isfinite(tex_space_inv(0, 0)))
    {
        Tempest::Vector3 norm = Cross(edge0, edge1);
        Tempest::Vector3 tangent, binormal;

        auto& tc_edge = tex_space.column(0);
        if(tc_edge.x != 0.0f)
        {
            tangent = Sign(tc_edge.x)*(edge0 + edge1);
            binormal = Cross(norm, tangent);
        }
        else
        {
            binormal = Sign(tc_edge.y)*(edge0 + edge1);
            tangent = Cross(binormal, norm);
        }

        *tangent_unorm = tangent;
        *binormal_unorm = binormal;
        return;
    }
    
    Vector2 x_vec = tex_space_inv.transformRotationInverse(Vector2{ edge0.x, edge1.x }),
            y_vec = tex_space_inv.transformRotationInverse(Vector2{ edge0.y, edge1.y }),
            z_vec = tex_space_inv.transformRotationInverse(Vector2{ edge0.z, edge1.z });

    *tangent_unorm = Vector3{ x_vec.x, y_vec.x, z_vec.x };
    *binormal_unorm = Vector3{ x_vec.y, y_vec.y, z_vec.y };
}

void DelaunayTriangulation(const Vector2* points_set, uint32_t point_count, uint32_t** indices, uint32_t* triangle_count);
}

#endif // _TEMPEST_TRIANGLE_HH_
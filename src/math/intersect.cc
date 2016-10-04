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

#include "tempest/math/intersect.hh"

namespace Tempest
{
// Really slow - use optimized data structure instead
bool IntersectTriangleList2D(const Vector2& point, const Vector2* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t* prim_id, Vector3* barycentric_coord)
{
    uint32_t index_count = 3*triangle_count;
    for(uint32_t idx_iter = 0; idx_iter < index_count;)
    {
        auto cur_prim_id = idx_iter/3;
        auto i0 = indices[idx_iter++];
        auto i1 = indices[idx_iter++];
        auto i2 = indices[idx_iter++];

        TGE_ASSERT(i0 < vertex_count && i1 < vertex_count && i2 < vertex_count, "Invalid indices");

        auto& v0 = vertices[i0];
        auto& v1 = vertices[i1];
        auto& v2 = vertices[i2];

        Vector3 tmp_barycentric;
        // TODO: Make it water-tight
        auto result = IntersectTriangle(point, v0, v1, v2, &tmp_barycentric);
        if(result)
        {
            *barycentric_coord = tmp_barycentric;
            *prim_id = cur_prim_id;
            return true;
        }
    }
    return false;
}
}
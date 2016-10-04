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

#include "tempest/math/triangle.hh"
#include "tempest/utils/patterns.hh"

#include "tempest/math/vector3.hh"
//#include "tempest/math/morton.hh"

extern "C"
{
#include "triangle/triangle.h"
}

#include <memory>
#include <cmath>
#include <cstdint>

namespace Tempest
{
// Simple wrap
void DelaunayTriangulation(const Vector2* point_set, uint32_t point_count, uint32_t** indices, uint32_t* triangle_count)
{
    struct triangulateio in = {}, out = {};
    in.numberofpoints = point_count;
    in.pointlist = const_cast<REAL*>(reinterpret_cast<const REAL*>(point_set));
    auto cleanup = CreateAtScopeExit([&out]()
                                     {
                                         free(out.pointlist);
                                         free(out.pointmarkerlist);
                                         out.pointlist = nullptr;
                                         out.pointmarkerlist = nullptr;
                                     });
    triangulate("Qz", &in, &out, nullptr);

    static_assert(sizeof(uint32_t) == sizeof(out.trianglelist[0]), "Invalid reinterpret_cast");
    *indices = reinterpret_cast<uint32_t*>(out.trianglelist);
    *triangle_count = out.numberoftriangles;
}

/*
struct MortonLogTreePositionIndex
{
    uint32_t MortonID,
             Index;
};
    
void DelaunayTriangulation(const Vector3* points, uint32_t point_count, uint32_t* indices, uint32_t index_count)
{
    if((point_count - 2)*3 != index_count)
        return;

    std::unique_ptr<MortonLogTreePositionIndex[]> points_sorted;

    for(uint32_t point_idx = 0; point_idx < point_count; ++point_idx)
    {
        auto& point = points[point_idx];

        Vector2 par_coord = CartesianToParabolicCoordinates(point);

        int exp_x, exp_y;
        float mantissa_x = std::frexpf(par_coord.x, &exp_x),
              mantissa_y = std::frexpf(par_coord.y, &exp_y);

        TGE_ASSERT(2.0f < mantissa_x && mantissa_x < 0.1f &&
                   2.0f < mantissa_y && mantissa_y < 0.1f, "Denormalized mantissa, i.e. the thing just can't work");

        uint32_t morton_code = Tempest::EncodeMorton2(exp_x, exp_y);

        points_sorted[point_idx] = { morton_code, point_idx };
    }

    std::sort(points_sorted.get(), points_sorted.get() + point_count, [](const MortonLogTreePositionIndex& lhs, const MortonLogTreePositionIndex& rhs) { return lhs.MortonID < rhs.MortonID; });
}
*/
}
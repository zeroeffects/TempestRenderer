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

#ifndef _TEMPEST_SHAPE_SPLIT_HH_
#define _TEMPEST_SHAPE_SPLIT_HH_

#include "tempest/math/matrix2.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/intersect.hh"
#include "tempest/math/shapes.hh"

namespace Tempest
{
// That's is good enough to break boxes into halves. It must be extended to work for box parts
// Probably keep list of convex set
inline EXPORT_CUDA bool BoxGridSplit(uint32_t norm_coord, float grid_dist, const Matrix2& rotate_scale, const Matrix2& inv_rotate_scale, const Vector2& pos,
                                     const AABB2& initial_rect,
                                     AABB2* box0, AABB2* box1)
{
    uint32_t tan_coord = 1 - norm_coord;

    auto init_rect_center = (initial_rect.MinCorner + initial_rect.MaxCorner)*0.5f;

    float cutting_plane = roundf(Array(init_rect_center)[norm_coord]/grid_dist)*grid_dist;

    // Cutting plane is not inside - ignore completely
    if(Array(initial_rect.MinCorner)[norm_coord] > cutting_plane || cutting_plane > Array(initial_rect.MaxCorner)[norm_coord])
        return false;

    float tmin, tmax;
    Vector2 tan{}, cutting_pos{};
    Array(tan)[tan_coord] = 1.0f;
    Array(cutting_pos)[norm_coord] = cutting_plane;
    // It is symmetric, so it probably fell between the ribs
    if(!IntersectLineRect2(tan, cutting_pos, inv_rotate_scale, pos, &tmin, &tmax))
        return false;
    
    AABB2 result[2];
    for(auto& aabb : result)
    {
        Array(aabb.MinCorner)[tan_coord] = Maxf(tmin, Array(initial_rect.MinCorner)[tan_coord]);
        Array(aabb.MaxCorner)[tan_coord] = Minf(tmax, Array(initial_rect.MaxCorner)[tan_coord]);
    }

    auto& box_tan = rotate_scale.column(0);
    auto& box_binorm = rotate_scale.column(1);

    auto tan_toward_result1 = Sign(Array(box_tan)[norm_coord])*box_tan;
    auto binorm_toward_result1 = Sign(Array(box_binorm)[norm_coord])*box_binorm;

    float pos_norm = Array(pos)[norm_coord];
    float tan_norm_tr = Array(tan_toward_result1)[norm_coord];
    float binorm_norm_tr = Array(binorm_toward_result1)[norm_coord];
    float sym_extend = tan_norm_tr + binorm_norm_tr;
    {
    float box_corner0 = (pos_norm - sym_extend)*1.001f;
    Array(result[0].MinCorner)[norm_coord] = Maxf(box_corner0, Array(initial_rect.MinCorner)[norm_coord]);
    Array(result[0].MaxCorner)[norm_coord] = cutting_plane;
    }

    {
    float box_corner1 = (pos_norm + sym_extend)*1.001f;
    Array(result[1].MinCorner)[norm_coord] = cutting_plane;
    Array(result[1].MaxCorner)[norm_coord] = Minf(box_corner1, Array(initial_rect.MaxCorner)[norm_coord]);
    }

    // These are the vectors that determine extends
    float pos_tan = Array(pos)[tan_coord];
    float tan_tan_tr = Array(tan_toward_result1)[tan_coord];
    float binorm_tan_tr = Array(binorm_toward_result1)[tan_coord];

    float asym_extend = tan_tan_tr - binorm_tan_tr;
    {
    float box_corner_tan_ext0 = (pos_tan - asym_extend)*1.001f;
    size_t mod_box0 = (size_t)(box_corner_tan_ext0 > cutting_plane);
    Array(result[mod_box0].MinCorner)[tan_coord] = Clampf(Array(result[mod_box0].MinCorner)[tan_coord], box_corner_tan_ext0, Array(initial_rect.MinCorner)[tan_coord]);
    Array(result[mod_box0].MaxCorner)[tan_coord] = Clampf(Array(result[mod_box0].MaxCorner)[tan_coord], Array(initial_rect.MaxCorner)[tan_coord], box_corner_tan_ext0);
    }

    {
    float box_corner_tan_ext1 = (pos_tan + asym_extend)*1.001f;
    size_t mod_box1 = (size_t)(box_corner_tan_ext1 > cutting_plane);
    Array(result[mod_box1].MinCorner)[tan_coord] = Clampf(Array(result[mod_box1].MinCorner)[tan_coord], box_corner_tan_ext1, Array(initial_rect.MinCorner)[tan_coord]);
    Array(result[mod_box1].MaxCorner)[tan_coord] = Clampf(Array(result[mod_box1].MaxCorner)[tan_coord], Array(initial_rect.MaxCorner)[tan_coord], box_corner_tan_ext1);
    }

    *box0 = result[0];
    *box1 = result[1];

    return true;
}
}

#endif // _TEMPEST_SHAPE_SPLIT_HH_
/*   The MIT License
*
*   Tempest Renderer
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

#ifndef _TEMPEST_LBVH2_HH_
#define _TEMPEST_LBVH2_HH_

#include "tempest/math/shape-split.hh"
#include "tempest/mesh/bvh-intersect-leaf.hh"

namespace Tempest
{
#define LBVH_LEAF_DECORATION (1u << 31u)

template<class TAABB>
struct LBVH2Node
{
    typedef TAABB BoundsType;
    BoundsType Bounds;
    union
    {
        uint32_t Child2;
        uint32_t Patch;
    };
};

template<class TAABB>
LBVH2Node<TAABB>* GenerateTriangleNodes(void* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t stride = sizeof(typename TAABB::BoundsType));

template<class TAABB, class TFunc>
inline EXPORT_CUDA void IntersectLBVHNode(const LBVH2Node<TAABB>* bvh, uint32_t node_id, const Vector2& pos, TFunc& intersect_func)
{
    auto& cur_node = bvh[node_id];
    if(cur_node.Bounds.MinCorner.x > pos.x || pos.x > cur_node.Bounds.MaxCorner.x ||
       cur_node.Bounds.MinCorner.y > pos.y || pos.y > cur_node.Bounds.MaxCorner.y)                                 
    {
        return;
    }

    if(cur_node.Patch & LBVH_LEAF_DECORATION)
    {
        intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);
    }
    else
    {
        IntersectLBVHNode(bvh, node_id + 1, pos, intersect_func);
        IntersectLBVHNode(bvh, cur_node.Child2, pos, intersect_func);
    }
}

template<class TAABB, class TFunc>
inline EXPORT_CUDA bool IntersectLBVHNodeSingle(const LBVH2Node<TAABB>* bvh, uint32_t node_id, const Vector2& pos, TFunc& intersect_func)
{
    auto& cur_node = bvh[node_id];
    if(cur_node.Bounds.MinCorner.x > pos.x || pos.x > cur_node.Bounds.MaxCorner.x ||
       cur_node.Bounds.MinCorner.y > pos.y || pos.y > cur_node.Bounds.MaxCorner.y)                                 
    {
        return false;
    }

    if(cur_node.Patch & LBVH_LEAF_DECORATION)
    {
        return intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);
    }
    else
    {
        return IntersectLBVHNodeSingle(bvh, node_id + 1, pos, intersect_func) ||
			   IntersectLBVHNodeSingle(bvh, cur_node.Child2, pos, intersect_func);
    }
}

template<class TAABB>
LBVH2Node<TAABB>* GenerateLBVH(LBVH2Node<TAABB>* interm_nodes, uint32_t total_node_count);
}

#endif // _TEMPEST_LBVH2_HH_
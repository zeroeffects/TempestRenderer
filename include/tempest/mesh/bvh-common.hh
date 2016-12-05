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

#ifndef _TEMPEST_BVH_COMMON_HH_
#define _TEMPEST_BVH_COMMON_HH_

#include "tempest/math/morton.hh"
#include "tempest/math/shapes.hh"

namespace Tempest
{
template<class TNode>
typename TNode::BoundsType ComputeCenterBounds(TNode* nodes, uint32_t node_count)
{
    if(node_count == 0)
        return {};

    auto init_center = (nodes->Bounds.MinCorner + nodes->Bounds.MaxCorner)*0.5f;

    typename TNode::BoundsType result{ init_center, init_center };
    for(uint32_t node_idx = 1; node_idx < node_count; ++node_idx)
    {
        auto& box = nodes[node_idx].Bounds;
        auto center = (box.MinCorner + box.MaxCorner)*0.5f;
        result.MinCorner = GenericMin(result.MinCorner, center);
        result.MaxCorner = GenericMax(result.MaxCorner, center);
    }

    return result;
}

inline uint32_t ComputeAABBMortonCode(const AABB2& box, const AABB2& set_bounds)
{
    auto center = (box.MinCorner + box.MaxCorner)*0.5f;

    auto set_span = set_bounds.MaxCorner - set_bounds.MinCorner;

    uint32_t x_quant = set_span.x ? (uint32_t)(Clampf((center.x - set_bounds.MinCorner.x)*65535.0f/set_span.x, 0.0f, 65535.0f) + 0.5f) : 0;
    uint32_t y_quant = set_span.y ? (uint32_t)(Clampf((center.y - set_bounds.MinCorner.y)*65535.0f/set_span.y, 0.0f, 65535.0f) + 0.5f) : 0;
    return EncodeMorton2(x_quant, y_quant);
}

inline uint32_t ComputePointMortonCode(const Vector3& point, const AABBUnaligned& set_bounds)
{
    auto set_span = set_bounds.MaxCorner - set_bounds.MinCorner;

    uint32_t x_quant = set_span.x ? (uint32_t)(Clampf((point.x - set_bounds.MinCorner.x)*1023.0f/set_span.x, 0.0f, 1023.0f) + 0.5f) : 0;
    uint32_t y_quant = set_span.y ? (uint32_t)(Clampf((point.y - set_bounds.MinCorner.y)*1023.0f/set_span.y, 0.0f, 1023.0f) + 0.5f) : 0;
    uint32_t z_quant = set_span.z ? (uint32_t)(Clampf((point.z - set_bounds.MinCorner.z)*1023.0f/set_span.z, 0.0f, 1023.0f) + 0.5f) : 0;
    return EncodeMorton3(x_quant, y_quant, z_quant);
}

inline uint32_t ComputeAABBMortonCode(const AABBUnaligned& box, const AABBUnaligned& set_bounds)
{
    auto center = (box.MinCorner + box.MaxCorner)*0.5f;
    return ComputePointMortonCode(center, set_bounds);
}

template<class TNode>
inline uint32_t FindSplitPlane(TNode* nodes, uint32_t first, uint32_t last)
{
    uint32_t first_code = nodes[first].MortonId;
    uint32_t last_code = nodes[last].MortonId;

    uint32_t split;
    if (first_code != last_code)
    {
        uint32_t common_prefix = __lzcnt(first_code ^ last_code);

        split = first;
        uint32_t step = last - first;

        // Basically, binary search
        do
        {
            step = (step + 1) >> 1;
            uint32_t new_split = split + step;

            if(new_split < last)
            {
                uint32_t split_code = nodes[new_split].MortonId;
                uint32_t split_prefix = __lzcnt(first_code ^ split_code);
                if (split_prefix > common_prefix)
                    split = new_split;
            }
        } while(step > 1);
    }
    else
    {
        split = (first + last) >> 1;
    }
    return split;
}

struct GenPatchNode
{
    uint32_t MortonId;
    uint32_t PatchId;
};
}

#endif // _TEMPEST_BVH_COMMON_HH_
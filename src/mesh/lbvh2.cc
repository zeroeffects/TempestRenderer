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

#include <algorithm>
#include <cstdint>
#include <memory>

#include "tempest/mesh/lbvh2.hh"
#include "tempest/utils/assert.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/functions.hh"
#include "tempest/mesh/bvh-common.hh"

namespace Tempest
{
template<class TAABB>
LBVH2Node<TAABB>* GenerateTriangleNodes(void* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t stride)
{
    std::unique_ptr<LBVH2Node<TAABB>[]> interm_nodes(new LBVH2Node<TAABB>[triangle_count]);
    uint32_t index_count = 3*triangle_count;
	uint8_t* vert_ptr = reinterpret_cast<uint8_t*>(vertices);
	typedef typename TAABB::BoundsType BoundsType;
    for(uint32_t idx_iter = 0; idx_iter < index_count;)
    {
        auto cur_prim_id = idx_iter/3;
        auto i0 = indices[idx_iter++];
        auto i1 = indices[idx_iter++];
        auto i2 = indices[idx_iter++];

        TGE_ASSERT(i0 < vertex_count && i1 < vertex_count && i2 < vertex_count, "Invalid indices");

        auto& v0 = *reinterpret_cast<BoundsType*>(vert_ptr + i0*stride);
        auto& v1 = *reinterpret_cast<BoundsType*>(vert_ptr + i1*stride);
        auto& v2 = *reinterpret_cast<BoundsType*>(vert_ptr + i2*stride);

		auto& interm_node = interm_nodes[cur_prim_id];
		interm_node.Patch = LBVH_LEAF_DECORATION | cur_prim_id;
		TriangleBounds(v0, v1, v2, &interm_node.Bounds);
    }
    return interm_nodes.release();
}

// Adapted from article by Tero Karras about LBVH
template<class TAABB>
static inline uint32_t GenerateHierarchy(const GenPatchNode* gen_nodes, const LBVH2Node<TAABB>* leaf_nodes, uint32_t first, uint32_t last, LBVH2Node<TAABB>* out_nodes, uint32_t* out_size)
{
    // TODO: Replace with explicit stack - this thing tends to lead to stack overflows
    uint32_t cur_node = *out_size;
    auto& out_node = out_nodes[(*out_size)++];
    if(first == last)
    {
        TGE_ASSERT((gen_nodes[first].PatchId & LBVH_LEAF_DECORATION) == 0, "Too many children");
        auto patch_id = gen_nodes[first].PatchId;
		auto& leaf_node = leaf_nodes[patch_id];
        out_node.Patch = leaf_node.Patch;
		out_node.Bounds = leaf_node.Bounds;
        
        return cur_node;
    }

    uint32_t split = FindSplitPlane(gen_nodes, first, last);

    uint32_t child0_idx = GenerateHierarchy(gen_nodes, leaf_nodes, first, split, out_nodes, out_size);
    TGE_ASSERT(child0_idx == cur_node + 1, "Invalid depth-first tree");
    uint32_t child1_idx = out_node.Child2 = GenerateHierarchy(gen_nodes, leaf_nodes, split + 1, last, out_nodes, out_size);
    TGE_ASSERT(child1_idx == child0_idx + 2*(split - first + 1) - 1, "Invalid depth-first hierarchy");

    auto& child0 = out_nodes[child0_idx];
    auto& child1 = out_nodes[child1_idx];

    out_node.Bounds.MinCorner = GenericMin(child0.Bounds.MinCorner, child1.Bounds.MinCorner);
    out_node.Bounds.MaxCorner = GenericMax(child0.Bounds.MaxCorner, child1.Bounds.MaxCorner);

    return cur_node;
}

template<class TAABB>
LBVH2Node<TAABB>* GenerateLBVH(LBVH2Node<TAABB>* interm_nodes, uint32_t total_node_count)
{
	std::unique_ptr<GenPatchNode[]> gen_patch_nodes(new GenPatchNode[total_node_count]);
    TAABB set_bounds = ComputeCenterBounds(interm_nodes, total_node_count);
	for(uint32_t i = 0; i < total_node_count; ++i)
    {
        auto& node = interm_nodes[i];
        auto& interm_patch = gen_patch_nodes[i];
        interm_patch.MortonId = ComputeAABBMortonCode(node.Bounds, set_bounds);
        interm_patch.PatchId = i;
    }

    std::sort(gen_patch_nodes.get(), gen_patch_nodes.get() + total_node_count, [](const GenPatchNode& lhs, const GenPatchNode& rhs) { return lhs.MortonId < rhs.MortonId; });
    
    // Because we are basically splitting a binary tree. There is a quite easy to compute upper bound
    uint32_t max_node_count = 2*total_node_count - 1;
    std::unique_ptr<LBVH2Node<TAABB>[]> nodes(new LBVH2Node<TAABB>[max_node_count]);

    uint32_t out_size = 0;
    auto first_node = GenerateHierarchy(gen_patch_nodes.get(), interm_nodes, 0, total_node_count - 1, nodes.get(), &out_size);
    TGE_ASSERT(first_node == 0, "Invalid node");

    return nodes.release();
}

template LBVH2Node<AABB2>* GenerateLBVH(LBVH2Node<AABB2>* interm_nodes, uint32_t total_node_count);
template LBVH2Node<AABBUnaligned>* GenerateLBVH(LBVH2Node<AABBUnaligned>* interm_nodes, uint32_t total_node_count);

template LBVH2Node<AABB2>* GenerateTriangleNodes(void* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t stride);
template LBVH2Node<AABBUnaligned>* GenerateTriangleNodes(void* vertices, uint32_t vertex_count, uint32_t* indices, uint32_t triangle_count, uint32_t stride);
}
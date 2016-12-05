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

#include <algorithm>
#include <cstdint>
#include <memory>

#include "tempest/mesh/sslbvh2.hh"
#include "tempest/utils/assert.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/functions.hh"
#include "tempest/mesh/bvh-common.hh"

namespace Tempest
{
template<class TAABB>
void GenerateSSLBVH(LBVH2Node<TAABB>* leaf_nodes, uint32_t total_node_count, SimpleStacklessLBVH2Node<TAABB>* out_nodes)
{
    std::unique_ptr<GenPatchNode[]> gen_nodes(new GenPatchNode[total_node_count]);
    TAABB set_bounds = ComputeCenterBounds(leaf_nodes, total_node_count);
	for(uint32_t i = 0; i < total_node_count; ++i)
    {
        auto& node = leaf_nodes[i];
        auto& interm_patch = gen_nodes[i];
        interm_patch.MortonId = ComputeAABBMortonCode(node.Bounds, set_bounds);
        interm_patch.PatchId = i;
    }

    std::sort(gen_nodes.get(), gen_nodes.get() + total_node_count, [](const GenPatchNode& lhs, const GenPatchNode& rhs) { return lhs.MortonId < rhs.MortonId; });
    
    std::unique_ptr<uint32_t[]> split_stack(new uint32_t[total_node_count]);

    uint32_t cur_node = 0, split_stack_pointer = 0,
             first = 0, last = total_node_count - 1,
             parent = SSLBVH_INVALID_NODE;

    for(;;)
    {
        auto& out_node = out_nodes[cur_node];
        out_node.Parent = parent;
        if(first == last)
        {
            TGE_ASSERT((gen_nodes[first].PatchId & SSLBVH_LEAF_DECORATION) == 0, "Too many children");
            auto patch_id = gen_nodes[first].PatchId;
		    auto& leaf_node = leaf_nodes[patch_id];
            out_node.Patch = leaf_node.Patch;
		    out_node.Bounds = leaf_node.Bounds;
        
            parent = out_nodes[cur_node].Parent;

            while(parent != SSLBVH_INVALID_NODE)
            {
                --split_stack_pointer;

                first = last + 1;
                last = split_stack[split_stack_pointer];

                if(parent + 1 == cur_node)
                {
                    ++split_stack_pointer;
                    cur_node = out_nodes[parent].Child2;
                    break;
                }
                else
                {
                    auto& parent_node = out_nodes[parent];
                    auto& child0 = out_nodes[parent + 1],
                        & child1 = out_nodes[cur_node];
                    parent_node.Bounds.MinCorner = GenericMin(child0.Bounds.MinCorner, child1.Bounds.MinCorner);
                    parent_node.Bounds.MaxCorner = GenericMax(child0.Bounds.MaxCorner, child1.Bounds.MaxCorner);
                }

                cur_node = parent;
                parent = out_nodes[cur_node].Parent;
            }
            if(parent == SSLBVH_INVALID_NODE)
            {
                break;
            }
        }
        else
        {
            uint32_t split = FindSplitPlane(gen_nodes.get(), first, last);

            uint32_t child0_idx = cur_node + 1;
            uint32_t child1_idx = out_node.Child2 = child0_idx + 2*(split - first + 1) - 1;

            auto& child0 = out_nodes[child0_idx];
            auto& child1 = out_nodes[child1_idx];

            out_node.Bounds.MinCorner = GenericMin(child0.Bounds.MinCorner, child1.Bounds.MinCorner);
            out_node.Bounds.MaxCorner = GenericMax(child0.Bounds.MaxCorner, child1.Bounds.MaxCorner);

            TGE_ASSERT(split_stack_pointer < total_node_count, "Invalid stack element");
            split_stack[split_stack_pointer] = last;

            parent = cur_node;
            ++cur_node;
            ++split_stack_pointer;
            last = split;
        }
    }
}


template<class TAABB>
SimpleStacklessLBVH2Node<TAABB>* GenerateSSLBVH(LBVH2Node<TAABB>* leaf_nodes, uint32_t total_node_count)
{
    uint32_t max_node_count = SSLBVHMaxNodeCount(total_node_count);
    std::unique_ptr<SimpleStacklessLBVH2Node<TAABB>[]> nodes(new SimpleStacklessLBVH2Node<TAABB>[max_node_count]);

    GenerateSSLBVH(leaf_nodes, total_node_count, nodes.get());

    return nodes.release();
}

template SimpleStacklessLBVH2Node<AABBUnaligned>* GenerateSSLBVH(LBVH2Node<AABBUnaligned>* interm_nodes, uint32_t total_node_count);
template SimpleStacklessLBVH2Node<AABB2>* GenerateSSLBVH(LBVH2Node<AABB2>* interm_nodes, uint32_t total_node_count); 
}
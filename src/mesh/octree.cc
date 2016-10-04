/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#include "tempest/math/morton.hh"
#include "tempest/mesh/octree.hh"
#include "tempest/math/vector3.hh"
#include "tempest/mesh/bvh-common.hh"

#include <memory>
#include <algorithm>

namespace Tempest
{
struct GenOctreeNode
{
    uint32_t MortonId;
    uint32_t PointId;
};

template<class TNode>
TNode ComputePointBounds(const typename TNode::BoundsType* points, uint32_t point_count)
{
    if(point_count == 0)
        return {};

    auto init_center = points[0];

    TNode result{ init_center, init_center };
    for(uint32_t point_idx = 1; point_idx < point_count; ++point_idx)
    {
        result.MinCorner = GenericMin(result.MinCorner, points[point_idx]);
        result.MaxCorner = GenericMax(result.MaxCorner, points[point_idx]);
    }

    return result;
}

uint32_t FindSplit(uint32_t depth, const GenOctreeNode* gen_nodes, uint32_t first, uint32_t last, uint32_t first_split_code)
{
    // Basically, binary search
    uint32_t split = first;
    uint32_t step = last - first;

    do
    {
        step = (step + 1) >> 1;
        uint32_t new_split = split + step;

        if(new_split < last)
        {
            uint32_t split_code = (gen_nodes[new_split].MortonId >> depth) & 1;
            if(first_split_code == split_code) // advance forward
                split = new_split;
        }
    } while(step > 1);

    return split;
}

static inline uint32_t GenerateOctreeHierarchy(uint32_t depth, const GenOctreeNode* gen_nodes, uint32_t first, uint32_t last, OctreeNode* out_nodes, uint32_t* out_size)
{
    // TODO: Replace with explicit stack - this thing tends to lead to stack overflows
    uint32_t cur_node = *out_size;
    auto& out_node = out_nodes[(*out_size)++];
    TGE_ASSERT(first != last, "Invalid range");

    struct Stack
    {
        uint32_t Indices[3];
        uint32_t ChildrenOffset = 0;
        uint32_t ChildrenIterator = 0;
        uint32_t ChildrenEnd = 0;
    } octree_stack[3];

    octree_stack->Indices[0] = first;
    octree_stack->Indices[1] = last;

    TGE_ASSERT((depth + 1) % 3 == 0, "Invalid depth");

    uint32_t cur_offset = 0;

    for(uint32_t dim_idx = 0;;)
    {
        auto& info = octree_stack[dim_idx];

        auto cur_first = info.Indices[0];
        auto cur_last = info.Indices[1];

        if(info.ChildrenOffset == 0 && info.ChildrenEnd == 0)
        {
            uint32_t first_code = gen_nodes[cur_first].MortonId;
            uint32_t last_code = gen_nodes[cur_last].MortonId;

            uint32_t first_split_code = (first_code >> depth) & 1;
            uint32_t last_split_code = (last_code >> depth) & 1;

            if(first_split_code != last_split_code)
            {
                uint32_t split = FindSplit(depth, gen_nodes, cur_first, cur_last, first_split_code);

                info.ChildrenIterator = info.ChildrenOffset = cur_offset;
                info.ChildrenEnd = cur_offset + (1 << (3 - dim_idx));

                info.Indices[0] = cur_first;
                info.Indices[1] = split;
                info.Indices[2] = cur_last;
            }
            else
            {
                uint32_t fill_in_size = (1 << (2 - dim_idx));
                if(first_split_code == 1)
                {
                    for(uint32_t child_idx = cur_offset, child_idx_end = cur_offset + fill_in_size; child_idx < child_idx_end; ++child_idx)
                        out_node.Children[child_idx] = OCTREE_INVALID_NODE;

                    info.ChildrenIterator = info.ChildrenOffset = cur_offset + fill_in_size;
                    info.ChildrenEnd = cur_offset + 2*fill_in_size;
                }
                else // last_split_code == 0
                {
                    for(uint32_t child_idx = cur_offset + fill_in_size, child_idx_end = cur_offset + 2*fill_in_size; child_idx < child_idx_end; ++child_idx)
                        out_node.Children[child_idx] = OCTREE_INVALID_NODE;

                    info.ChildrenIterator = info.ChildrenOffset = cur_offset;
                    info.ChildrenEnd = cur_offset + fill_in_size;
                }
            }
        }
        
        if(dim_idx == 2)
        {
            auto child_idx = info.ChildrenIterator;
            if(info.ChildrenIterator == info.ChildrenEnd)
            {
                info.ChildrenOffset = 0;
                info.ChildrenEnd = 0;
                // Rollback
                for(;;)
                {
                    --dim_idx;
                    ++depth;
                    auto& next_info = octree_stack[dim_idx];
                    next_info.ChildrenIterator += (1 << (2 - dim_idx));
                    if(next_info.ChildrenIterator != next_info.ChildrenEnd)
                        break;

                    if(dim_idx == 0)
                    {
                        return cur_node;
                    }
                        
                    next_info.ChildrenOffset = next_info.ChildrenEnd = 0;
                }
            }
            else
            {
                uint32_t idx = info.ChildrenIterator - info.ChildrenOffset;
                uint32_t cur_sub_first = info.Indices[idx]; 
                if(idx)
                    ++cur_sub_first;
                uint32_t cur_sub_last = info.Indices[idx + 1];

            #ifndef NDEBUG
                uint32_t first_code = (gen_nodes[cur_sub_first].MortonId >> depth) & 7;
                uint32_t last_code = (gen_nodes[cur_sub_last].MortonId >> depth) & 7;

                TGE_ASSERT(first_code == last_code, "Invalid splits");
                TGE_ASSERT(first_code == info.ChildrenIterator, "Invalid splits");
            #endif
                if(cur_sub_first == cur_sub_last)
                {
                    out_node.Children[info.ChildrenIterator] = OCTREE_LEAF_NODE_DECORATION | gen_nodes[cur_sub_first].PointId;
                }
                else
                {
                    //auto child_idx = 
                    out_node.Children[info.ChildrenIterator] = GenerateOctreeHierarchy(depth - 1, gen_nodes, cur_sub_first, cur_sub_last, out_nodes, out_size);
                    //out_nodes[child_idx].Parent = cur_node;

                    #ifndef NDEBUG
                        uint32_t children_count = 0;
                        for(uint32_t child_idx = 0; child_idx < 8; ++child_idx)
                        {
                            if(out_node.Children[child_idx])
                                ++children_count;
                        }

                        TGE_ASSERT(children_count, "Node lacks children");
                    #endif
                }
                ++info.ChildrenIterator;
            }
        }
        else
        {
            uint32_t idx = (info.ChildrenIterator - info.ChildrenOffset)/((2 - dim_idx)*2);
            auto& next_info = octree_stack[dim_idx + 1];
            next_info.Indices[0] = info.Indices[idx];
            if(idx)
                ++next_info.Indices[0];
            next_info.Indices[1] = info.Indices[idx + 1];
            cur_offset = info.ChildrenIterator;
            ++dim_idx;
            --depth;
        }
    }
    
    return cur_node;
}

inline uint32_t ComputePointMortonCodeTrunc(const Vector3& point, const AABBUnaligned& set_bounds)
{
    auto set_span = set_bounds.MaxCorner - set_bounds.MinCorner;

    uint32_t x_quant = set_span.x ? (uint32_t)(Clampf((point.x - set_bounds.MinCorner.x)*1024.0f/set_span.x, 0.0f, 1023.0f)) : 0;
    uint32_t y_quant = set_span.y ? (uint32_t)(Clampf((point.y - set_bounds.MinCorner.y)*1024.0f/set_span.y, 0.0f, 1023.0f)) : 0;
    uint32_t z_quant = set_span.z ? (uint32_t)(Clampf((point.z - set_bounds.MinCorner.z)*1024.0f/set_span.z, 0.0f, 1023.0f)) : 0;
    return EncodeMorton3(x_quant, y_quant, z_quant);
}

Octree BuildOctreeMorton(const Tempest::Vector3* points, uint32_t point_count)
{
    if(point_count == 0)
        return {};

    std::unique_ptr<GenOctreeNode[]> gen_nodes(new GenOctreeNode[point_count]);
    AABBUnaligned set_bounds = ComputePointBounds<AABBUnaligned>(points, point_count);
    for(uint32_t i = 0; i < point_count; ++i)
    {
        auto& point = points[i];
        auto& interm_patch = gen_nodes[i];
        interm_patch.MortonId = ComputePointMortonCodeTrunc(point, set_bounds);
        interm_patch.PointId = i;
    }
    
    std::sort(gen_nodes.get(), gen_nodes.get() + point_count, [](const GenOctreeNode& lhs, const GenOctreeNode& rhs) { return lhs.MortonId < rhs.MortonId; });

    for(uint32_t node_idx = 0; node_idx < point_count - 1; ++node_idx)
    {
        TGE_ASSERT(gen_nodes[node_idx].MortonId != gen_nodes[node_idx + 1].MortonId, "unable to generate octree");
        if(gen_nodes[node_idx].MortonId == gen_nodes[node_idx + 1].MortonId)
            return {};
    }

    TGE_ASSERT((gen_nodes[point_count - 1].MortonId & 3u << 30u) == 0, "Invalid morton codes");

    const uint32_t worst_case_count = 2*point_count - 1;
    Octree octree;
    octree.Bounds = set_bounds;
    octree.Hierarchy = new OctreeNode[worst_case_count];
    uint32_t tree_size = 0;
    auto cur_node = GenerateOctreeHierarchy(29, gen_nodes.get(), 0, point_count - 1, octree.Hierarchy, &tree_size);
    TGE_ASSERT(cur_node == 0 && tree_size <= worst_case_count, "Broken tree generator");
    //octree.Hierarchy[cur_node].Parent = OCTREE_INVALID_NODE;

    return std::move(octree);
}

// TODO: Traversal trace
/*
inline EXPORT_CUDA uint32_t RollbackAndSearch(const OctreeNode* octree, uint32_t node_id)
{
	// Rollback and search for second child along the hierarchy that was not traversed
    uint32_t parent = octree[node_id].Parent;

    while(parent != OCTREE_INVALID_NODE)
    {
        auto& parent_node = octree[parent];
        for(uint32_t idx = 0; idx < 8; ++idx)
        {
            auto child = parent_node.Children[idx];
            if(child == node_id && idx != 7)
                return parent_node.Children[idx + 1];
        }

        node_id = parent;
        parent = octree[node_id].Parent;
    }

    return OCTREE_INVALID_NODE;
}
*/

bool CheckOctreeNode(OctreeNode* octree, uint32_t node_id, const AABBUnaligned& box, const Tempest::Vector3* points, uint32_t point_count, uint8_t* occurance)
{
    auto extend = (box.MaxCorner - box.MinCorner)*0.5f; 

    auto& cur_node = octree[node_id];

    uint32_t node_count = 0;
    for(uint32_t idx = 0; idx < TGE_FIXED_ARRAY_SIZE(octree->Children); ++idx)
    {
        auto child_id = cur_node.Children[idx];
        
        if(child_id == OCTREE_INVALID_NODE)
            continue;

        uint32_t x = (idx     ) & 1,
                 y = (idx >> 1) & 1,
                 z = (idx >> 2) & 1;

        AABBUnaligned cur_box { box.MinCorner + Tempest::Vector3{ (float)x, (float)y , (float)z }*extend,
                                box.MinCorner + Tempest::Vector3{ (float)x + 1.0f, (float)y + 1.0f, (float)z + 1.0f }*extend }; 

        if(child_id & OCTREE_LEAF_NODE_DECORATION)
        {
            child_id &= (~OCTREE_LEAF_NODE_DECORATION);
            if(child_id >= point_count)
                return false;
            auto& point = points[child_id];
            occurance[child_id]++;
            if(!(cur_box.MinCorner <= point && point <= cur_box.MaxCorner))
                return false;
        }
        else if(!CheckOctreeNode(octree, child_id, cur_box, points, point_count, occurance))
            return false;

        if(child_id != OCTREE_INVALID_NODE)
            ++node_count;
    }
    if(node_count == 0 || node_count > 8)
        return false;

    return true;
}

bool CheckOctree(const Octree& octree, const Tempest::Vector3* points, uint32_t point_count)
{
    std::unique_ptr<uint8_t[]> occurance(new uint8_t[point_count]);
    memset(occurance.get(), 0, point_count*sizeof(occurance[0]));
    bool status = CheckOctreeNode(octree.Hierarchy, 0, octree.Bounds, points, point_count, occurance.get());
    if(!status)
        return false;

    for(uint32_t idx = 0; idx < point_count; ++idx)
    {
        if(occurance[idx] != 1)
            return false;
    }

    return true;
}
}
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

#ifndef _TEMPEST_OCTREE_HH_
#define _TEMPEST_OCTREE_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/shapes.hh"
#include "tempest/math/intersect.hh"

namespace Tempest
{
#define OCTREE_LEAF_NODE_DECORATION (1u << 31u)
#define OCTREE_INVALID_NODE (~0u)

struct OctreeNode
{
 //   uint32_t Parent;
    uint32_t Children[8];
};

struct Octree
{
    AABBUnaligned Bounds = AABBUnaligned{};
    OctreeNode*   Hierarchy = nullptr;

    Octree()=default;

    Octree(Octree&& oc)
        :   Bounds(oc.Bounds)
    {
        Hierarchy = oc.Hierarchy;
        oc.Hierarchy = {};
    }

    Octree& operator=(Octree&& oc)
    {
        Bounds = oc.Bounds;
        Hierarchy = oc.Hierarchy;
        oc.Hierarchy = {};
        return *this;
    }

    Octree(const Octree&)=delete;
    Octree& operator=(const Octree&)=delete;

    ~Octree() { delete[] Hierarchy; }
};

Octree BuildOctreeMorton(const Tempest::Vector3* points, uint32_t point_count);

template<class TProcess>
void OctreeProcess(const OctreeNode* octree, uint32_t node_id, const Tempest::Vector3* points, uint32_t point_count, TProcess& process)
{
    auto& cur_node = octree[node_id];

    uint32_t node_count = 0;
    for(uint32_t idx = 0; idx < TGE_FIXED_ARRAY_SIZE(octree->Children); ++idx)
    {
        auto child_id = cur_node.Children[idx];
        
        if(child_id == OCTREE_INVALID_NODE)
            continue;

        if(child_id & OCTREE_LEAF_NODE_DECORATION)
        {
            child_id &= (~OCTREE_LEAF_NODE_DECORATION);
            process(child_id);
        }
        else
        {
            OctreeProcess(octree, child_id, points, point_count, process);
        }
    }
}

template<class TProcess>
void OctreeIntersectImpl(const OctreeNode* octree, uint32_t node_id, const Tempest::Vector3* points, uint32_t point_count, const Tempest::AABBUnaligned& root_box, const Tempest::AABBUnaligned& intersection_box, TProcess& process)
{
    auto extend = (root_box.MaxCorner - root_box.MinCorner)*0.5f; 

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

        AABBUnaligned cur_box { root_box.MinCorner + Tempest::Vector3{ x, y , z }*extend,
                                root_box.MinCorner + Tempest::Vector3{ x + 1, y + 1, z + 1 }*extend }; 

        if(child_id & OCTREE_LEAF_NODE_DECORATION)
        {
            child_id &= (~OCTREE_LEAF_NODE_DECORATION);
            auto& point = points[child_id];

            if(intersection_box.MinCorner <= point && point <= intersection_box.MaxCorner)
                process(child_id);
        }
        else
        {
            if(intersection_box.MinCorner <= cur_box.MinCorner &&
               cur_box.MaxCorner <= intersection_box.MaxCorner)
            {
                OctreeProcess(octree, child_id, points, point_count, process);
            }
            else if(IntersectAABBAABB(cur_box, intersection_box))
            {
                OctreeIntersectImpl(octree, child_id, points, point_count, cur_box, intersection_box, process);
            }
        }
    }
}

template<class TProcess>
void OctreeIntersect(const Octree& octree, const Tempest::Vector3* points, uint32_t point_count, const Tempest::AABBUnaligned& box, const TProcess& process)
{
    OctreeIntersectImpl(octree.Hierarchy, 0, points, point_count, octree.Bounds, box, process);
}

bool CheckOctree(const Octree& octree, const Tempest::Vector3* points, uint32_t point_count);
}

#endif // _TEMPEST_OCTREE_HH_
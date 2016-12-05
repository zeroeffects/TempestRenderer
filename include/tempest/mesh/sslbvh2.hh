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

#ifndef _TEMPEST_SSLBVH2_HH_
#define _TEMPEST_SSLBVH2_HH_

#include "tempest/math/shape-split.hh"
#include "tempest/mesh/lbvh2.hh"

namespace Tempest
{
#define SSLBVH_LEAF_DECORATION (1u << 31u)
#define SSLBVH_INVALID_NODE ~0u

template<class TAABB>
struct SimpleStacklessLBVH2Node
{
    TAABB    Bounds;
	uint32_t Parent;
    union
    {
        uint32_t Child2;
        uint32_t Patch;
    };
};

template<class TAABB>
inline EXPORT_CUDA uint32_t RollbackAndSearch(const SimpleStacklessLBVH2Node<TAABB>* bvh, uint32_t node_id)
{
	// Rollback and search for second child along the hierarchy that was not traversed
    uint32_t parent = bvh[node_id].Parent;

    while(parent != SSLBVH_INVALID_NODE)
    {
		if(parent + 1 == node_id)
			return bvh[parent].Child2;

        node_id = parent;
        parent = bvh[node_id].Parent;
    }

    return SSLBVH_INVALID_NODE;
}

template<class TAABB, class TFunc, class TIntersectData>
inline EXPORT_CUDA void IntersectSSLBVHNode(const SimpleStacklessLBVH2Node<TAABB>* bvh, const TIntersectData& pos, TFunc& leaf_intersect_func)
{
    uint32_t node_id = 0;

    for(;;)
    {
        auto& cur_node = bvh[node_id];
        if(!IntersectPrimBVH(pos, cur_node.Bounds))                                 
        {
			node_id = RollbackAndSearch(bvh, node_id);
			if(node_id == SSLBVH_INVALID_NODE)
				return;
            
			continue;
        }

        if(cur_node.Patch & SSLBVH_LEAF_DECORATION)
        {
            leaf_intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);
            
			node_id = RollbackAndSearch(bvh, node_id);
			if(node_id == SSLBVH_INVALID_NODE)
				return;
        }
        else
        {
            ++node_id; // We traverse the first child
        }
    }
}

template<class TAABB, class TFunc, class TIntersectData>
inline EXPORT_CUDA bool IntersectSSLBVHNodeSingle(const SimpleStacklessLBVH2Node<TAABB>* bvh, TIntersectData& pos, TFunc& leaf_intersect_func)
{
    uint32_t node_id = 0;

    for(;;)
    {
        auto& cur_node = bvh[node_id];
        if(!IntersectPrimBVH(pos, cur_node.Bounds))                                 
        {
			node_id = RollbackAndSearch(bvh, node_id);
			if(node_id == SSLBVH_INVALID_NODE)
				return false;
            
			continue;
        }

        if(cur_node.Patch & SSLBVH_LEAF_DECORATION)
        {
            auto intersect = leaf_intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);
            if(intersect)
                return true;

            node_id = RollbackAndSearch(bvh, node_id);
			if(node_id == SSLBVH_INVALID_NODE)
				return false;
        }
        else
        {
            ++node_id; // We traverse the first child
        }
    }
}

// Because we are basically splitting a binary tree. There is a quite easy to compute upper bound
inline uint32_t SSLBVHMaxNodeCount(uint32_t total_node_count) { return 2*total_node_count - 1; }

template<class TAABB>
SimpleStacklessLBVH2Node<TAABB>* GenerateSSLBVH(LBVH2Node<TAABB>* interm_nodes, uint32_t total_node_count);

template<class TAABB>
void GenerateSSLBVH(LBVH2Node<TAABB>* interm_nodes, uint32_t total_node_count, SimpleStacklessLBVH2Node<TAABB>* out_nodes);
}

#endif // _TEMPEST_SSLBVH2_HH_
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

#ifndef _TEMPEST_ESSBVH2_HH_
#define _TEMPEST_ESSBVH2_HH_

#include "tempest/math/shape-split.hh"
#include "tempest/mesh/lbvh2.hh"

namespace Tempest
{
#define ESSBVH_LEAF_DECORATION (1u << 31u)
#define ESSBVH_INVALID_NODE ~0u

template<class TAABB>
struct EditableSSBVH2Node
{
    TAABB    Bounds;
	uint32_t Parent;
    union
    {
        uint32_t Child[2];
        uint32_t Patch;
    };
};

template<class TNode>
union AllocationNode
{
    uint32_t     NextFree;
    TNode        Node;
};

template<class TAABB>
inline EXPORT_CUDA uint32_t RollbackAndSearch(const EditableSSBVH2Node<TAABB>* bvh, uint32_t node_id)
{
	// Rollback and search for second child along the hierarchy that was not traversed
    TGE_ASSERT(node_id != ESSBVH_INVALID_NODE, "Invalid node");
    uint32_t parent = bvh[node_id].Parent;

    while(parent != ESSBVH_INVALID_NODE)
    {
        if(parent + 1 == node_id)
            return bvh[parent].Child[1];

        node_id = parent;
        parent = bvh[node_id].Parent;
    }

    return ESSBVH_INVALID_NODE;
}

template<class TAABB, class TFunc, class TIntersectData>
inline EXPORT_CUDA void IntersectESSBVHNode(const EditableSSBVH2Node<TAABB>* bvh, const TIntersectData& pos, TFunc& leaf_intersect_func)
{
    uint32_t node_id = 0;

    for(;;)
    {
        auto& cur_node = bvh[node_id];
        if(!IntersectPrimBVH(pos, cur_node.Bounds))
        {
            node_id = RollbackAndSearch(bvh, node_id);
            if(node_id == ESSBVH_INVALID_NODE)
                return;

            continue;
        }

        if(cur_node.Patch & ESSBVH_LEAF_DECORATION)
        {
            leaf_intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);

            node_id = RollbackAndSearch(bvh, node_id);
            if(node_id == ESSBVH_INVALID_NODE)
                return;
        }
        else
        {
            node_id = cur_node.Child[0];
        }
    }
}

template<class TAABB>
inline uint32_t SizeOfESSBVH(const EditableSSBVH2Node<TAABB>* bvh)
{
    uint32_t size = 0,
             node_id = 0;

    for(;;)
    {
        auto& cur_node = bvh[node_id];
        if(cur_node.Patch & ESSBVH_LEAF_DECORATION)
        {
            ++size;

            node_id = RollbackAndSearch(bvh, node_id);
            if(node_id == ESSBVH_INVALID_NODE)
                break;
        }
        else
        {
            node_id = cur_node.Child[0];
        }
    }
    return size;
}

template<class TAABB, class TFunc, class TIntersectData>
inline EXPORT_CUDA bool IntersectESSBVHNodeSingle(const EditableSSBVH2Node<TAABB>* bvh, TIntersectData& pos, TFunc& leaf_intersect_func)
{
    uint32_t node_id = 0;

    for(;;)
    {
        auto& cur_node = bvh[node_id];
        if(!IntersectPrimBVH(pos, cur_node.Bounds))
        {
            node_id = RollbackAndSearch(bvh, node_id);
            if(node_id == ESSBVH_INVALID_NODE)
                return false;

            continue;
        }

        if(cur_node.Patch & ESSBVH_LEAF_DECORATION)
        {
            auto intersect = leaf_intersect_func((cur_node.Patch & ~LBVH_LEAF_DECORATION), pos);
            if(intersect)
                return true;

            node_id = RollbackAndSearch(bvh, node_id);
            if(node_id == ESSBVH_INVALID_NODE)
                return false;
        }
        else
        {
            node_id = cur_node.Child[0];
        }
    }
}

template<class TAABB>
void GenerateESSBVH(LBVH2Node<TAABB>* interm_nodes, uint32_t total_node_count, EditableSSBVH2Node<TAABB>* out_tree);

template<class TAABB, class TData>
class EditableSSBVH2Iterator
{
    EditableSSBVH2Node<TAABB>* m_Nodes;
    EditableSSBVH2Node<TAABB>* m_CurrentNode = nullptr;
    TData*                     m_Data;
public:
    EditableSSBVH2Iterator()=default;

    EditableSSBVH2Iterator(EditableSSBVH2Node<TAABB>* nodes, TData* data)
        :   m_Nodes(nodes),
            m_Data(data)
    {
        if(m_Nodes && m_Nodes->Child[0] != ESSBVH_INVALID_NODE)
        {
            for(m_CurrentNode = m_Nodes; (m_CurrentNode->Patch & ESSBVH_LEAF_DECORATION) == 0; m_CurrentNode = m_Nodes + m_CurrentNode->Child[0])
                ;
        }
    }

    bool operator!=(const EditableSSBVH2Iterator& other) const { return m_CurrentNode != other.m_CurrentNode; }

    bool operator==(const EditableSSBVH2Iterator& other) const { return m_CurrentNode == other.m_CurrentNode; }

    EditableSSBVH2Iterator& operator++(int)
    {
        EditableSSBVH2Iterator old_iter(*this);
        if(!m_CurrentNode)
        {
            return old_iter;
        }

        auto node_id = RollbackAndSearch(m_Nodes, static_cast<uint32_t>(m_CurrentNode - m_Nodes));
        if(node_id == ESSBVH_INVALID_NODE)
        {
            m_CurrentNode = nullptr;
            return old_iter;
        }

        for(m_CurrentNode = m_Nodes + node_id; (m_CurrentNode->Patch & ESSBVH_LEAF_DECORATION) == 0; m_CurrentNode = m_Nodes + m_CurrentNode->Child[0])
            ;
        return old_iter;
    }

    EditableSSBVH2Iterator operator++()
    {
        if(!m_CurrentNode)
        {
            return *this;
        }

        auto node_id = RollbackAndSearch(m_Nodes, static_cast<uint32_t>(m_CurrentNode - m_Nodes));
        if(node_id == ESSBVH_INVALID_NODE)
        {
            m_CurrentNode = nullptr;
            return *this;
        }

        for(m_CurrentNode = m_Nodes + node_id; (m_CurrentNode->Patch & ESSBVH_LEAF_DECORATION) == 0; m_CurrentNode = m_Nodes + m_CurrentNode->Child[0])
            ;
        return *this;
    }

    TData& operator*()
    {
        return m_Data[m_CurrentNode->Patch & ~ESSBVH_LEAF_DECORATION];
    }

    TData* operator->()
    {
        return m_Data + (m_CurrentNode->Patch & ~ESSBVH_LEAF_DECORATION);
    }
};

template<class TAABB, class TData>
struct NaiveEditableSSBVH2
{
    typedef AllocationNode<EditableSSBVH2Node<TAABB>> TreeNode;
    uint32_t                                     m_ElementCount = 0;
    uint32_t                                     m_Capacity = 0;

    EditableSSBVH2Node<TAABB>*                   m_Tree;
    LBVH2Node<TAABB>*                            m_IntermNodes;
    TData*                                       m_Data;

    TAABB                                        m_Bounds;
public:
    NaiveEditableSSBVH2(const TAABB& bounds, uint32_t capacity = 0)
        :   m_Capacity(capacity ? capacity : 256),
            m_Tree(new EditableSSBVH2Node<TAABB>[2*m_Capacity - 1]),
            m_IntermNodes(new LBVH2Node<TAABB>[m_Capacity]),
            m_Data(new TData[m_Capacity]),
            m_Bounds(bounds)
    {
        memset(m_Tree, 0, (2*m_Capacity - 1)*sizeof(EditableSSBVH2Node<TAABB>));
        m_Tree->Patch = ESSBVH_INVALID_NODE;
    }

    ~NaiveEditableSSBVH2()
    {
        delete m_Tree;
        delete m_IntermNodes;
        delete m_Data;
    }

    TData* getData() { return m_Data; }
    EditableSSBVH2Node<TAABB>* getNodes() { return m_ElementCount ? m_Tree : nullptr; }
    uint32_t getNodeCount() { return 2*m_Capacity - 1; }

    EditableSSBVH2Iterator<TAABB, TData> begin() { return EditableSSBVH2Iterator<TAABB, TData>(m_Tree, m_Data); }

    EditableSSBVH2Iterator<TAABB, TData> end() { return EditableSSBVH2Iterator<TAABB, TData>(); }

    inline uint32_t size() const
    {
        return m_ElementCount;
    }

    inline void insert(const TData& data, const TAABB& bounds)
    {
        if(m_ElementCount == m_Capacity)
        {
            auto old_capacity = m_Capacity;
            m_Capacity *= 2;
            
            delete m_Tree;
            m_Tree = new EditableSSBVH2Node<TAABB>[2*m_Capacity - 1];
            memset(m_Tree, 0, (2*m_Capacity - 1)*sizeof(EditableSSBVH2Node<TAABB>));
            auto new_data = new TData[m_Capacity];
            memcpy(new_data, m_Data, old_capacity*sizeof(TData));
            delete m_Data;
            m_Data = new_data;
            auto new_interm_nodes = new LBVH2Node<TAABB>[m_Capacity];
            memcpy(new_interm_nodes, m_IntermNodes, old_capacity*sizeof(LBVH2Node<TAABB>));
            delete m_IntermNodes;
            m_IntermNodes = new_interm_nodes;
        }

        m_Data[m_ElementCount] = data;
        auto& interm_node = m_IntermNodes[m_ElementCount];
        interm_node.Bounds = bounds;
        interm_node.Patch = ESSBVH_LEAF_DECORATION | m_ElementCount;

        ++m_ElementCount;

        GenerateESSBVH(m_IntermNodes, m_ElementCount, m_Tree);
    }

    inline void assign(const TData* data_arr, const LBVH2Node<TAABB>* bounds_arr, uint32_t elem_count)
    {
        m_ElementCount = elem_count;
        if(elem_count > m_Capacity)
        {
            m_Capacity = 2*elem_count;

            delete m_Tree;
            uint32_t tree_size = 2*m_Capacity - 1;
            m_Tree = new EditableSSBVH2Node<TAABB>[tree_size];
            memset(m_Tree, 0, tree_size*sizeof(EditableSSBVH2Node<TAABB>));

            delete m_Data;
            m_Data = new TData[m_Capacity];

            delete m_IntermNodes;
            m_IntermNodes = new LBVH2Node<TAABB>[m_Capacity];
        }

        if(elem_count)
        {
            memcpy(m_Data, data_arr, elem_count*sizeof(TData));
            memcpy(m_IntermNodes, bounds_arr, elem_count*sizeof(LBVH2Node<TAABB>));
            GenerateESSBVH(m_IntermNodes, m_ElementCount, m_Tree);
        }
    }

    void clear()
    {
        m_ElementCount = 0;
        m_Tree->Patch = ESSBVH_INVALID_NODE;
    }
};


template<class TAABB, class TData>
struct EditableSSBVH2
{
    typedef AllocationNode<EditableSSBVH2Node<TAABB>> TreeNode;
    uint32_t    m_Capacity = 0;
    uint8_t*    m_Data;
    
    TAABB       m_Bounds;
    TreeNode*   m_NodeFreeList;
public:
    EditableSSBVH2(const TAABB& bounds, uint32_t count = 0)
        :   m_Capacity(count ? count : 256),
            m_Data(reinterpret_cast<uint8_t*>(malloc((2*m_Capacity - 1)*sizeof(EditableSSBVH2Node<AABB2>) + m_Capacity*sizeof(TData)))),
            m_Bounds(bounds)
    {
        static_assert(sizeof(TData) >= sizeof(uint32_t), "Invalid data size");

        auto top = reinterpret_cast<TreeNode*>(m_Data);
        m_NodeFreeList = top;
        for(uint32_t idx = 0; idx < m_Capacity - 1; ++idx)
        {
            m_NodeFreeList[idx].NextFree = idx + 1;
        }
        m_NodeFreeList[m_Capacity - 1].NextFree = ESSBVH_INVALID_NODE;
    }

    ~EditableSSBVH2()
    {
        free(m_Data);
    }

    EditableSSBVH2Node<TAABB>* getNodes() { auto top = reinterpret_cast<TreeNode*>(m_Data); return m_NodeFreeList != top ? &top->Node : nullptr; }
    TData* getData() { auto scratches = reinterpret_cast<TData*>(m_Data + m_Capacity*sizeof(TreeNode)); return scratches;  }

    EditableSSBVH2Iterator<TAABB, TData> begin() { auto top = reinterpret_cast<TreeNode*>(m_Data); return EditableSSBVH2Iterator<TAABB, TData>(&top->Node, reinterpret_cast<TData*>(top + m_Capacity)); }

    EditableSSBVH2Iterator<TAABB, TData> end() { return EditableSSBVH2Iterator<TAABB, TData>(); }

    inline uint32_t size() const
    {
        auto top = reinterpret_cast<TreeNode*>(m_Data);
        return m_NodeFreeList != top ? SizeOfESSBVH(&reinterpret_cast<TreeNode*>(m_Data)->Node) : 0;
    }

    inline void insert(const TData& data, const TAABB& bounds)
    {
        auto top = reinterpret_cast<TreeNode*>(m_Data);
        if(m_NodeFreeList == top)
        {
            *top = data;
            m_NodeFreeList = top + m_NodeFreeList->NextFree;
        }

        uint32_t insert_id = ComputeAABBMortonCode(bounds, m_Bounds),
                 cur_id = ComputeAABBMortonCode(top->Bounds, m_Bounds);

        TreeNode* nodes = reinterpret_cast<TreeNode*>(m_Data),
                * cur_node = nodes;

        for(;;)
        {
            auto sub_idx = 0;
            
            if((cur_node->Patch & ESSBVH_LEAF_DECORATION) == 0)
            {
                uint32_t child_ids[2] = { ComputeAABBMortonCode(nodes[cur_node->Child[0]]->Bounds),
                                          ComputeAABBMortonCode(nodes[cur_node->Child[1]]->Bounds) };

                auto diff = child_ids[0] ^ child_ids[1];

                auto lz_diff = __lzcnt(diff);
                if(lz_diff != 32)
                {
                    sub_idx = (insert_id >> (32 - lz_diff)) & 1;

                    auto child_id = child_ids[sub_idx];

                    auto header_diff = (child_ids[sub_idx] ^ insert_id) & ((1 << (32 - lz_diff)) - 1);
                    if(header_diff == 0)
                    {
                        cur_node = top + cur_node->Child[sub_idx];
                        cur_id = child_id;
                        continue;
                    }
                }
            }
            else
            {
                auto diff = cur_id ^ insert_id;
                auto lz_diff = __lzcnt(diff);
                sub_idx = (insert_id >> (32 - lz_diff)) & 1;
            }

            auto other_sub_idx = 1 - sub_idx;

            auto replace_pos = insertNode();
            auto replace_node = top + replace_pos;
            *replace_node = *cur_node;
            replace_node->Parent = cur_node - top;
            cur_node->Child[other_sub_idx] = replace_pos;
            cur_node->Bounds = { GenericMin(bounds.MinCorner, cur_node->Bounds.MinCorner),
                                    GenericMax(bounds.MaxCorner, cur_node->Bounds.MaxCorner) };
            auto insert_pos = cur_node->Child[sub_idx] = insertNode();
            auto insert_node = top + insert_pos;
            insert_node->Bounds = bounds;
            auto data = reinterpret_cast<TData*>(top + m_Capacity); // TODO: Make it more compact
            insert_node->Patch = ESSBVH_LEAF_DECORATION | insert_pos; 
            break;
        }
    }

private:
    inline void eraseNode(uint32_t idx)
    {
        auto top = reinterpret_cast<TreeNode*>(m_Data);
        // We do keep nodes sorted for improved cache locality
        if(m_NodeFreeList == nullptr)
        {
            m_NodeFreeList = top + idx;
        }
        else if(m_NodeFreeList - top > idx)
        {
            auto prev_node_idx = m_NodeFreeList - top;
            auto cur_node = m_NodeFreeList = top + idx;
            cur_node->NextNode = prev_node_idx;
        }
        else
        {
            auto cur_node = m_NodeFreeList;
            while(cur_node->NextNode > idx)
            {
                if(cur_node->NextNode == ESSBVH_INVALID_NODE)
                {
                    cur_node->NextNode = idx;
                    top[idx].NextNode = ESSBVH_INVALID_NODE;
                    return;
                }
                cur_node = top + cur_node->NextNode;
            }
            TGE_ASSERT(cur_node - top != idx, "Double delete");
            auto next_node = cur_node->NextNode;
            cur_node->NextNode = idx;
            top[idx].NextNode = next_node;
        }
    }

    size_t insertNode()
    {
        auto top = reinterpret_cast<TreeNode*>(m_Data);
        if(!m_NodeFreeList)
        {
            auto old_data = m_Data;
            auto old_capacity = m_Capacity;
            m_Capacity *= 2;

            m_Data = reinterpret_cast<uint8_t*>(malloc(m_Capacity*(sizeof(TreeNode) + sizeof(TData))));
            memcpy(m_Data, old_data, old_capacity*sizeof(TreeNode));
            memcpy(m_Data + m_Capacity*sizeof(TreeNode), old_data + old_capacity*sizeof(TreeNode), old_capacity*sizeof(TData));
            free(old_data);

            m_NodeFreeList = reinterpret_cast<TreeNode*>(m_Data) + old_capacity;

            for(uint32_t idx = old_capacity; idx < m_Capacity - 1; ++idx)
            {
                m_NodeFreeList[idx].NextFree = idx + 1;
            }
            m_NodeFreeList[m_Capacity - 1].NextFree = ESSBVH_INVALID_NODE;
        }

        auto insert_pos = m_NodeFreeList - top;
        auto next_node = m_NodeFreeList->NextFree;
        m_NodeFreeList = next_node != ESSBVH_INVALID_NODE ? top + next_node : nullptr;

        return insert_pos;
    }
};
}

#endif // _TEMPEST_ESSBVH2_HH_

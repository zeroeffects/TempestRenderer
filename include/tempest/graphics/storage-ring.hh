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

#include "tempest/graphics/preferred-backend.hh"

namespace Tempest
{
template<class TBackend>
class StorageRing
{
public:
    typedef typename TBackend::StorageType StorageType;
    typedef typename TBackend::FenceType FenceType;

private:
    TBackend*       m_Backend;
    StorageType*    m_Storage;
    FenceType*      m_Fence;

    uint32_t        m_WriteStart = 0,
                    m_Border = 0,
                    m_Index = 0;
public:
    static const uint32_t InvalidIndex = ~0u;

    StorageRing(TBackend* backend, uint32_t storage_size)
        :   m_Backend(backend)
    {
        storage_size = NextPowerOf2(storage_size);

        TGE_ASSERT(sizeof(uint32_t) % storage_size, "Invalid size");

        m_Storage = m_Backend->createStorageBuffer(StorageMode::BufferWrite, storage_size);
        m_Fence = m_Backend->createFence();
    }

    ~StorageRing() { m_Backend->destroyRenderResource(m_Storage); }

    StorageType* getStorage() { return m_Storage; }

    uint32_t getSize() const { return m_Storage->getSize(); }

    uint32_t tryPushData(void* data, uint32_t size)
    {
        if(size == 0)
            return m_Index;

        auto storage_size = m_Storage->getSize();
        auto start_index = m_Index;
        auto next_index = start_index + size;
        auto wrap_index = (next_index - 1)/storage_size; 
        if(wrap_index != start_index/storage_size)
        {
            start_index = wrap_index*storage_size;
            next_index = start_index + size;
        }

        if(next_index > m_Border + storage_size)
        {
            return InvalidIndex;
        }

        start_index %= storage_size;

        m_Storage->storeLinearBuffer(start_index, size, data);

        m_Index = next_index;

        return start_index;
    }

    uint32_t pushData(void* data, uint32_t size)
    {
        if(size > m_Storage->getSize())
            return InvalidIndex;

        auto index = tryPushData(data, size);
        if(index != InvalidIndex)
            return index;
        
        flip();
        index = tryPushData(data, size);
        if(index != InvalidIndex)
            return index;

        clear();
        return tryPushData(data, size);
    }

    void clear()
    {
        if(m_WriteStart - m_Border)
        {
            m_Backend->waitFence(m_Fence);
        }
        if(m_Index - m_WriteStart)
        {
            m_Backend->pushFence(m_Fence);
            m_Backend->waitFence(m_Fence);
        }
        m_Border = m_Index = m_WriteStart = 0;
    }

    void flip()
    {
        auto storage_size = m_Storage->getSize();
        int prev_size = m_Border - m_WriteStart;
        TGE_ASSERT(0 <= prev_size && prev_size <= (int)storage_size, "Invalid size");

        if(prev_size > 0)
        {
            m_Backend->waitFence(m_Fence);
        }

        m_Border = m_WriteStart;

        int cur_size = m_Index - m_WriteStart;
        TGE_ASSERT(0 <= cur_size && cur_size <= (int)storage_size, "Invalid size");

        if(cur_size > 0)
        {
            m_Backend->pushFence(m_Fence);
            m_WriteStart = m_Index;
        }
        else
        {
            m_Border = m_Index = m_WriteStart = 0;
        }
    }
};
}
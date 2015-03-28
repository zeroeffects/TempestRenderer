/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2014 Zdravko Velinov
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

#include "tempest/texture/texture-table.hh"
#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/utils/memory.hh"

#include <numeric>

namespace Tempest
{
struct SlotTraitDescription
{
    DataFormat      Format;
    TextureTiling   Tiling;
    uint16          Edge;
};

#define TEMPEST_TEXTURE_TABLE_SLOT(name, format, tiling, edge) { format, tiling, edge },

static const SlotTraitDescription SlotTraits[] =
{
    TEMPEST_TEXTURE_TABLE_SLOTS
};

#undef TEMPEST_TEXTURE_TABLE_SLOT

template<class TBackend> TextureTable<TBackend>::TextureTable(TBackend* backend, const TextureTableDescription& desc)
    :   m_Backend(backend),
        m_UploadHeap(backend->createStorageBuffer(StorageMode::PixelUnpack, desc.UploadHeapSize)),
        m_IOCommandBuffer(backend->createIOCommandBuffer(IOCommandBufferDescription{ desc.UploadQueueSize })),
        m_UploadHeapSize(desc.UploadHeapSize),
        m_BakedTable(new BakedResourceTable(4*sizeof(GLfloat)*TEMPEST_TEXTURE_SLOTS)) // aligned to 4*sizeof(float)
{
    std::fill(std::begin(m_Fence), std::end(m_Fence), nullptr);
    
    char* baked_table_ptr = m_BakedTable->get();

    memset(m_BakedTable->get(), 0, 4*sizeof(GLfloat)*TEMPEST_TEXTURE_SLOTS);

    TextureDescription tex_desc;
    for(size_t i = 0; i < TEMPEST_TEXTURE_SLOTS; ++i)
    {
        auto& trait = SlotTraits[i];
        tex_desc.Format = trait.Format;
        tex_desc.Width = tex_desc.Height = trait.Edge;
        tex_desc.Tiling = trait.Tiling;
        auto& array_desc = m_Textures[i];
        array_desc.LastSlot = 0;
        array_desc.SlotCount = tex_desc.Depth = desc.Slots[i];
        auto* tex = array_desc.Texture = backend->createTexture(tex_desc);
        tex->setFilter(Tempest::FilterMode::Linear, Tempest::FilterMode::Linear, Tempest::FilterMode::Linear);
        tex->setWrapMode(Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp);
        *reinterpret_cast<uint64*>(baked_table_ptr) = tex->getHandle();
        baked_table_ptr += 4*sizeof(GLfloat); // aligned
    }

    m_UploadHeapBoundary[0] = 0;
    std::fill(std::begin(m_UploadHeapBoundary) + 1, std::end(m_UploadHeapBoundary), desc.UploadHeapSize);
    std::generate(std::begin(m_Fence), std::end(m_Fence), [backend]() { return backend->createFence(); });
}

template<class TBackend> TextureTable<TBackend>::~TextureTable()
{
    clearPendingTextures();
    m_Backend->destroyRenderResource(m_IOCommandBuffer);
    m_Backend->destroyRenderResource(m_UploadHeap);
    for(size_t i = 0; i < TEMPEST_TEXTURE_SLOTS; ++i)
    {
        m_Backend->destroyRenderResource(m_Textures[i].Texture);
    }
    for(auto* fence : m_Fence)
    {
        m_Backend->destroyRenderResource(fence);
    }
}

template<class TStorage, class TIOCommandBuffer, class TTextureArray>
static bool InsertIntoStorage(Texture* tex, uint32 tex_size, uint32 heap_size, uint32 buf_idx,
                              int32* upload_heap_boundaries, TStorage* upload_heap, TIOCommandBuffer* io_cmd_buf,
                              uint32 slot, TTextureArray* tex_array)
{
    auto next_buf_idx = (buf_idx + 1) % TEMPTEST_TEXTURE_TABLE_BUFFER_COUNT;
    auto boundary_idx = upload_heap_boundaries[buf_idx];
    auto next_boundary_idx = upload_heap_boundaries[next_buf_idx];
    int32 boundary_after_idx;
    auto start_offset = boundary_idx % heap_size;
    if(boundary_idx % heap_size + tex_size > heap_size)
    {
        boundary_after_idx = boundary_idx + heap_size + tex_size;
        start_offset = 0;
    }
    else
    {
        boundary_after_idx = boundary_idx + tex_size;
    }
    if(next_boundary_idx - boundary_after_idx < 0)
    {
        return false;
    }

    auto& hdr = tex->getHeader();

    GLIOCommand cmd;
    cmd.Source.Storage = upload_heap;
    cmd.Destination.Texture = tex_array;
    cmd.CommandType = IOCommandMode::CopyStorageToTexture;
    cmd.SourceOffset = start_offset;
    cmd.DestinationSlice = slot;
    cmd.Width = hdr.Width;
    cmd.Height = hdr.Height;
    // So if we can't enqueue more commands. We should perform a full on flush to proceed.
    if(!io_cmd_buf->enqueueCommand(cmd))
        return false;

    upload_heap->storeTexture(start_offset, tex->getHeader(), tex->getData());
    upload_heap_boundaries[buf_idx] = boundary_after_idx;
    return true;
}

// TODO: direct copy, if the buffer has enough space
// TODO: End of buffer fragmentation?
template<class TBackend> Vector4 TextureTable<TBackend>::loadTexture(const Path& filename)
{
    std::unique_ptr<Texture> tex(LoadImage(filename));
    if(!tex)
        return InvalidSlot;
    tex->convertToRGBA();
    auto& hdr = tex->getHeader();
    uint32 tex_size = hdr.Width*hdr.Height*DataFormatElementSize(hdr.Format);
    // Don't even bother
    TGE_ASSERT(tex_size <= m_UploadHeapSize, "Upload heap too small");
    if(tex_size > m_UploadHeapSize)
        return InvalidSlot;

    auto biggest_side = std::max(hdr.Width, hdr.Height);
    uint32 best_slot = 0, best_fit = std::numeric_limits<uint32>::max();
    for(uint32 i = 0; i < TEMPEST_TEXTURE_SLOTS; ++i)
    {
        auto& slot_trait = SlotTraits[i];
        if(hdr.Format != slot_trait.Format ||
           biggest_side > slot_trait.Edge)
            continue;
        if(slot_trait.Edge < best_fit)
        {
            best_fit = slot_trait.Edge;
            best_slot = i;
        }
    }

    if(best_fit == std::numeric_limits<uint32>::max())
    {
        return InvalidSlot;
    }

    auto& slot_trait = SlotTraits[best_slot];
    auto& subtable = m_Textures[best_slot];
    auto slice = subtable.LastSlot;
    if(slice == subtable.SlotCount)
    {
        return InvalidSlot;
    }
    ++subtable.LastSlot;

    if(!InsertIntoStorage(tex.get(), tex_size, m_UploadHeapSize, m_BufferIndex, m_UploadHeapBoundary, m_UploadHeap, m_IOCommandBuffer,
                          slice, m_Textures[best_slot].Texture))
    {
        m_PendingTextures.push_back(PendingTexture{ best_slot, slice, tex.release() });
    }

    return Vector4(static_cast<float>(hdr.Width) / slot_trait.Edge, static_cast<float>(hdr.Height) / slot_trait.Edge,
                   static_cast<float>(best_slot), static_cast<float>(slice));
}

template<class TBackend> void TextureTable<TBackend>::executeIOOperations()
{
    // TODO: not really correct for triple buffering
    auto end_pending = m_PendingTextures.size();
    do
    {
        m_Backend->submitCommandBuffer(m_IOCommandBuffer);

        auto cur_buf_idx = m_BufferIndex;
        m_UploadHeapBoundary[cur_buf_idx] += m_UploadHeapSize;
        m_Backend->pushFence(m_Fence[cur_buf_idx]);

        auto next_buf_idx = m_BufferIndex = (cur_buf_idx + 1) % TEMPTEST_TEXTURE_TABLE_BUFFER_COUNT;
        auto fence = m_Fence[next_buf_idx];
        m_Backend->waitFence(fence);

        for(; m_ProcessedTextures < end_pending; ++m_ProcessedTextures)
        {
            auto& tex = m_PendingTextures[m_ProcessedTextures];
            auto& hdr = tex.Texture->getHeader();
            uint32 tex_size = hdr.Width*hdr.Height*DataFormatElementSize(hdr.Format);
            if(!InsertIntoStorage(tex.Texture, tex_size, m_UploadHeapSize, m_BufferIndex, m_UploadHeapBoundary, m_UploadHeap, m_IOCommandBuffer,
                                  tex.Slice, m_Textures[tex.Slot].Texture))
                break;
        }
    } while(m_ProcessedTextures < end_pending);
   
    clearPendingTextures();

    m_ProcessedTextures = 0;
}

template<class TBackend>
void TextureTable<TBackend>::clearPendingTextures()
{
    for(auto& tex : m_PendingTextures)
    {
        delete tex.Texture;
    }
    m_PendingTextures.clear();
}

template<class TBackend>
void TextureTable<TBackend>::setTextures(TBackend* backend)
{
    backend->setTextures(m_BakedTable.get());
}

template class TextureTable<GLRenderingBackend>;
}
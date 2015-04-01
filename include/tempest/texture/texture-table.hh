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

#ifndef _TEMPEST_TEXTURE_TABLE_HH_
#define _TEMPEST_TEXTURE_TABLE_HH_

#include "tempest/graphics/opengl-backend/gl-library.hh"

#include "tempest/image/image.hh"
#include "tempest/utils/types.hh"
#include "tempest/math/vector4.hh"
#include "tempest/graphics/texture.hh"

#include <memory>
#include <limits>

namespace Tempest
{
class Texture;
class BakedResourceTable;

#define TEMPEST_TEXTURE_TABLE_SLOTS \
    TEMPEST_TEXTURE_TABLE_SLOT(TEMPEST_RGBA_128x128, DataFormat::RGBA8UNorm, TextureTiling::Array, 128) \
    TEMPEST_TEXTURE_TABLE_SLOT(TEMPEST_RGBA_256x256, DataFormat::RGBA8UNorm, TextureTiling::Array, 256) \
    TEMPEST_TEXTURE_TABLE_SLOT(TEMPEST_RGBA_512x512, DataFormat::RGBA8UNorm, TextureTiling::Array, 512) \
    TEMPEST_TEXTURE_TABLE_SLOT(TEMPEST_RGBA_1024x1024, DataFormat::RGBA8UNorm, TextureTiling::Array, 1024) \
    TEMPEST_TEXTURE_TABLE_SLOT(TEMPEST_RGBA_CUBE, DataFormat::RGBA8UNorm, TextureTiling::Cube, 256)
    
#define TEMPEST_TEXTURE_TABLE_SLOT(name, format, tiling, edge) name,

enum
{
    TEMPEST_TEXTURE_TABLE_SLOTS
    TEMPEST_TEXTURE_SLOTS
};

#undef TEMPEST_TEXTURE_TABLE_SLOT

struct TextureTableDescription
{
    uint32          Slots[TEMPEST_TEXTURE_SLOTS];
    uint32          UploadHeapSize = 32 * 1024 * 1024;
    uint32          UploadQueueSize = 32;

    TextureTableDescription()
    {
        std::fill(std::begin(Slots), std::end(Slots), 16);
    }
};

#define TEMPTEST_TEXTURE_TABLE_BUFFER_COUNT 2
const Vector4 InvalidSlot = Vector4(-1.0f, -1.0f, -1.0f, -1.0f);

template<class TBackend>
class TextureTable
{
    typedef typename TBackend::TextureType TextureType;
    typedef typename TBackend::StorageType StorageType;
    typedef typename TBackend::IOCommandBufferType IOCommandBufferType;

    TBackend*               m_Backend;
    struct
    {
        uint32          SlotCount;
        uint32          LastSlot;
        TextureType*    Texture;
    } m_Textures[TEMPEST_TEXTURE_SLOTS];
    StorageType*            m_UploadHeap;
    int32                   m_UploadHeapBoundary[TEMPTEST_TEXTURE_TABLE_BUFFER_COUNT];
    uint32                  m_UploadHeapSize = 0;
    uint32                  m_BufferIndex = 0;

    IOCommandBufferType*    m_IOCommandBuffer;

    struct PendingTexture
    {
        uint32   Slot;
        uint32   Slice;
        Texture* Texture;
    };

    typedef typename TBackend::FenceType FenceType;
    typedef std::vector<PendingTexture> TextureQueue;
    TextureQueue            m_PendingTextures;
    uint32                  m_ProcessedTextures = 0;
    FenceType*              m_Fence[TEMPTEST_TEXTURE_TABLE_BUFFER_COUNT];

    std::unique_ptr<BakedResourceTable> m_BakedTable;
public:
    TextureTable(TBackend* backend, const TextureTableDescription& desc = TextureTableDescription());
     ~TextureTable();
    
    Vector4 loadTexture(const Path& filename);
    Vector4 loadCube(const Path& posx_filename,
                     const Path& negx_filename,
                     const Path& posy_filename,
                     const Path& negy_filename,
                     const Path& posz_filename,
                     const Path& negz_filename);

    void setTextures(TBackend* backend);

    void executeIOOperations();
private:
    void clearPendingTextures();
};
}

#endif // _TEMPEST_TEXTURE_TABLE_HH_
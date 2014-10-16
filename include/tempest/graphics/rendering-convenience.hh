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

#ifndef TEMPEST_RENDERING_CONVENIENCE_HH
#define TEMPEST_RENDERING_CONVENIENCE_HH

#include "tempest/graphics/rendering-definitions.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/vector4.hh"
#include "tempest/image/image.hh"

#include <array>
#include <memory>
#include <bits/stl_vector.h>

namespace Tempest
{
template<class T>
struct ConvertTextureFormat;

#define CONVERT_TEXTURE_FORMAT(T, fmt) template<> struct ConvertTextureFormat<T> { static const DataFormat format = fmt; }

CONVERT_TEXTURE_FORMAT(float, DataFormat::R32F);
CONVERT_TEXTURE_FORMAT(Vector2, DataFormat::RG32F);
CONVERT_TEXTURE_FORMAT(Vector3, DataFormat::RGB32F);
CONVERT_TEXTURE_FORMAT(Vector4, DataFormat::RGBA32F);
CONVERT_TEXTURE_FORMAT(int8, DataFormat::R8);
CONVERT_TEXTURE_FORMAT(uint8, DataFormat::uR8);
    
template<class TBackend, class T>
class ResourceDestructor
{
    TBackend* m_Backend;
public:
    explicit ResourceDestructor(TBackend* backend)
        :   m_Backend(backend) {}
    
    void operator()(T* ptr)
    {
        m_Backend->destroyRenderResource(ptr);
    }
};

template<class TBackend, class T>
using UniqueResource = std::unique_ptr<T, ResourceDestructor<TBackend, T>>;

template<class TBackend, class T>
UniqueResource<TBackend, T> CreateUniqueResource(TBackend* backend, T* ptr)
{
    return UniqueResource<TBackend, T>(ptr, ResourceDestructor<TBackend, T>(backend));
}

template<class TBackend, class TResource, class T>
class SubresourceDestructor
{
    TBackend*  m_Backend;
    TResource* m_Resource;
public:
    explicit SubresourceDestructor(TBackend* backend, TResource* res)
        :   m_Backend(backend),
            m_Resource(res) {}
    
    void operator()(T* ptr)
    {
        if(ptr)
        {
            m_Resource->destroyRenderResource(m_Backend, ptr);
        }
    }
};

template<class TBackend, class TResource, class T>
using UniqueSubresource = std::unique_ptr<T, SubresourceDestructor<TBackend, TResource, T>>;

template<class TBackend, class TResource, class T>
UniqueSubresource<TBackend, TResource, T> CreateUniqueSubresource(TBackend* backend, TResource* res, T* ptr)
{
    return UniqueSubresource<TBackend, TResource, T>(ptr, SubresourceDestructor<TBackend, TResource, T>(backend, res));
}

template<class TBackend, class T>
UniqueResource<TBackend, typename TBackend::BufferType> CreateBuffer(TBackend* backend, const T& arr, VBType vb_type, uint32 flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createBuffer(arr.size()*sizeof(typename T::value_type), vb_type, flags, &arr.front()));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::BufferType> CreateBuffer(TBackend* backend, const void* arr, size_t size, VBType vb_type, uint32 flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createBuffer(size, vb_type, flags, arr));
}

template<class TBackend, class T, size_t TWidth, size_t THeight>
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const T arr[TWidth][THeight], uint32 flags = RESOURCE_STATIC_DRAW)
{
    TextureDescription desc;
    desc.Width = TWidth;
    desc.Height = THeight;
    desc.Depth = 1;
    desc.Format = ConvertTextureFormat<T>::format;
    desc.Tiling = TextureTiling::Flat;

    return CreateUniqueResource(backend, backend->createTexture(desc, flags, &arr[0][0]));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const string& filename, uint32 flags = RESOURCE_STATIC_DRAW)
{
    std::unique_ptr<Texture> tex(LoadImage(Path(filename)));
    return CreateUniqueResource(backend, tex ? backend->createTexture(tex->getHeader(), flags, tex->getData()) : nullptr);
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::CommandBufferType> CreateCommandBuffer(TBackend* backend)
{
    return CreateUniqueResource(backend, backend->createCommandBuffer());
}

template<class TBackend, class TShader, class T>
UniqueSubresource<TBackend, TShader, typename TShader::InputLayoutType> CreateInputLayout(TBackend* backend, TShader* shader, const T& arr)
{
    return CreateUniqueSubresource(backend, shader, shader->createInputLayout(backend, &arr.front(), arr.size()));
}

class BasicFileLoader: public FileLoader
{
public:
    virtual FileDescription* loadFileContent(const string& name) final;
    virtual void freeFileContent(FileDescription* ptr) final;
};

template<class TCompiler>
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const string& filename,
                                                                              const string& technique_name = string(), const string& pass_name = string())
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader, technique_name, pass_name));
}

template<class TShader>
UniqueResource<TShader, typename TShader::ResourceTableType> CreateResourceTable(TShader* shader, const string& table, size_t extended = 0)
{
    return CreateUniqueResource(shader, shader->createResourceTable(table, extended));
}

template<class TTable>
std::unique_ptr<typename TTable::BakedResourceTableType> ExtractBakedResourceTable(TTable* table)
{
    return std::unique_ptr<typename TTable::BakedResourceTableType>(table->extractBakedTable());
}

template<class TShader>
std::unique_ptr<typename TShader::LinkedShaderProgram> LinkShaderProgram(TShader* shader, typename TShader::ResourceTableType::BakedResourceTableType* baked_table)
{
    return std::unique_ptr<typename TShader::LinkedShaderProgram>(shader->link(baked_table));
}
}

#endif // TEMPEST_RENDERING_CONVENIENCE_HH
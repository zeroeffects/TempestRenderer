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
#include "tempest/shader/file-loader.hh"
#include "tempest/graphics/texture.hh"

#include <array>
#include <memory>
#include <bits/stl_vector.h>

namespace Tempest
{
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
        m_Resource->destroyRenderResource(m_Backend, ptr);
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
UniqueResource<TBackend, typename TBackend::BufferType> CreateBuffer(TBackend* backend, const T& arr, VBType vb_type, VBUsage usage = VBUsage::StaticDraw)
{
    return CreateUniqueResource(backend, backend->createBuffer(arr.size()*sizeof(typename T::value_type), vb_type, usage, &arr.front()));
}

template<class TBackend, class T, size_t TWidth, size_t THeight>
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const T arr[TWidth][THeight], VBUsage usage = VBUsage::StaticDraw)
{
    TextureDescription desc;
    desc.Width = TWidth;
    desc.Height = THeight;
    desc.Depth = 1;
    desc.Tiling = TextureTiling::Flat;

    return CreateUniqueResource(backend, backend->createTexture(desc, usage, backend, &arr.front()));
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
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const string& filename)
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader));
}

template<class TShader>
UniqueResource<TShader, typename TShader::ResourceTableType> CreateResourceTable(TShader* shader, const string& table)
{
    return CreateUniqueResource(shader, shader->createResourceTable(table));
}

template<class TTable>
std::unique_ptr<typename TTable::BakedResourceTableType> ExtractBakedResourceTable(TTable* shader)
{
    return std::unique_ptr<typename TTable::BakedResourceTableType>(shader->extractBakedTable());
}
}

#endif // TEMPEST_RENDERING_CONVENIENCE_HH
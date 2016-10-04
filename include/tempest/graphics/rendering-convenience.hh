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
#include "tempest/graphics/os-window.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/memory.hh"
#include "tempest/parser/file-loader.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/vector4.hh"
#include "tempest/image/image.hh"

#include <string>
#include <array>
#include <memory>

namespace Tempest
{
template<class T>
struct ConvertTextureFormat;
struct RasterizerStates;
struct BlendStates;
struct DepthStencilStates;
struct CommandBufferDescription;
struct WindowDescription;
class BakedResourceTable;

#define CONVERT_TEXTURE_FORMAT(T, fmt) template<> struct ConvertTextureFormat<T> { static const DataFormat format = fmt; }

CONVERT_TEXTURE_FORMAT(float, DataFormat::R32F);
CONVERT_TEXTURE_FORMAT(Vector2, DataFormat::RG32F);
CONVERT_TEXTURE_FORMAT(Vector3, DataFormat::RGB32F);
CONVERT_TEXTURE_FORMAT(Vector4, DataFormat::RGBA32F);
CONVERT_TEXTURE_FORMAT(int8_t, DataFormat::R8);
CONVERT_TEXTURE_FORMAT(uint8_t, DataFormat::uR8);
    
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

template<class TSystem>
std::unique_ptr<TSystem> CreateSystemAndWindowSimple(const WindowDescription& wdesc)
{
    typedef std::unique_ptr<TSystem> TSystemPtr;
    auto sys = Tempest::make_unique<TSystem>();

    auto status = sys->Library.initDeviceContextLibrary();
    if(!status)
        return TSystemPtr();

    status = sys->Window.init(sys->Display, 0, wdesc);
    if(!status)
        return TSystemPtr();

    status = sys->Backend.attach(sys->Display, sys->Window);
    if(!status)
        return TSystemPtr();

    status = sys->Library.initGraphicsLibrary();
    if(!status)
        return TSystemPtr();

    sys->Backend.init();

    return sys;
}

template<class TBackend, class TResource, class T>
using UniqueSubresource = std::unique_ptr<T, SubresourceDestructor<TBackend, TResource, T>>;

template<class TBackend, class TResource, class T>
UniqueSubresource<TBackend, TResource, T> CreateUniqueSubresource(TBackend* backend, TResource* res, T* ptr)
{
    return UniqueSubresource<TBackend, TResource, T>(ptr, SubresourceDestructor<TBackend, TResource, T>(backend, res));
}

template<class TBackend, class T>
UniqueResource<TBackend, typename TBackend::BufferType> CreateBuffer(TBackend* backend, const T& arr, ResourceBufferType buffer_type, uint32_t flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createBuffer(arr.size()*sizeof(typename T::value_type), buffer_type, flags, &arr.front()));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::BufferType> CreateBuffer(TBackend* backend, const void* arr, size_t size, ResourceBufferType buffer_type, uint32_t flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createBuffer(size, buffer_type, flags, arr));
}

template<class TBufferType, class T>
void UploadConstantBuffer(TBufferType* buffer, const T& data)
{
    buffer->uploadConstantBuffer(&data, sizeof(T));
}

template<class TBackend, class T, size_t TWidth, size_t THeight>
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const T arr[TWidth][THeight], uint32_t flags = RESOURCE_STATIC_DRAW)
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
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const std::string& filename, uint32_t flags = RESOURCE_STATIC_DRAW)
{
    std::unique_ptr<Texture> tex(LoadImage(Path(filename)));
    return CreateUniqueResource(backend, tex ? backend->createTexture(tex->getHeader(), flags, tex->getData()) : nullptr);
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::SamplerType> CreateSampler(TBackend* backend, const SamplerDescription& desc)
{
    return CreateUniqueResource(backend, backend->createSampler(desc));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::FramebufferType> CreateFramebuffer(TBackend* backend, typename TBackend::RenderTargetType** color, uint32_t color_rt_count, typename TBackend::RenderTargetType* depth = nullptr)
{
    return CreateUniqueResource(backend, backend->createFramebuffer(color, color_rt_count, depth));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::TextureType> CreateTexture(TBackend* backend, const TextureDescription& tex_desc, uint32_t flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createTexture(tex_desc, flags, nullptr));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::RenderTargetType> CreateRenderTarget(TBackend* backend, const TextureDescription& tex_desc, uint32_t flags = RESOURCE_STATIC_DRAW)
{
    return CreateUniqueResource(backend, backend->createRenderTarget(tex_desc, flags));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::CommandBufferType> CreateCommandBuffer(TBackend* backend, const CommandBufferDescription& desc)
{
    return CreateUniqueResource(backend, backend->createCommandBuffer(desc));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::CommandBufferType> CreateCommandBuffer(TBackend* backend, const CommandBufferDescription& desc, typename TBackend::FramebufferType* fb,
                                                                                   typename TBackend::RenderPassType* pass, typename TBackend::BindPointLayoutType* layout)
{
    return CreateUniqueResource(backend, backend->createCommandBuffer(desc, fb, pass, layout));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::IOCommandBufferType> CreateIOCommandBuffer(TBackend* backend, const IOCommandBufferDescription& desc)
{
    return CreateUniqueResource(backend, backend->createIOCommandBuffer(desc));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::StorageType> CreateStorageBuffer(TBackend* backend, StorageMode storage_type, uint32_t size)
{
    return CreateUniqueResource(backend, backend->createStorageBuffer(storage_type, size));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::RenderPassType> CreateRenderPass(TBackend* backend, DataFormat* fmts, uint32_t attachment_count,
                                                                             const ClearValue* clear_value, uint32_t flags)
{
    return CreateUniqueResource(backend, backend->createRenderPass(fmts, attachment_count, clear_value, flags));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::BindPointLayoutType> CreateBindPointLayout(TBackend* backend, const BindPointLayout& layout)
{
    return CreateUniqueResource(backend, backend->createBindPointLayout(layout));
}

class BasicFileLoader: public FileLoader
{
public:
    virtual FileDescription* loadFileContent(const std::string& name) final;
    virtual void freeFileContent(FileDescription* ptr) final;
};

class SubdirectoryFileLoader: public FileLoader
{
    std::string m_Subdirectory;
public:
    SubdirectoryFileLoader(const std::string& dir);

    virtual FileDescription* loadFileContent(const std::string& name) final;
    virtual void freeFileContent(FileDescription* ptr) final;
};

template<class TBackend>
class CUDASurfaceDeleter
{
    TBackend* m_Backend;
public:
    CUDASurfaceDeleter(TBackend* backend)
        :   m_Backend(backend) {}

    void operator()(Tempest::CUDASurfaceResource& surf) { m_Backend->unmapCudaSurface(&surf); }
};

template<class TBackend, class TTexture>
ScopedObject<CUDASurfaceResource, CUDASurfaceDeleter<TBackend>> MapCudaSurface(TBackend* backend, TTexture* tex, uint32_t flags)
{
    auto cuda_tex = Tempest::CreateScoped<Tempest::CUDASurfaceResource, CUDASurfaceDeleter<TBackend>>(CUDASurfaceDeleter<TBackend>(backend));
    backend->mapToCudaSurface(tex, flags, &cuda_tex);
    return cuda_tex;
}

template<class TBackend, class TPtr, class TDeleter, template<class T, class U> class TTexture>
ScopedObject<CUDASurfaceResource, CUDASurfaceDeleter<TBackend>> MapCudaSurface(TBackend* backend, TTexture<TPtr, TDeleter>& tex, uint32_t flags)
{
    auto cuda_tex = Tempest::CreateScoped<Tempest::CUDASurfaceResource, CUDASurfaceDeleter<TBackend>>(CUDASurfaceDeleter<TBackend>(backend));
    backend->mapToCudaSurface(tex.get(), flags, &cuda_tex);
    return cuda_tex;
}

template<class TCompiler>
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const std::string& filename)
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader, nullptr, 0));
}

template<class TCompiler, size_t TArraySize>
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const std::string& filename,
                                                                              const std::string _array[TArraySize])
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader, _array, TArraySize));
}

template<class TCompiler>
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const std::string& filename,
                                                                              const std::string* _array, uint32_t size)
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader, _array, size));
}

template<class TCompiler, template<class T, class TAlloc> class TContainer, class TOptAlloc>
UniqueResource<TCompiler, typename TCompiler::ShaderProgramType> CreateShader(TCompiler* compiler, const std::string& filename,
                                                                              const TContainer<std::string, TOptAlloc>& opts)
{
    BasicFileLoader loader;
    return CreateUniqueResource(compiler, compiler->compileShaderProgram(filename, &loader, &opts.front(), opts.size()));
}

template<class TShader>
std::unique_ptr<typename TShader::ResourceTableType> CreateResourceTable(TShader* shader, const std::string& table, size_t extended = 0)
{
    return std::unique_ptr<typename TShader::ResourceTableType>(shader->createResourceTable(table, extended));
}

template<class TTable>
std::unique_ptr<BakedResourceTable> ExtractBakedResourceTable(TTable* table)
{
    return std::unique_ptr<BakedResourceTable>(table->extractBakedTable());
}

template<class TShader>
std::unique_ptr<typename TShader::LinkedShaderProgram> LinkShaderProgram(TShader* shader, BakedResourceTable* baked_table)
{
    return std::unique_ptr<typename TShader::LinkedShaderProgram>(shader->link(baked_table));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::StateObjectType> CreateStateObject(TBackend* backend,
                                                                               DataFormat* rt_fmt,
                                                                               size_t rt_count,
                                                                               DataFormat ds_fmt,
                                                                               typename TBackend::ShaderProgramType* shader_program,
                                                                               DrawModes primitive_type = DrawModes::TriangleList,
                                                                               const RasterizerStates* rasterizer_states = nullptr,
                                                                               const BlendStates* blend_states = nullptr,
                                                                               const DepthStencilStates* depth_stencil_state = nullptr)
{
    return CreateUniqueResource(backend, backend->createStateObject(rt_fmt, rt_count, ds_fmt, shader_program, primitive_type, rasterizer_states, blend_states, depth_stencil_state));
}

template<class TBackend>
UniqueResource<TBackend, typename TBackend::StateObjectType> CreateStateObject(TBackend* backend,
                                                                               BufferBinding* buffer,
                                                                               size_t buffer_count,
                                                                               DataFormat* rt_fmt,
                                                                               size_t rt_count,
                                                                               DataFormat ds_fmt,
                                                                               typename TBackend::RenderPassType* render_pass,
                                                                               typename TBackend::ShaderProgramType* shader_program,
                                                                               typename TBackend::BindPointLayoutType* layout,
                                                                               DrawModes primitive_type = DrawModes::TriangleList,
                                                                               const RasterizerStates* rasterizer_states = nullptr,
                                                                               const BlendStates* blend_states = nullptr,
                                                                               const DepthStencilStates* depth_stencil_state = nullptr)
{
    return CreateUniqueResource(backend, backend->createStateObject(buffer, buffer_count, rt_fmt, rt_count, ds_fmt, render_pass, shader_program, layout, primitive_type, rasterizer_states, blend_states, depth_stencil_state));
}
}

#endif // TEMPEST_RENDERING_CONVENIENCE_HH
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

#include "tempest/graphics/scene-rasterizer.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/math/functions.hh"

// DEBUG
#include "tempest/utils/file-system.hh"
#include "tempest/image/image.hh"
// /DEBUG

namespace Tempest
{
static const std::string MeshFormatNames[] =
{
#define MESH_FORMAT(name) #name,
    MESH_FORMAT_LIST
#undef MESH_FORMAT
};

static const uint32_t UploadHeapSize = 16*1024*1024;
static const uint32_t SphereTriangleCount = 128;

static const char* IlluminationModelsShaderNames[] =
{
    nullptr, //Emissive,
    SOURCE_SHADING_DIR "/deferred-blinn-phong-pass.tfx", //BlinnPhong,
    nullptr, //KajiyaKay,
    nullptr, //AshikhminShirley,
    nullptr, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
    nullptr, //Patchwork,
    nullptr, //Mirror,
    nullptr, //_Free4,
    nullptr, //GGXMicrofacet,
    nullptr, //GGXMicrofacetConductor,
    nullptr, //GGXMicrofacetDielectric,
    nullptr, //Mix,
    nullptr, //GGXMicrofacetAnisotropic,
    nullptr, //GGXMicrofacetConductorAnisotropic,
    nullptr, //GGXMicrofacetDielectricAnisotropic,
	nullptr, //_Free5,
    nullptr, //_Free6,
    nullptr, //StochasticRotator,
    nullptr, //_Free7,
    nullptr, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    SOURCE_SHADING_DIR "/deferred-sggx-surface-pass.tfx", //SGGXSurface,
    nullptr, //SGGXPseudoVolume,
    nullptr, //_Free8,
    nullptr, //BTF,
    nullptr, //BeckmannMicrofacet,
    nullptr, //BeckmannMicrofacetConductor,
    nullptr, //BeckmannMicrofacetDielectric,
    nullptr, //BeckmannMicrofacetAnisotropic,
    nullptr, //BeckmannMicrofacetConductorAnisotropic,
    nullptr, //BeckmannMicrofacetDielectricAnisotropic,
};

SceneRasterizer::SceneRasterizer(PreferredBackend* backend, PreferredShaderCompiler* shader_compiler, uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
    :   m_Backend(backend),
        m_ShaderCompiler(shader_compiler),
        m_TextureTable(backend, {}),
        m_UploadHeap(backend, UploadHeapSize),
        m_ViewProjectionMatrix(view_proj_inv.inverse()),
        m_ViewProjectionInverseMatrix(view_proj_inv),
        m_DeferredShadingTextures(m_TextureTable.getBakedTable()->getSize() + (TGE_FIXED_ARRAY_SIZE(m_GBuffer) + 1)*4*sizeof(float))
{
    std::fill(std::begin(m_ShaderTable), std::end(m_ShaderTable), nullptr);
    std::fill(std::begin(m_GBuffer), std::end(m_GBuffer), nullptr);
    std::fill(std::begin(m_DeferredShading), std::end(m_DeferredShading), nullptr);
    std::fill(std::begin(m_DeferredShadingStates), std::end(m_DeferredShadingStates), nullptr);

    memset(m_DeferredShadingTextures.get(), 0, m_DeferredShadingTextures.getSize());

    DataFormat fmts[TGE_FIXED_ARRAY_SIZE(m_GBuffer)];
    std::fill(fmts, fmts + TGE_FIXED_ARRAY_SIZE(m_GBuffer), DataFormat::RGBA8UNorm);

    DepthStencilStates ds_state;
    ds_state.DepthTestEnable = true;
    ds_state.DepthWriteEnable = true;

    for(size_t shader_idx = 0, shader_idx_end = (size_t)MeshFormat::Count; shader_idx < shader_idx_end; ++shader_idx)
    {
        auto shader = m_MeshGbuffer[shader_idx] = shader_compiler->compileShaderProgram(SOURCE_SHADING_DIR "/mesh-gbuffer.tfx", nullptr, &MeshFormatNames[shader_idx], 1);
        m_MeshGBufferStates[shader_idx] = backend->createStateObject(fmts, TGE_FIXED_ARRAY_SIZE(fmts), DataFormat::D24S8, shader, DrawModes::TriangleList, nullptr, nullptr, &ds_state);
    }

    Tempest::IOCommandBufferDescription io_cmd_buf_desc;
    io_cmd_buf_desc.CommandCount = 1024;
    m_IOCommandBuffer = m_Backend->createIOCommandBuffer(io_cmd_buf_desc);

    updateGBuffer(width, height);

    Sphere sphere;
    sphere.Center = {};
    sphere.Radius = 1.0f;

    ScopedArray<Vector3> vertices;
    ScopedArray<int32_t> indices;
    uint32_t vert_count, index_count;
    TriangleTessellationNoNormals(sphere, 2*SphereTriangleCount, SphereTriangleCount, &vertices, &vert_count, &indices, &index_count);

    std::unique_ptr<uint16_t[]> indices16(new uint16_t[index_count]);
    for(uint32_t idx = 0; idx < index_count; ++idx)
    {
        indices16[idx] = static_cast<uint16_t>(indices[idx]);
    }

    m_SphereDraw.BaseIndex = 0;
    m_SphereDraw.BaseVertex = 0;
    m_SphereDraw.IndexBuffer = m_Backend->createBuffer(index_count*sizeof(uint16_t), ResourceBufferType::IndexBuffer, 0, indices16.get());;
    m_SphereDraw.VertexBuffers[0].VertexBuffer = m_Backend->createBuffer(vert_count*sizeof(Vector3), ResourceBufferType::VertexBuffer, 0, vertices.get());;
    m_SphereDraw.VertexCount = index_count;
}

SceneRasterizer::~SceneRasterizer()
{
    m_Backend->destroyRenderResource(m_SphereDraw.VertexBuffers[0].VertexBuffer);
    m_Backend->destroyRenderResource(m_SphereDraw.IndexBuffer);
    m_Backend->destroyRenderResource(m_IOCommandBuffer);
    m_Backend->destroyRenderResource(m_DeferredGeometryCommandBuffer);
    m_Backend->destroyRenderResource(m_DeferredShadingCommandBuffer);
    //m_Backend->destroyRenderResource(m_Fence);

    m_Backend->destroyRenderResource(m_DeferredShadingFramebuffer);

    for(auto& gbuffer : m_GBuffer)
    {
        m_Backend->destroyRenderResource(gbuffer);
    }

    m_Backend->destroyRenderResource(m_GBufferDepth);
    m_Backend->destroyRenderResource(m_BackbufferFramebuffer);
    m_Backend->destroyRenderResource(m_Backbuffer);

    for(auto& baked_table : m_LightSourceBakedTables)
    {
        delete baked_table;
    }

    for(auto& submesh : m_Submeshes)
    {
        delete submesh.Batch.ResourceTable;
        delete[] submesh.BVH;
    }

    for(auto* buf : m_Buffers)
    {
        m_Backend->destroyRenderResource(buf);
    }

    for(auto* state : m_MeshGBufferStates)
    {
        m_Backend->destroyRenderResource(state);
    }

    for(auto* shader : m_MeshGbuffer)
    {
        m_ShaderCompiler->destroyRenderResource(shader);
    }

    for(auto* state : m_DeferredShadingStates)
    {
        m_Backend->destroyRenderResource(state);
    }

    for(auto* shader : m_DeferredShading)
    {
        m_ShaderCompiler->destroyRenderResource(shader);
    }

    for(auto* shader : m_ShaderTable)
    {
        m_ShaderCompiler->destroyRenderResource(shader);
    }
}

void SceneRasterizer::setBackgroundSpectrum(Spectrum spectrum)
{
    TGE_ASSERT(false, "Stub");
}

void SceneRasterizer::setPicturePostProcess(PicturePostProcess mode)
{
    TGE_ASSERT(false, "Stub");
}

void SceneRasterizer::setGlobalCubeMap(CubeMap* cube_map)
{
    TGE_ASSERT(false, "Stub");
}

void SceneRasterizer::setRenderMode(RenderMode render_mode)
{
    TGE_ASSERT(false, "Stub");
}

RenderMode SceneRasterizer::getRenderMode() const
{
    TGE_ASSERT(false, "Stub"); return {};
}

void addLightSource(LightSource* light)
{
    TGE_ASSERT(false, "Stub");
}

unsigned SceneRasterizer::addSphereLightSource(SphereAreaLight* light)
{
    TGE_ASSERT(false, "Stub"); return {};
}

void SceneRasterizer::clearLightSources()
{
    TGE_ASSERT(false, "Stub");
}

void SceneRasterizer::updateGBuffer(uint32_t width, uint32_t height)
{
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = width;
    tex_desc.Height = height;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    m_Backbuffer = m_Backend->createRenderTarget(tex_desc);

    m_BackbufferFramebuffer = m_Backend->createFramebuffer(&m_Backbuffer, 1);

    for(auto& gbuffer : m_GBuffer)
    {
        gbuffer = m_Backend->createRenderTarget(tex_desc);
    }

    tex_desc.Format = Tempest::DataFormat::D24S8;
    m_GBufferDepth = m_Backend->createRenderTarget(tex_desc);

    auto tex_slots_start = m_DeferredShadingTextures.get();
    auto tex_slots_size = m_DeferredShadingTextures.getSize();
    auto tex_slots_data = tex_slots_start + (tex_slots_size - (TGE_FIXED_ARRAY_SIZE(m_GBuffer) + 1)*4*sizeof(float));
    *reinterpret_cast<uint64_t*>(tex_slots_data) = m_GBufferDepth->getHandle();
    tex_slots_data += 4*sizeof(float);
    size_t idx = 0;
    for(auto tex_slots_data_end = tex_slots_start + tex_slots_size; tex_slots_data < tex_slots_data_end; tex_slots_data += 4*sizeof(float))
    {
        *reinterpret_cast<uint64_t*>(tex_slots_data) = m_GBuffer[idx++]->getHandle();
    }

    m_DeferredShadingFramebuffer = m_Backend->createFramebuffer(m_GBuffer, TGE_FIXED_ARRAY_SIZE(m_GBuffer), m_GBufferDepth);
}

void SceneRasterizer::draw(uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
{
    m_ViewProjectionMatrix = view_proj_inv.inverse();
    m_ViewProjectionInverseMatrix = view_proj_inv;

    auto& hdr = m_GBufferDepth->getDescription();
    if(hdr.Width != width || hdr.Height != height)
    {
        m_Backend->destroyRenderResource(m_DeferredShadingFramebuffer);

        for(auto& gbuffer : m_GBuffer)
        {
            m_Backend->destroyRenderResource(gbuffer);
        }

        m_Backend->destroyRenderResource(m_GBufferDepth);
        m_Backend->destroyRenderResource(m_BackbufferFramebuffer);
        m_Backend->destroyRenderResource(m_Backbuffer);
        
        updateGBuffer(width, height);
    }

    for(auto& submesh : m_Submeshes)
    {
        auto view = m_ViewProjectionInverseMatrix.rotationFromPerspectiveInverseMatrix();
        auto rot_transform = view*submesh.TransformMatrix.normalTransform();
        auto view_proj_world = m_ViewProjectionMatrix*submesh.TransformMatrix;
        auto res_table = submesh.Batch.ResourceTable;
        res_table->setValue(submesh.WorldViewProjectionMatrixOffset, view_proj_world);
        res_table->setValue(submesh.RotateViewMatrixOffset, rot_transform);
    }
    drawOnce();
}

void SceneRasterizer::drawOnce()
{
    auto& hdr = m_Backbuffer->getDescription();

    m_Backend->setViewportRect(0, 0, hdr.Width, hdr.Height);

    m_Backend->setFramebuffer(m_DeferredShadingFramebuffer);
    
    for(uint32_t idx = 0; idx < TGE_FIXED_ARRAY_SIZE(m_GBuffer); ++idx)
    {
        m_Backend->clearColorBuffer(idx, Tempest::Vector4{0.0f, 0.0f, 0.0f, 0.0f});
    }
    m_Backend->clearDepthStencilBuffer();

    m_Backend->submitCommandBuffer(m_DeferredGeometryCommandBuffer);

    m_Backend->setFramebuffer(m_BackbufferFramebuffer);

    m_Backend->clearColorBuffer(0, Tempest::Vector4{0.0f, 1.0f, 0.0f, 0.0f});
    m_Backend->clearDepthStencilBuffer();

    m_Backend->setTextures(&m_DeferredShadingTextures);

    m_Backend->submitCommandBuffer(m_DeferredShadingCommandBuffer);

    //m_Backend->blitAttachmentToScreen(Tempest::AttachmentType::Color, 0, 0, 0, 0, 0, hdr.Width, hdr.Height);

    /*
    // DEBUG
    static size_t counter = 0;

    if(counter++ == 0)
    {
        auto data_size = hdr.Width*hdr.Height*sizeof(uint32_t);
        auto storage = m_Backend->createStorageBuffer(StorageMode::PixelPack, data_size);

        IOCommandBufferType::IOCommandType io_cmd;
        io_cmd.CommandType = IOCommandMode::CopyTextureToStorage;
        io_cmd.SourceOffset = 0;
        io_cmd.DestinationOffset = 0;
        io_cmd.Source.Texture = m_Backbuffer;
        io_cmd.Destination.Storage = storage;
        io_cmd.Width = hdr.Width;
        io_cmd.Height = hdr.Height;

        auto iocmdbuf = m_Backend->createIOCommandBuffer({ 1 });
        iocmdbuf->enqueueCommand(io_cmd);
        m_Backend->submitCommandBuffer(iocmdbuf);

        std::unique_ptr<uint32_t[]> data(new uint32_t[hdr.Width*hdr.Height]);
        storage->extractTexture(0, hdr, data.get());

        m_Backend->destroyRenderResource(iocmdbuf);
        m_Backend->destroyRenderResource(storage);

        Tempest::SaveImage(hdr, data.get(), Path("dbg-defer.png"));

        TGE_TRAP();
    }
    // /DEBUG
    //*/

    m_Backend->setFramebuffer(nullptr);
}

const void* SceneRasterizer::bindTexture(const Texture* tex)
{
    auto scale_tex = m_TextureTable.loadTexture(tex);
    m_TextureSlots.push_back(scale_tex);
    return &m_TextureSlots.back();
}

void SceneRasterizer::bindSurfaceAndTexture(Texture* tex, const void** tex_obj, void** surf)
{
    TGE_ASSERT(false, "Stub");
}

void* SceneRasterizer::bindBuffer(void* buf, size_t)
{
    TGE_ASSERT(false, "Stub"); return {};
}

void SceneRasterizer::addTriangleMesh(const Matrix4& world,
                                      size_t submesh_count, 
                                      RTSubmesh* submeshes,
                                      size_t index_count, int32_t* tris,
                                      size_t vert_size, void* verts,
                                      MeshOptions* mesh_opts,
						              uint64_t* geom_ids)
{
    index_count *= 3;
    std::unique_ptr<uint16_t[]> converted_indices(new uint16_t[index_count]);
    for(size_t idx = 0; idx < index_count; ++idx)
    {
        TGE_ASSERT(tris[idx] < ((1u << 16u) - 1u), "Unsupported index count");
        converted_indices[idx] = static_cast<uint16_t>(tris[idx]);
    }

    if(index_count*sizeof(uint16_t) > UploadHeapSize)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Index buffer is bigger than upload heap");
        return;
    }

    if(vert_size > UploadHeapSize)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Vertex buffer is bigger than upload heap");
    }

    auto submesh_offset = m_Submeshes.size();
    m_Submeshes.resize(submesh_offset + submesh_count);

    
    auto index_buffer = m_Backend->createBuffer(index_count*sizeof(uint16_t), ResourceBufferType::IndexBuffer);
    auto vertex_buffer = m_Backend->createBuffer(vert_size, ResourceBufferType::VertexBuffer);

    m_Buffers.push_back(index_buffer);
    m_Buffers.push_back(vertex_buffer);

    for(size_t submesh_idx = 0; submesh_idx < submesh_count; ++submesh_idx)
    {
        auto& submesh = submeshes[submesh_idx];
        auto& result_submesh = m_Submeshes[submesh_offset + submesh_idx];
     
		auto tri_count = submesh.VertexCount/3;

        void* submesh_vert_ptr = reinterpret_cast<uint8_t*>(verts) + submesh.VertexOffset;
		auto submesh_ind_ptr = reinterpret_cast<uint32_t*>(tris + submesh.BaseIndex);
		uint32_t submesh_vert_count = (static_cast<uint32_t>(vert_size) - submesh.VertexOffset)/submesh.Stride;
        std::unique_ptr<LBVH2Node<AABBUnaligned>> nodes(GenerateTriangleNodes<AABBUnaligned>(submesh_vert_ptr, submesh_vert_count, submesh_ind_ptr, tri_count, submesh.Stride));
	
        result_submesh.BVH = GenerateSSLBVH(nodes.get(), tri_count);

        std::unique_ptr<PreferredBackend::ShaderProgramType::ResourceTableType> res_table;
        StateObjectType* state_object;

        switch(submesh.Stride)
        {
        #define MESH_FORMAT(name) \
            case sizeof(name): { \
                res_table = decltype(res_table)(m_MeshGbuffer[(size_t)MeshFormat::name]->createResourceTable("Globals", 1)); \
                state_object = m_MeshGBufferStates[(size_t)MeshFormat::name]; \
            } break;
            MESH_FORMAT_LIST
        #undef MESH_FORMAT
        default: TGE_ASSERT(false, "Unsupported format"); return; 
        }

        auto rot_transform = world.normalTransform();

        auto view_proj_world = m_ViewProjectionMatrix*world;

        auto wvp_res_idx = res_table->getResourceIndex("Globals.WorldViewProjectionTransform");
        auto rv_res_idx = res_table->getResourceIndex("Globals.RotateViewTransform");
        res_table->setResource("Globals.MaterialID", (unsigned)submesh.Material->Model);
        res_table->setResource(wvp_res_idx, view_proj_world);
        auto view = m_ViewProjectionInverseMatrix.rotationFromPerspectiveInverseMatrix();
        res_table->setResource(rv_res_idx, view*rot_transform);

        result_submesh.Model = submesh.Material->Model;

        result_submesh.WorldViewProjectionMatrixOffset = wvp_res_idx.BaseOffset;
        result_submesh.RotateViewMatrixOffset = rv_res_idx.BaseOffset;
        result_submesh.TransformMatrix = world;

        result_submesh.Batch.VertexCount = submesh.VertexCount;
        result_submesh.Batch.BaseVertex = 0;
        result_submesh.Batch.BaseIndex = submesh.BaseIndex;
        result_submesh.Batch.ResourceTable = res_table->extractBakedTable();
        result_submesh.Batch.PipelineState = state_object;
        result_submesh.Batch.IndexBuffer = index_buffer;
        result_submesh.Batch.VertexBuffers[0].VertexBuffer = vertex_buffer;
        result_submesh.Batch.VertexBuffers[0].Stride = submesh.Stride;
        result_submesh.Batch.VertexBuffers[0].Offset = submesh.VertexOffset;
    }

    {
    uint32_t index_size = (uint32_t)index_count*sizeof(uint16_t);
    auto data_index = m_UploadHeap.pushData(converted_indices.get(), index_size);

    IOCommandBufferType::IOCommandType io_cmd;
    io_cmd.CommandType = IOCommandMode::CopyStorageToBuffer;
    io_cmd.Source.Storage = m_UploadHeap.getStorage();
    io_cmd.Destination.Buffer = index_buffer;
    io_cmd.SourceOffset = data_index;
    io_cmd.DestinationOffset = 0;
    io_cmd.Width = index_size;

    if(!m_IOCommandBuffer->enqueueCommand(io_cmd))
    {
        m_Backend->submitCommandBuffer(m_IOCommandBuffer);
        m_IOCommandBuffer->clear();
    }
    }

    {
    auto data_index = m_UploadHeap.pushData(verts, (uint32_t)vert_size);
    
    IOCommandBufferType::IOCommandType io_cmd;
    io_cmd.CommandType = IOCommandMode::CopyStorageToBuffer;
    io_cmd.Source.Storage = m_UploadHeap.getStorage();
    io_cmd.Destination.Buffer = vertex_buffer;
    io_cmd.SourceOffset = data_index;
    io_cmd.DestinationOffset = 0;
    io_cmd.Width = (uint32_t)vert_size;

    if(!m_IOCommandBuffer->enqueueCommand(io_cmd))
    {
        m_Backend->submitCommandBuffer(m_IOCommandBuffer);
        m_IOCommandBuffer->clear();
    }
    }
}

uint64_t SceneRasterizer::addEllipsoid(const Ellipsoid& ellipsoid, RTMaterial* material, const RTObjectSettings* settings)
{
    TGE_ASSERT(false, "Stub"); return {};
}

uint64_t SceneRasterizer::addSphere(const Sphere& sphere, RTMaterial* material)
{
    TGE_ASSERT(false, "Stub"); return {};
}

uint64_t SceneRasterizer::addCylinder(const Cylinder& cylinder, RTMaterial* material)
{
    TGE_ASSERT(false, "Stub"); return {};
}

uint64_t SceneRasterizer::addObliqueCylinder(const ObliqueCylinder& cylinder, RTMaterial* material)
{
    TGE_ASSERT(false, "Stub"); return {};
}

void SceneRasterizer::addHair(const Matrix4& world,
                              size_t submesh_count, 
                              RTSubhair* submeshes,
                              size_t curve_count, int32_t* curves,
                              size_t vert_size, void* verts, size_t stride,
				              unsigned* geom_ids)
{
    TGE_ASSERT(false, "Stub");
}

uint64_t SceneRasterizer::addBlocker(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size)
{
    TGE_ASSERT(false, "Stub"); return {};
}

uint64_t SceneRasterizer::addRect(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size, RTMaterial* material, const AABB2* tc, const void* tangent_map)
{
    TGE_ASSERT(false, "Stub"); return {};
}

uint64_t SceneRasterizer::addDisk(const Vector3& pos, const Vector3& norm, float inner_radius, float outer_radius, RTMaterial* material)
{
    TGE_ASSERT(false, "Stub"); return {};
}

void SceneRasterizer::addHierarchicalVolume(VolumeRoot* volume, RTMaterial* material)
{
    TGE_ASSERT(false, "Stub");
}
    
void SceneRasterizer::commitScene()
{
    m_Backend->submitCommandBuffer(m_IOCommandBuffer);
    m_IOCommandBuffer->clear();

    uint64_t shader_bitfield[((size_t)IlluminationModel::Count + 63)/64] = {};

    CommandBufferDescription cmd_buf_desc;
    cmd_buf_desc.CommandCount = static_cast<uint32_t>(m_Submeshes.size());
    cmd_buf_desc.ConstantsBufferSize = cmd_buf_desc.CommandCount*1024;
    m_DeferredGeometryCommandBuffer = m_Backend->createCommandBuffer(cmd_buf_desc);

    for(auto& submesh : m_Submeshes)
    {
        size_t index = static_cast<size_t>(submesh.Model),
               major_idx = index/64,
               minor_idx = index%64;
        auto& bitfield = shader_bitfield[major_idx];
        bitfield |= minor_idx;
    }

    size_t pass_count = 0;
    for(auto bitfield : shader_bitfield)
    {
        pass_count += __popcnt64(bitfield);
    }

    uint32_t light_source_bitfield = 0;
    for(auto& light_source : m_LightSources)
    {
        switch(light_source->Type)
        {
        case LightSourceType::Directional: light_source_bitfield |= (1 << (uint32_t)SupportedLightSource::DirectionalLight);
        case LightSourceType::Point: light_source_bitfield |= (1 << (uint32_t)SupportedLightSource::PointLight);
        }
    }

    pass_count *= __popcnt(light_source_bitfield);
    cmd_buf_desc.CommandCount = static_cast<uint32_t>(pass_count);
    cmd_buf_desc.ConstantsBufferSize = pass_count*1024;
    m_DeferredShadingCommandBuffer = m_Backend->createCommandBuffer(cmd_buf_desc);

    memset(shader_bitfield, 0, sizeof(shader_bitfield));

    for(auto& submesh : m_Submeshes)
    {
        m_DeferredGeometryCommandBuffer->enqueueBatch(submesh.Batch);

        size_t model_index = static_cast<size_t>(submesh.Model),
               major_idx = model_index/64,
               minor_idx = model_index%64;
        auto& bitfield = shader_bitfield[major_idx]; 
        if((bitfield & minor_idx) == 0)
        {
            for(auto& light_src : m_LightSources)
            {
                DataFormat fmt = DataFormat::RGBA8UNorm;

                CommandBufferType::DrawBatchType batch;

                switch(static_cast<LightSourceType>(light_src->Type))
                {
                case LightSourceType::Directional:
                {
                    std::string opt = "DirectionalLight";

                    size_t state_index = model_index*(size_t)SupportedLightSource::Count + (size_t)LightSourceType::Directional;

                    auto& shader = m_DeferredShading[state_index];
                    if(!shader)
                    {
                        shader = m_ShaderCompiler->compileShaderProgram(IlluminationModelsShaderNames[model_index], nullptr, &opt, 1);
                        batch.PipelineState = m_DeferredShadingStates[state_index] = m_Backend->createStateObject(&fmt, 1, DataFormat::D24S8, shader, DrawModes::TriangleStrip, nullptr, nullptr, nullptr);
                    }

                    auto dir_light = static_cast<DirectionalLight*>(light_src);
                    auto globals = CreateResourceTable(shader, "Globals", 1);
                    globals->setResource("Globals.LightDirection", dir_light->Direction);
                    globals->setResource("Globals.Radiance", dir_light->Radiance);

                    auto table = globals->extractBakedTable();
                    m_LightSourceBakedTables.push_back(table);

                    batch.ResourceTable = table;
                    batch.VertexCount = 3;
                } break;
                case LightSourceType::Point:
                {
                    batch = m_SphereDraw;

                    std::string opt = "PointLight";
                    
                    size_t state_index = model_index*(size_t)SupportedLightSource::Count + (size_t)LightSourceType::Point;

                    auto shader = m_DeferredShading[state_index];
                    if(!shader)
                    {
                        shader = m_DeferredShading[state_index] = m_ShaderCompiler->compileShaderProgram(IlluminationModelsShaderNames[model_index], nullptr, &opt, 1);
                        batch.PipelineState = m_DeferredShadingStates[state_index] = m_Backend->createStateObject(&fmt, 1, DataFormat::D24S8, shader, DrawModes::TriangleStrip, nullptr, nullptr, nullptr);
                    }

                    auto point_light = static_cast<PointLight*>(light_src);
                    auto radiance = point_light->Radiance/Tempest::MathTau;
                    auto radius = sqrtf(MaxValue(radiance))*10.0f;

                    auto globals = CreateResourceTable(shader, "Globals", 1);
                    globals->setResource("Globals.LightGeometry", Vector4{ point_light->Position.x, point_light->Position.y, point_light->Position.z, radius });
                    globals->setResource("Globals.Radiance", radiance);
                    
                    auto table = globals->extractBakedTable();                    
                    m_LightSourceBakedTables.push_back(table);

                    batch.ResourceTable = table;
                } break;
                default: continue;
                }

                bitfield |= minor_idx;

                m_DeferredShadingCommandBuffer->enqueueBatch(batch);
            }
        }
    }
}
    
uint64_t SceneRasterizer::rayQuery(const Vector2& tc, SampleData* sample_data)
{
    TGE_ASSERT(false, "Stub"); return {};
}

void SceneRasterizer::initWorkers()
{
    m_Backend->setActiveTextures(m_DeferredShadingTextures.getSize()/(4*sizeof(float)));
}

void SceneRasterizer::addLightSource(LightSource* light)
{
    m_LightSources.push_back(light);
}
}

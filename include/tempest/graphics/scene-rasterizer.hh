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

#ifndef _TEMPEST_SCENE_RASTERIZER_HH_
#define _TEMPEST_SCENE_RASTERIZER_HH_

#include "tempest/utils/threads.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/graphics/storage-ring.hh"
#include "tempest/mesh/sslbvh2.hh"

namespace Tempest
{
class CubeMap;
struct VolumeRoot;

#define MESH_FORMAT_LIST \
    MESH_FORMAT(PNFormat) \
	MESH_FORMAT(PTNFormat) \
	MESH_FORMAT(PTTNFormat) \
	MESH_FORMAT(PcNFormat) \
	MESH_FORMAT(PTcNFormat) \
	MESH_FORMAT(PTTcNFormat)

enum class MeshFormat
{
#define MESH_FORMAT(name) name,
    MESH_FORMAT_LIST
#undef MESH_FORMAT
    Count
};

struct RasterizerSubmesh
{
    typedef PreferredBackend::CommandBufferType::DrawBatchType DrawBatchType;

    IlluminationModel                        Model;
    uint32_t                                 WorldViewProjectionMatrixOffset,
                                             RotateViewMatrixOffset;
    Matrix4                                  TransformMatrix;
    SimpleStacklessLBVH2Node<AABBUnaligned>* BVH = nullptr;
    DrawBatchType                            Batch;
};

enum class SupportedLightSource: uint32_t
{
    PointLight,
    DirectionalLight,
    Count
};

class SceneRasterizer
{
public:
    typedef PreferredShaderCompiler::ShaderProgramType ShaderType;
    typedef PreferredBackend::BufferType BufferType;
    typedef PreferredBackend::StorageType StorageType;
    typedef PreferredBackend::FenceType FenceType;
    typedef PreferredBackend::IOCommandBufferType IOCommandBufferType;
    typedef PreferredBackend::CommandBufferType CommandBufferType;
    typedef PreferredBackend::StateObjectType StateObjectType;
    typedef PreferredBackend::FramebufferType FramebufferType;
    typedef PreferredBackend::RenderTargetType RenderTargetType;
private:
    PreferredBackend*                 m_Backend;
    PreferredShaderCompiler*          m_ShaderCompiler;
    ThreadPool                        m_ThreadPool;
    uint32_t                          m_ThreadId;

    FramebufferType*                  m_DeferredShadingFramebuffer = nullptr,
                   *                  m_BackbufferFramebuffer = nullptr;
    RenderTargetType*                 m_GBuffer[3];
    RenderTargetType*                 m_GBufferDepth = nullptr;
    RenderTargetType*                 m_Backbuffer = nullptr;

    TextureTable<PreferredBackend>    m_TextureTable;

    CommandBufferType::DrawBatchType  m_SphereDraw;

    Matrix4                           m_ViewProjectionMatrix;
    Matrix4                           m_ViewProjectionInverseMatrix;

    std::vector<Vector4>              m_TextureSlots; // TODO: Meh, something better can be probably done in this case.
    std::vector<LightSource*>         m_LightSources;

    StorageRing<PreferredBackend>     m_UploadHeap;

    FenceType*                        m_Fence = nullptr;

    IOCommandBufferType*              m_IOCommandBuffer = nullptr;
    CommandBufferType*                m_DeferredGeometryCommandBuffer = nullptr,
                     *                m_DeferredShadingCommandBuffer = nullptr;

    std::vector<BufferType*>          m_Buffers;
    std::vector<RasterizerSubmesh>    m_Submeshes;
    
    ShaderType*                       m_DeferredShading[(size_t)IlluminationModel::Count];
    StateObjectType*                  m_DeferredShadingStates[(size_t)SupportedLightSource::Count*(size_t)IlluminationModel::Count];
    std::vector<BakedResourceTable*>  m_LightSourceBakedTables;

    ShaderType*                       m_MeshGbuffer[(size_t)MeshFormat::Count];
    StateObjectType*                  m_MeshGBufferStates[(size_t)MeshFormat::Count];

    ShaderType*                       m_ShaderTable[(size_t)IlluminationModel::Count];

    BakedResourceTable                m_DeferredShadingTextures;
public:
    SceneRasterizer(PreferredBackend* backend, PreferredShaderCompiler* shader_compiler, uint32_t width, uint32_t height, const Matrix4& view_proj_inv);
    ~SceneRasterizer();

    RenderTargetType* getBackbuffer() { return m_Backbuffer; }

    void setChunkSize(uint32_t) {}
    void setSamplesLocalAreaLight(uint32_t) {}
    void setSamplesGlobalIllumination(uint32_t) {}
	void setSamplesCamera(uint32_t) {}
    void setMaxRayDepth(uint32_t) {}
    void setRussianRoulette(float) {}

    void setBackgroundSpectrum(Spectrum spectrum);

    void setPicturePostProcess(PicturePostProcess mode);

    void setGlobalCubeMap(CubeMap* cube_map);
    
    void setRenderMode(RenderMode render_mode);
    RenderMode getRenderMode() const;

    void repaint() {}

    void addLightSource(LightSource* light);
    unsigned addSphereLightSource(SphereAreaLight* light);

    void clearLightSources();

    void draw(uint32_t width, uint32_t height, const Matrix4& view_proj_inv);
    void drawOnce();

    const void* bindTexture(const Texture* tex);
    void bindSurfaceAndTexture(Texture* tex, const void** tex_obj, void** surf);
	void* bindBuffer(void* buf, size_t);

    void addTriangleMesh(const Matrix4& world,
                         size_t submesh_count, 
                         RTSubmesh* submeshes,
                         size_t index_count, int32_t* tris,
                         size_t vert_size, void* verts,
                         MeshOptions* mesh_opts = nullptr,
						 uint64_t* geom_ids = nullptr);

    uint64_t addEllipsoid(const Ellipsoid& ellipsoid, RTMaterial* material, const RTObjectSettings* settings = nullptr);
    uint64_t addSphere(const Sphere& sphere, RTMaterial* material);
    uint64_t addCylinder(const Cylinder& cylinder, RTMaterial* material);
    uint64_t addObliqueCylinder(const ObliqueCylinder& cylinder, RTMaterial* material);
    void addHair(const Matrix4& world,
                 size_t submesh_count, 
                 RTSubhair* submeshes,
                 size_t curve_count, int32_t* curves,
                 size_t vert_size, void* verts, size_t stride,
				 unsigned* geom_ids = nullptr);

    uint64_t addBlocker(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size);
    uint64_t addRect(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size, RTMaterial* material, const AABB2* tc = nullptr, const void* tangent_map = nullptr);

    uint64_t addDisk(const Vector3& pos, const Vector3& norm, float inner_radius, float outer_radius, RTMaterial* material);

    void addHierarchicalVolume(VolumeRoot* volume, RTMaterial* material);
    
    void commitScene();
    
	uint64_t rayQuery(const Vector2& tc, SampleData* sample_data);

    void initWorkers();

    ThreadPool& getThreadPool() { return m_ThreadPool; }
    uint32_t    getThreadId() const { return m_ThreadId; }
private:
    void updateGBuffer(uint32_t width, uint32_t height);
};
}

#endif // _TEMPEST_SCENE_RASTERIZER_HH_
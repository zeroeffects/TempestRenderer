/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_RAY_TRACER_CUDA_HH_
#define _TEMPEST_RAY_TRACER_CUDA_HH_

#include "tempest/utils/config.hh"

#ifndef DISABLE_CUDA

#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/shapes.hh"
#include "tempest/math/vector3.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/threads.hh"

namespace Tempest
{
struct CUDATextureResource;

struct VolumeRoot;
struct RTMaterial;
struct RTVolumeCuda;
struct LightSource;
struct SphereAreaLight;

enum RayState
{
    RayIdle,
    RayActive
};

struct RTRayCuda
{
    Vector3               Origin;
    float                 Near;
    SampleData            Data;
    float                 Far;
/*  
    unsigned              X;
    unsigned              Y;
//*/
    unsigned              GeometryID;
    unsigned              PrimitiveID;
    unsigned              InstanceID;
    RayState              State;
    Spectrum              Throughput;
    Spectrum              Radiance;
};

struct RTNodeCuda
{
    AABBUnaligned         AABB;
    uint32_t              GeometryOffset;
};

class CubeMap;

class RayTracerCuda
{
    Spectrum                   m_Background = Spectrum{};
    CUDASurfaceResource*       m_Backbuffer;

    uint32_t                   m_SamplesCamera = 1;
    uint32_t                   m_MaxRayDepth = 1;
    uint32_t                   m_SamplesLocalAreaLight = 4;
    uint64_t                   m_SceneGeometryTypes = 0;

    Matrix4                    m_ViewProjectionInverse;

	std::vector<uint32_t>      m_LightSources;
	std::vector<RTVolumeCuda*> m_Volumes;
    std::vector<RTNodeCuda>    m_Boxes;
    std::vector<unsigned long long> m_InternalTextures;
	std::vector<void*>		   m_InternalBuffers;

	std::vector<unsigned long long> m_InternalSurfaces;

    RTRayCuda*                 m_GPURays = nullptr;
    RTNodeCuda*                m_GPUBoxes = nullptr;
	uint32_t*                  m_GPULightSources = nullptr;
    uint8_t*                   m_GPUGeometry = nullptr;

    size_t                     m_OccupiedTextureMemorySize = 0;
    size_t                     m_OccupiedTotalMemorySize = 0;
    size_t                     m_DeviceMemorySize = 0;

    unsigned                   m_PrevRayCount = 0;

    RenderMode                 m_RenderMode = RenderMode::Normal;

#ifndef NDEBUG
    std::unique_ptr<Tempest::Texture> m_DebugBackbuffer;
#endif

    ThreadPool                 m_Pool;
    uint32_t                   m_ThreadId;

    MemoryPoolAllocation       m_DataPool;

public:
    RayTracerCuda(CUDASurfaceResource* backbuffer, const Matrix4& view_proj_inv, const RTSettings& settings = RTSettings());
	~RayTracerCuda();

	void addHierarchicalVolume(VolumeRoot* volume, RTMaterial* material);
    void addLightSource(LightSource* light_source);
    uint64_t addRect(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size, RTMaterial* material, const AABB2* tc = nullptr, const void* tangent_texture = nullptr);

	void addTriangleMesh(const Matrix4& world,
                         size_t submesh_count, 
                         RTSubmesh* submeshes,
                         size_t tri_count, int32_t* tris,
                         size_t vert_size, void* verts,
                         MeshOptions* mesh_opts = nullptr,
						 uint64_t* geom_ids = nullptr);

    void addSphereLightSource(SphereAreaLight*);
    void setSamplesLocalAreaLight(uint32_t rays) { m_SamplesLocalAreaLight = rays; }
    void setSamplesGlobalIllumination(uint32_t rays) {}
	void setSamplesCamera(uint32_t rays) { m_SamplesCamera = rays; }
    void setMaxRayDepth(uint32_t depth) { m_MaxRayDepth = depth; }
    void setRussianRoulette(float rval) {}
    void setPicturePostProcess(PicturePostProcess mode) { /*m_PicturePostProcess = mode;*/ TGE_ASSERT(false, "Stub"); }

    const void* bindTexture(const Texture* tex);

    void bindSurfaceAndTexture(Texture* tex, const void** tex_obj, void** surf);

	void* bindBuffer(void* buf, size_t size);

    void commitScene();

    void initWorkers() {}

    void repaint() {}

    uint64_t rayQuery(const Vector2& tc, SampleData* sample_data);

    void setBackgroundSpectrum(Spectrum color) { m_Background = color; }

    void drawOnce();
    void draw(CUDASurfaceResource* backbuffer, const Matrix4& view_proj_inv);

	ThreadPool& getThreadPool() { return m_Pool; }
    uint32_t getThreadId() const { return m_ThreadId; }

    void setGlobalCubeMap(const Tempest::CubeMap* cube_map);

    RenderMode getRenderMode() const { return m_RenderMode; }
    void setRenderMode(RenderMode mode) { m_RenderMode = mode; }
};
}

#endif
#endif // _TEMPEST_RAY_TRACER_CUDA_HH_

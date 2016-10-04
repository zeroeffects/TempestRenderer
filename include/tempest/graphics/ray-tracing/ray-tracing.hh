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

#ifndef _RAY_TRACING_HH_
#define _RAY_TRACING_HH_

#include <cstdint>
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector4.hh"

#include "tempest/utils/threads.hh"
#include "tempest/utils/timer.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/volume/volume.hh"
#include "tempest/math/shapes.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"

#include <atomic>
#include <memory>
#include <thread>
#include <condition_variable>

struct RTCRay;

#ifndef __RTCORE_H__
typedef struct __RTCDevice {}* RTCDevice;
typedef struct __RTCScene* RTCScene;
#endif

namespace Tempest
{
struct VolumeRoot;    
class CubeMap;

struct RTVertex  { float x, y, z, w; };

struct RTVolume
{
    Vector3                  MinCorner;
    Vector3                  MaxCorner;
    Box                      Dimensions;
    uint8_t                 *GridData;

    ~RTVolume()
    {
        delete[] GridData;
    }
};

struct RTGeometry
{
    unsigned                 GeometryID;

    RTGeometry(unsigned geom_id)
        :   GeometryID(geom_id) {}

    // TODO: Sample more than single parameter. Is it better to put everything in one table to optimize instruction cache?
    virtual void sample(RTCRay& ray, SampleData* data) const = 0;
};

struct RTVolumeSet: public RTGeometry
{
    RTMaterial*              Material;
    Vector3                  MinCorner;
    Vector3                  MaxCorner;
	Box						 Dimensions;
    float                    RcpMaxExtinction;
	RTVolume**				 VolumeGrid;
    PACKED_DATA(RTVolume)    Volumes;

    virtual void sample(RTCRay& ray, SampleData* data) const override;

    bool sampleScattering(float threshold, const Vector3& pos, SampleData* data);

private:
    RTVolumeSet(uint32_t count, unsigned geom_id)
        :   RTGeometry(geom_id),
            Volumes(count) {}
    ~RTVolumeSet() { delete[] VolumeGrid; }
};

struct RTHair: public RTGeometry
{
    int32_t*            IndexBuffer;
    void*               VertexBuffer;
    uint32_t            Stride;
    RTMaterial*         Material;

    RTHair(unsigned geom_id, int32_t* ib, void* vb, uint32_t stride, RTMaterial* material)
        :   RTGeometry(geom_id),
            IndexBuffer(ib),
            VertexBuffer(vb),
            Stride(stride),
            Material(material) {}
    virtual void sample(RTCRay& ray, SampleData* data) const override;
};


struct RTMesh: public RTGeometry
{
    int32_t*				IndexBuffer;
    void*					VertexBuffer;
    uint32_t				Stride;
    uint32_t                TriangleCount;
    RTMaterial*				Material;
	GeometrySamplerFunction Sampler = nullptr;
    void*                   UserData = nullptr;

    RTMesh(unsigned geom_id, int32_t* ib, void* vb, uint32_t tri_count, uint32_t stride, RTMaterial* material)
        :   RTGeometry(geom_id),
            IndexBuffer(ib),
            VertexBuffer(vb),
            Stride(stride),
            Material(material) {}
    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

class RayTracerScene;

struct FrameData
{
    friend class RayTracerScene;

    Matrix4                       ViewProjectionInverse;
    std::unique_ptr<Texture>      Backbuffer;
private:
    std::atomic_int               Counter;
};

struct SphereAreaLightGeometry: public RTGeometry
{
    SphereAreaLight*             Light;
    RTMicrofacetMaterial         Material;
    
    SphereAreaLightGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct EllipsoidGeometry: public RTGeometry
{
    Ellipsoid                   EllipsoidShape;
    RTMaterial                  *Material;
    AABBUnaligned               Bounds;

    EllipsoidGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct SphereGeometry: public RTGeometry
{
    Sphere                       SphereShape;
    RTMaterial                   *Material;
    SphereGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct CylinderGeometry: public RTGeometry
{
    Cylinder                     CylinderShape;
    RTMaterial                   *Material;

    CylinderGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct ObliqueCylinderGeometry: public RTGeometry
{
    ObliqueCylinder              ObliqueCylinderShape;
    RTMaterial                   *Material;

    ObliqueCylinderGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct BlockerGeometry: public RTGeometry
{
    Rect3                        BlockerRect;

    BlockerGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct RectGeometry: public RTGeometry
{
    Rect3                        Rect;
    Vector2                      TexCoordStart;
    Vector2                      TexCoordMultiplier;
    RTMaterial                   *Material;
    const Texture                *TangentMap;

    RectGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

struct DiskGeometry: public RTGeometry
{
    Disk3                        Disk;
    RTMaterial                   *Material;

    DiskGeometry(unsigned geom_id)
        :   RTGeometry(geom_id) {}

    virtual void sample(RTCRay& ray, SampleData* data) const override;
};

class RayTracerScene
{
    static const uint32_t Buffering = 2;

    TimeQuery                         m_FrameTimer;

    uint64_t                          m_FrameStart;
  
    RTCDevice                         m_Device;
    RTCScene                          m_Scene;
    
    CubeMap                          *m_GlobalCubeMap = nullptr;

    Spectrum                          m_BackgroundSpectrum = Spectrum{};
    uint32_t                          m_ChunkSize = 16;
    uint32_t                          m_FrameIndex = 0;
    uint32_t                          m_SamplesGlobalIllumination = 64;
	uint32_t                          m_SamplesLocalAreaLight = 4;
	uint32_t                          m_SamplesCamera = 1; // Increse to enable anti-aliasing, but remove other type of samples
	uint32_t                          m_MaxRayDepth = 1;
    float                             m_RussianRoulette = 1.0f;
    bool                              m_TransparentBackground = false;

    RenderMode                        m_RenderMode = RenderMode::Normal;

	std::unique_ptr<uint8_t[]>        m_ScratchMemory;

	ThreadPool                        m_ThreadPool;
	uint32_t                          m_ThreadId;

    FrameData                         m_FrameData[Buffering];
    
    PicturePostProcess                m_PicturePostProcess = PicturePostProcess::SRGB;

	class DrawTask
	{
		RayTracerScene* m_RayScene; 
	public:
		DrawTask(RayTracerScene* ray_scene)
			:	m_RayScene(ray_scene) {}

		void operator()(uint32_t worker_id, uint32_t idx, uint32_t chunk_size) { m_RayScene->subdraw(worker_id, idx, chunk_size); }
	};

	ParallelForLoop<DrawTask>		  m_DrawLoop;

	Spectrum						 *m_AccumBuffer = nullptr;
	int								  m_AccumData = 0;

    std::unique_ptr<Spectrum[]>       m_LogAverages;
    uint32_t                          m_WorkerThreads;

    std::vector<uint8_t*>             m_VertexCache;
    std::vector<RTVolumeSet*>         m_Volumes;

    // TODO: Shirley, et al or similar approach to sampling multiple light sources
    std::vector<LightSource*>         m_LightSources;
    std::vector<RTGeometry*>          m_InternalGeometry;

    MemoryPoolAllocation              m_DataPool;
public:
    RayTracerScene(uint32_t width, uint32_t height, const Matrix4& view_proj_inv, const RTSettings& settings = RTSettings());
    ~RayTracerScene();

    void setChunkSize(uint32_t chunk_size) { m_ChunkSize = chunk_size; }
    void setSamplesLocalAreaLight(uint32_t rays) { m_SamplesLocalAreaLight = rays; }
    void setSamplesGlobalIllumination(uint32_t rays) { m_SamplesGlobalIllumination = rays; }
	void setSamplesCamera(uint32_t rays) { m_SamplesCamera = rays; }
    void setMaxRayDepth(uint32_t depth) { m_MaxRayDepth = depth; }
    void setRussianRoulette(float rval) { m_RussianRoulette = rval; }

    void setBackgroundSpectrum(Spectrum spectrum) { m_BackgroundSpectrum = spectrum; }

    void setPicturePostProcess(PicturePostProcess mode) { m_PicturePostProcess = mode; }

    // It doesn't actually manage the cube map
    void setGlobalCubeMap(CubeMap* cube_map) { m_GlobalCubeMap = cube_map; }
    
    void setTransparentBackground(bool enabled) { m_TransparentBackground = enabled; }

    void setRenderMode(RenderMode render_mode) { m_RenderMode = render_mode; }
    RenderMode getRenderMode() const { return m_RenderMode; }

    void repaint() { m_AccumData = -1; }

    unsigned addLightSource(LightSource* light) { auto light_index = m_LightSources.size(); m_LightSources.push_back(light); return (unsigned)light_index; }
    void updateLightSource(unsigned id, LightSource* light) { repaint(); TGE_ASSERT(id < m_LightSources.size(), "Invalid light source"); m_LightSources[id] = light; }
    unsigned addSphereLightSource(SphereAreaLight* light);

    void clearLightSources() { m_LightSources.clear(); }

    // TODO: something more sophisticated for dynamic scenes
    MemoryPool getPool() { return m_DataPool.getBase(); }
    template<class T> PoolPtr<T> allocateFromPool() { return m_DataPool.allocate<T>(); }

    const FrameData* draw(uint32_t width, uint32_t height, const Matrix4& view_proj_inv);
    const FrameData* drawOnce();

    // TODO: Tiling
    const void* bindTexture(const Texture* tex)
    {
        return tex;
    }

    void bindSurfaceAndTexture(Texture* tex, const void** tex_obj, void** surf)
    {
        *tex_obj = tex;
        *surf = tex;
    }

	void* bindBuffer(void* buf, size_t)
	{
		return buf;
	}

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

    // We don't have a pure rect because it is sort of annoying to handle two-sided polygons
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
    void subdraw(uint32_t worker_id, uint32_t idx, uint32_t chunk_size);
    void postprocess();
};
}

#endif // _RAY_TRACING_HH_
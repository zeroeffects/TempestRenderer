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

#ifndef NDEBUG
//#   define CPU_DEBUG
#   define RAY_TRACER
#endif

#ifdef CPU_DEBUG
#   define EXECUTE_PARALLEL_FOR_LOOP_2D ExecuteParallelForLoop2DCPU
#   define EXECUTE_PARALLEL_FOR_LOOP ExecuteParallelForLoopCPU
#   define EXPORT_DEVICE __host__
#   define GENERIC_FREE free
#   define SHARED_CODE __host__

cudaError_t cudaMallocReplacement(void** ptr, size_t size)
{
    *ptr = malloc(size);
    return *ptr != nullptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

#   define GENERIC_MALLOC cudaMallocReplacement
#   define GENERIC_MEMSET memset
#else
#   define ILLUMINATION_MODEL_IMPLEMENTATION
#   define EXECUTE_PARALLEL_FOR_LOOP_2D ExecuteParallelForLoop2DGPU
#   define EXECUTE_PARALLEL_FOR_LOOP ExecuteParallelForLoopGPU
#   define EXPORT_DEVICE __device__
#   define EXPORT_TABLE __device__
#   define SHARED_CODE __host__ __device__
#   define GENERIC_FREE cudaFree
#   define GENERIC_MALLOC cudaMalloc
#   define GENERIC_MEMSET cudaMemset
#endif

#include "tempest/compute/compute-definitions.hh"
#include "tempest/compute/ray-tracing-cuda.hh"
#include "tempest/utils/logging.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/volume/volume.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/math/intersect.hh"
#include "tempest/utils/memory.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/mesh/lbvh2.hh"
#include "tempest/mesh/sslbvh2.hh"
#include "tempest/compute/compute-texture.hh"
#include "tempest/compute/compute-convenience.hh"

#include "cuda_gl_interop.h"
#include "cuda_runtime_api.h"

#define ENABLE_MIS 1
//#define ENABLE_MIS_POWER_HEURISTIC 1

#ifdef ENABLE_MIS_POWER_HEURISTIC
#	define BalanceBias(x) (x)*(x)
#else
#	define BalanceBias(x) (x)
#endif

namespace Tempest
{
#ifndef CPU_DEBUG
#   ifndef NDEBUG
    const size_t CudaStackSize = 2*1024;
#   else
    const size_t CudaStackSize = 1024;
#   endif
#endif

enum class RTGeometryType: uint32_t
{
    Mesh,
    Rect,
    Sphere,
    Cylinder,
    ObliqueCylinder,
    Volume,
    TwoSidedMesh,
    Count
};

struct GeometryDescriptionHeader
{
    unsigned                  GeometryType;

	GeometryDescriptionHeader(unsigned _type)
		:	GeometryType(_type) {}
};

struct Rect3Geometry: public GeometryDescriptionHeader
{
    Rect3                     Rect;
    Vector2                   TexCoordStart;
    Vector2                   TexCoordMultiplier;
    const void*     		  TangentMap;
    RTMaterial				  Material[];
    
	Rect3Geometry()
		:	GeometryDescriptionHeader((unsigned)RTGeometryType::Rect) {}
};

struct SphereGeometry: public GeometryDescriptionHeader
{
    Sphere                    SphereShape;
	RTMaterial				  Material[];

	SphereGeometry()
		:	GeometryDescriptionHeader((unsigned)RTGeometryType::Sphere) {}
};

struct SubmeshGeometry
{
	uint32_t                 VertexCount;
    uint32_t                 VertexOffset;
    uint32_t                 BaseIndex;
	uint32_t				 Stride;
	uint32_t				 BVHOffset;
	uint32_t				 MaterialOffset;
};

struct MeshGeometry: public GeometryDescriptionHeader
{
	Matrix4					  InverseTransform;
	uint32_t				  SubmeshCount;
	uint32_t				  IndexCount;
	uint32_t				  VertexSize;
	uint32_t				  SubmeshOffset;
};

typedef bool (*IntersectTestFunction)(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
typedef void (*GeometrySampleFunction)(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);

__device__ __host__ bool MeshIntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ bool MeshOcclusionTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ void MeshSampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);

__device__ __host__ bool TwoSidedMeshIntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ void TwoSidedMeshSampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ bool TwoSidedMeshOcclusionTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);

__device__ __host__ bool Rect3IntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ void RectGeometrySampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);

__device__ __host__ bool SphereGeometryIntersectCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);
__device__ __host__ void SphereGeometrySampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray);

#define WRAP_DIRECT_GEOMETRY(name, intersect, sample) \
    struct DirectGeometry##name { \
        inline EXPORT_CUDA static bool intersectTestFunction(const GeometryDescriptionHeader& geom, RTRayCuda* ray) { return intersect(geom, ray); } \
        inline EXPORT_CUDA static void geometrySampleFunction(const GeometryDescriptionHeader& geom, RTRayCuda* ray) { return sample(geom, ray); } };

#define INTERSECT_TESTS_LIST {\
    MeshIntersectTestCuda, /*Mesh,*/ \
    Rect3IntersectTestCuda, /*Rect,*/ \
    SphereGeometryIntersectCuda, /*Sphere,*/ \
    nullptr, /*Cylinder,*/ \
    nullptr, /*ObliqueCylinder,*/ \
    nullptr, /*Volume,*/ \
    TwoSidedMeshIntersectTestCuda, /*Mesh,*/ \
}

#define OCCLUSION_TESTS_LIST {\
    MeshOcclusionTestCuda, /*Mesh,*/ \
    Rect3IntersectTestCuda, /*Rect,*/ \
    SphereGeometryIntersectCuda, /*Sphere,*/ \
    nullptr, /*Cylinder,*/ \
    nullptr, /*ObliqueCylinder,*/ \
    nullptr, /*Volume,*/ \
    TwoSidedMeshOcclusionTestCuda, /*Mesh,*/ \
}

#define GEOMETRY_SAMPLE_LIST {\
    MeshSampleCuda, /*Triangles,*/ \
    RectGeometrySampleCuda, /*Rect,*/ \
    SphereGeometrySampleCuda, /*Sphere,*/ \
    nullptr, /*Cylinder,*/ \
    nullptr, /*ObliqueCylinder,*/ \
    nullptr, /*Volume,*/ \
    TwoSidedMeshSampleCuda, /*MeshTwoSided,*/ \
}

WRAP_DIRECT_GEOMETRY(Mesh, MeshIntersectTestCuda, MeshSampleCuda);

IntersectTestFunction IntersectTestsCPU[(size_t)RTGeometryType::Count] = INTERSECT_TESTS_LIST;
IntersectTestFunction OcclusionTestsCPU[(size_t)RTGeometryType::Count] = OCCLUSION_TESTS_LIST;
GeometrySampleFunction GeometrySampleCPU[(size_t)RTGeometryType::Count] = GEOMETRY_SAMPLE_LIST;
#ifndef CPU_DEBUG
__device__ IntersectTestFunction IntersectTestsGPU[(size_t)RTGeometryType::Count] = INTERSECT_TESTS_LIST;
__device__ IntersectTestFunction OcclusionTestsGPU[(size_t)RTGeometryType::Count] = OCCLUSION_TESTS_LIST;
__device__ GeometrySampleFunction GeometrySampleGPU[(size_t)RTGeometryType::Count] = GEOMETRY_SAMPLE_LIST;
#endif

#if defined(__CUDA_ARCH__) && !defined(CPU_DEBUG)
#   define INTERSECT_TESTS IntersectTestsGPU
#   define OCCLUSION_TESTS OcclusionTestsGPU
#   define GEOMETRY_SAMPLE GeometrySampleGPU
#else
#   define INTERSECT_TESTS IntersectTestsCPU
#   define OCCLUSION_TESTS OcclusionTestsCPU
#   define GEOMETRY_SAMPLE GeometrySampleCPU
#endif

struct RTVolumeCuda
{
	cudaArray*			BrickPoolArray = nullptr;
	cudaArray*			RootBrickArray = nullptr;

	cudaTextureObject_t BrickPool = 0;
	cudaTextureObject_t RootBrick = 0;

    Box                 Dimensions;
    Box                 BrickDimensions;

	~RTVolumeCuda()
	{
		if(BrickPool)
		{
			cudaDestroyTextureObject(BrickPool);
		}
		if(RootBrick)
		{
			cudaDestroyTextureObject(RootBrick);
		}
		if(BrickPoolArray)
		{
			cudaFreeArray(BrickPoolArray);
		}
		if(RootBrickArray)
		{
			cudaFreeArray(RootBrickArray);
		}
	}
};

__device__ __host__ bool MeshIntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
	auto* ptr = reinterpret_cast<const uint8_t*>(&geom_header);
	auto& mesh = static_cast<const MeshGeometry&>(geom_header);

	ptr += sizeof(MeshGeometry);
	auto* indices = reinterpret_cast<const uint32_t*>(ptr);
	ptr += mesh.IndexCount*sizeof(uint32_t);
	auto* verts = reinterpret_cast<const uint8_t*>(ptr);
	ptr += mesh.VertexSize;

	auto* submeshes = reinterpret_cast<const SubmeshGeometry*>(ptr);
	ptr += mesh.SubmeshCount*sizeof(SubmeshGeometry);
	auto* var_size_area = ptr;

    auto transform_inc_light = mesh.InverseTransform * ray->Data.IncidentLight;
    auto transform_origin  = mesh.InverseTransform * ray->Origin;
	RayIntersectData intersect_data{ transform_inc_light, transform_origin, ray->Near, ray->Far };
	bool status = false;
    float intersect_dist = INFINITY;
	for(uint32_t submesh_idx = 0; submesh_idx < mesh.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];
		auto* bvh = reinterpret_cast<const SimpleStacklessLBVH2Node<AABBUnaligned>*>(var_size_area + submesh.BVHOffset);

        IntersectTriangleQuery3DCull intersect_tri{ verts + submesh.VertexOffset, submesh.Stride, indices + submesh.BaseIndex };
		intersect_tri.IntersectDistance = intersect_dist;

		IntersectSSLBVHNode(bvh, intersect_data, intersect_tri);
        if(intersect_tri.IntersectDistance != intersect_dist)
        {
            ray->PrimitiveID = intersect_tri.PrimitiveID;
            ray->InstanceID = submesh_idx;
            ray->Data.Normal = Normalize(intersect_tri.Normal);
            ray->Far = intersect_tri.IntersectDistance;
            ray->Data.TexCoord = { intersect_tri.BarycentricCoordinates.x, intersect_tri.BarycentricCoordinates.y };
            intersect_dist = intersect_tri.IntersectDistance;
            status = true;
        }
	}

	return status;
}

__device__ __host__ bool MeshOcclusionTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
    auto* ptr = reinterpret_cast<const uint8_t*>(&geom_header);
	auto& mesh = static_cast<const MeshGeometry&>(geom_header);

	ptr += sizeof(MeshGeometry);
	auto* indices = reinterpret_cast<const uint32_t*>(ptr);
	ptr += mesh.IndexCount*sizeof(uint32_t);
	auto* verts = reinterpret_cast<const uint8_t*>(ptr);
	ptr += mesh.VertexSize;

	auto* submeshes = reinterpret_cast<const SubmeshGeometry*>(ptr);
	ptr += mesh.SubmeshCount*sizeof(SubmeshGeometry);
	auto* var_size_area = ptr;

    auto transform_inc_light = mesh.InverseTransform * ray->Data.IncidentLight;
    auto transform_origin  = mesh.InverseTransform * ray->Origin;
	RayIntersectData intersect_data{ transform_inc_light, transform_origin, ray->Near, ray->Far };
	for(uint32_t submesh_idx = 0; submesh_idx < mesh.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];
		auto* bvh = reinterpret_cast<const SimpleStacklessLBVH2Node<AABBUnaligned>*>(var_size_area + submesh.BVHOffset);

        IntersectTriangleQuery3DCull intersect_tri{ verts + submesh.VertexOffset, submesh.Stride, indices + submesh.BaseIndex };
		intersect_tri.IntersectDistance = INFINITY;

		IntersectSSLBVHNodeSingle(bvh, intersect_data, intersect_tri);
        if(intersect_tri.IntersectDistance != INFINITY)
        {
            ray->PrimitiveID = intersect_tri.PrimitiveID;
            ray->InstanceID = submesh_idx;
            ray->Data.Normal = Normalize(intersect_tri.Normal);
            ray->Far = intersect_tri.IntersectDistance;
            ray->Data.TexCoord = { intersect_tri.BarycentricCoordinates.x, intersect_tri.BarycentricCoordinates.y };
            return true;
        }
	}
    return false;
}

__device__ __host__ bool TwoSidedMeshIntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
	auto* ptr = reinterpret_cast<const uint8_t*>(&geom_header);
	auto& mesh = static_cast<const MeshGeometry&>(geom_header);

	ptr += sizeof(MeshGeometry);
	auto* indices = reinterpret_cast<const uint32_t*>(ptr);
	ptr += mesh.IndexCount*sizeof(uint32_t);
	auto* verts = reinterpret_cast<const uint8_t*>(ptr);
	ptr += mesh.VertexSize;

	auto* submeshes = reinterpret_cast<const SubmeshGeometry*>(ptr);
	ptr += mesh.SubmeshCount*sizeof(SubmeshGeometry);
	auto* var_size_area = ptr;

    auto transform_inc_light = mesh.InverseTransform * ray->Data.IncidentLight;
    auto transform_origin  = mesh.InverseTransform * ray->Origin;
	RayIntersectData intersect_data{ transform_inc_light, transform_origin, ray->Near, ray->Far };
	bool status = false;
    float intersect_dist = INFINITY;
	for(uint32_t submesh_idx = 0; submesh_idx < mesh.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];
		auto* bvh = reinterpret_cast<const SimpleStacklessLBVH2Node<AABBUnaligned>*>(var_size_area + submesh.BVHOffset);

        IntersectTriangleQuery3DTwoSided intersect_tri{ verts + submesh.VertexOffset, submesh.Stride, indices + submesh.BaseIndex };
		intersect_tri.IntersectDistance = intersect_dist;

		IntersectSSLBVHNode(bvh, intersect_data, intersect_tri);
        if(intersect_tri.IntersectDistance != intersect_dist)
        {
            ray->PrimitiveID = intersect_tri.PrimitiveID;
            ray->InstanceID = submesh_idx;
            ray->Data.Normal = Normalize(intersect_tri.Normal);
            ray->Far = intersect_tri.IntersectDistance;
            ray->Data.TexCoord = { intersect_tri.BarycentricCoordinates.x, intersect_tri.BarycentricCoordinates.y };
            intersect_dist = intersect_tri.IntersectDistance;
            status = true;
        }
	}

	return status;
}

__device__ __host__ bool TwoSidedMeshOcclusionTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
	auto* ptr = reinterpret_cast<const uint8_t*>(&geom_header);
	auto& mesh = static_cast<const MeshGeometry&>(geom_header);

	ptr += sizeof(MeshGeometry);
	auto* indices = reinterpret_cast<const uint32_t*>(ptr);
	ptr += mesh.IndexCount*sizeof(uint32_t);
	auto* verts = reinterpret_cast<const uint8_t*>(ptr);
	ptr += mesh.VertexSize;

	auto* submeshes = reinterpret_cast<const SubmeshGeometry*>(ptr);
	ptr += mesh.SubmeshCount*sizeof(SubmeshGeometry);
	auto* var_size_area = ptr;

    auto transform_inc_light = mesh.InverseTransform * ray->Data.IncidentLight;
    auto transform_origin  = mesh.InverseTransform * ray->Origin;
	RayIntersectData intersect_data{ transform_inc_light, transform_origin, ray->Near, ray->Far };
	for(uint32_t submesh_idx = 0; submesh_idx < mesh.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];
		auto* bvh = reinterpret_cast<const SimpleStacklessLBVH2Node<AABBUnaligned>*>(var_size_area + submesh.BVHOffset);

        IntersectTriangleQuery3DTwoSided intersect_tri{ verts + submesh.VertexOffset, submesh.Stride, indices + submesh.BaseIndex };
		intersect_tri.IntersectDistance = INFINITY;

		IntersectSSLBVHNodeSingle(bvh, intersect_data, intersect_tri);
        if(intersect_tri.IntersectDistance != INFINITY)
        {
            ray->PrimitiveID = intersect_tri.PrimitiveID;
            ray->InstanceID = submesh_idx;
            ray->Data.Normal = Normalize(intersect_tri.Normal);
            ray->Far = intersect_tri.IntersectDistance;
            ray->Data.TexCoord = { intersect_tri.BarycentricCoordinates.x, intersect_tri.BarycentricCoordinates.y };
            return true;
        }
	}

	return false;
}

__device__ __host__ bool Rect3IntersectTestCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
    auto& rect_geom = static_cast<const Rect3Geometry&>(geom_header);
    Tempest::Vector2 tc;
    auto status = IntersectRect3(ray->Data.IncidentLight, ray->Origin, rect_geom.Rect, &ray->Far, &tc.x, &tc.y, &ray->Data.Normal);
    ray->Data.TexCoord = tc*rect_geom.TexCoordMultiplier + rect_geom.TexCoordStart;
    ray->InstanceID = 0;
    return status;
}

__device__ __host__ void RectGeometrySampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
    auto& rect_geom = static_cast<const Rect3Geometry&>(geom_header);
    ray->Data.Material = rect_geom.Material;

    if(rect_geom.TangentMap)
    {
        Tempest::Matrix3 rel_tangent_space;
        auto tangent = SampleRGB(rect_geom.TangentMap, ray->Data.TexCoord);
        NormalizeSelf(&tangent);
        rel_tangent_space.makeBasisTangent(tangent);

        Tempest::Matrix3 mod_tangent_space = ToMatrix3(rect_geom.Rect.Orientation);
        mod_tangent_space *= rel_tangent_space;

        ray->Data.Tangent = mod_tangent_space.tangent();
        ray->Data.Binormal = mod_tangent_space.binormal();
        ray->Data.Normal = mod_tangent_space.normal();
    }
    else
    {
        Matrix3 tangent_space = ToMatrix3(rect_geom.Rect.Orientation);
        ray->Data.Tangent = tangent_space.tangent();
        ray->Data.Binormal = tangent_space.binormal();
        ray->Data.Normal = tangent_space.normal();
    }
}

__device__ __host__ bool SphereGeometryIntersectCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
	auto& sphere = static_cast<const SphereGeometry&>(geom_header);
    ray->InstanceID = 0;
	return IntersectSphere(ray->Data.IncidentLight, ray->Origin, sphere.SphereShape, &ray->Far, &ray->Data.TexCoord.x, &ray->Data.TexCoord.y, &ray->Data.Normal);
}

__device__ __host__ void MeshSampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
    auto prim_idx = ray->PrimitiveID;
    auto inst_idx = ray->InstanceID;

    auto* ptr = reinterpret_cast<const uint8_t*>(&geom_header);
	auto& mesh = static_cast<const MeshGeometry&>(geom_header);

	ptr += sizeof(MeshGeometry);
	auto* indices = reinterpret_cast<const uint32_t*>(ptr);
	ptr += mesh.IndexCount*sizeof(uint32_t);
	auto* verts = reinterpret_cast<const uint8_t*>(ptr);
	ptr += mesh.VertexSize;

	auto* submeshes = reinterpret_cast<const SubmeshGeometry*>(ptr);
	ptr += mesh.SubmeshCount*sizeof(SubmeshGeometry);
	auto* var_size_area = ptr;

    auto& cur_submesh = submeshes[inst_idx];
    auto base_idx = cur_submesh.BaseIndex;
    auto stride = cur_submesh.Stride;

	verts += cur_submesh.VertexOffset;

	size_t normal_offset = stride - 3*sizeof(float);
    uint32_t index0 = indices[prim_idx*3 + base_idx];
    uint32_t index1 = indices[prim_idx*3 + 1 + base_idx];
    uint32_t index2 = indices[prim_idx*3 + 2 + base_idx];
    auto norm0 = (Vector3*)((char*)verts + index0*stride + normal_offset);
    auto norm1 = (Vector3*)((char*)verts + index1*stride + normal_offset);
    auto norm2 = (Vector3*)((char*)verts + index2*stride + normal_offset);

    float u = ray->Data.TexCoord.x, v = ray->Data.TexCoord.y, w = 1.0f - u - v;

	auto norm = Normalize(*norm0*w + *norm1*u + *norm2*v);
    if(stride >= sizeof(PTNFormat))
    {
        auto& tc0 = reinterpret_cast<const PTNFormat*>(verts + index0*stride)->TexCoord;
        auto& tc1 = reinterpret_cast<const PTNFormat*>(verts + index1*stride)->TexCoord;
        auto& tc2 = reinterpret_cast<const PTNFormat*>(verts + index2*stride)->TexCoord;

        ray->Data.TexCoord = tc0*w + tc1*u + tc2*v;

#if 1
		if(stride == sizeof(PTTcNFormat))
		{
			auto a0 = reinterpret_cast<const PTTcNFormat*>(verts + index0*stride)->InterpolationConstant;
			auto a1 = reinterpret_cast<const PTTcNFormat*>(verts + index1*stride)->InterpolationConstant;
			auto a2 = reinterpret_cast<const PTTcNFormat*>(verts + index2*stride)->InterpolationConstant;

			float a_coef = a0*w + a1*u + a2*v;
			
			norm = ComputeConsistentNormal(-ray->Data.IncidentLight, norm, a_coef);
		}
#endif

		if(stride >= sizeof(PTTNFormat))
		{
			auto& tan0 = reinterpret_cast<const PTTNFormat*>(verts + index0*stride)->Tangent;
			auto& tan1 = reinterpret_cast<const PTTNFormat*>(verts + index1*stride)->Tangent;
			auto& tan2 = reinterpret_cast<const PTTNFormat*>(verts + index2*stride)->Tangent;

			auto tangent = tan0*w + tan1*u + tan2*v;

			Tempest::Matrix3 basis;
			basis.makeBasisOrthogonalize(tangent, norm);

			ray->Data.Tangent = basis.tangent();
			ray->Data.Binormal = basis.binormal();
		}
    }
    else
    {
        ray->Data.TexCoord = Vector2{ 0.0f, 0.0f };
    }

	ray->Data.Normal = norm;

    ray->Data.Material = reinterpret_cast<const RTMaterial*>(var_size_area + cur_submesh.MaterialOffset);
}

__device__ __host__ void TwoSidedMeshSampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
    auto prev_norm = ray->Data.Normal;
    MeshSampleCuda(geom_header, ray);
    if(Dot(prev_norm, ray->Data.Normal) < 0.0f)
        ray->Data.Normal = -ray->Data.Normal;
}

__device__ __host__ void SphereGeometrySampleCuda(const GeometryDescriptionHeader& geom_header, RTRayCuda* ray)
{
	auto& sphere = static_cast<const SphereGeometry&>(geom_header);
	ray->Data.Material = sphere.Material;
    Vector3 norm = ray->Origin + ray->Data.IncidentLight*ray->Far - sphere.SphereShape.Center;
    NormalizeSelf(&norm);
	ray->Data.Normal = norm;
	ray->Data.Tangent = Vector3{0.0f, 0.0f, 0.0f};
	ray->Data.Binormal = Vector3{0.0f, 0.0f, 0.0f};
}

RayTracerCuda::RayTracerCuda(CUDASurfaceResource* backbuffer, const Matrix4& view_proj_inv, const RTSettings& settings)
    :   m_Backbuffer(backbuffer),
        m_ViewProjectionInverse(view_proj_inv),
        m_ThreadId(m_Pool.allocateThreadNumber()),
        m_DataPool(settings.DataPoolSize)
{
    InitSpectrum();

    cudaError_t err;

#ifndef CPU_DEBUG
 
#if SPECTRUM_SAMPLES != 3
    cudaMemcpy(CIE_ReducedSpectrumGPU, CIE_ReducedSpectrumCPU, sizeof(CIE_ReducedSpectrumCPU), cudaMemcpyHostToDevice);
    cudaMemcpy(SpectrumCurveSetGPU, SpectrumCurveSetCPU, sizeof(SpectrumCurveSetCPU), cudaMemcpyHostToDevice);
#else
    size_t aligned_size = AlignAddress(sizeof(CIE_SpectrumCPU), 256);
    err = cudaMalloc(&CIE_SpectrumGPU, aligned_size);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Fatal, "Failed to allocate CIE conversion table");
        return;
    }

    err = cudaMemcpyToSymbol(CIE_SpectrumGPU, CIE_SpectrumCPU, sizeof(CIE_SpectrumCPU), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to upload CIE conversion table");
    }
#endif

    err = cudaDeviceSetLimit(cudaLimitStackSize, CudaStackSize);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to set device stack size");
    }
    //size_t stack_size;
    //cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

    cudaThreadSynchronize();
    err = cudaGetLastError();

#else
    Tempest::TextureDescription bb_tex_desc;
    bb_tex_desc.Width = backbuffer->Texture.Description.Width;
    bb_tex_desc.Height = backbuffer->Texture.Description.Height;
    bb_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    m_DebugBackbuffer = decltype(m_DebugBackbuffer)(new Tempest::Texture(bb_tex_desc, reinterpret_cast<uint8_t*>(new uint32_t[bb_tex_desc.Width*bb_tex_desc.Height])));
#endif

    cudaDeviceProp device_prop;
    err = cudaGetDeviceProperties(&device_prop, 0);
    if(err == cudaSuccess)
    {
        m_DeviceMemorySize = device_prop.totalGlobalMem;
    }
}

RayTracerCuda::~RayTracerCuda()
{
	for(auto& rt_vol : m_Volumes)
	{
		delete rt_vol;
	}

#ifndef CPU_DEBUG
    for(auto cuda_tex : m_InternalTextures)
    {
        CudaTextureDeleter(cuda_tex);
    }

	for(auto cuda_surf : m_InternalSurfaces)
	{
		cudaDestroySurfaceObject(cuda_surf);
	}

    cudaFree(m_GPUBoxes);

#endif

    cudaFree(m_GPURays);
}

inline EXPORT_CUDA float frac(float x)
{
    return x - truncf(x);
}

__global__ void VolumeDebug(cudaTextureObject_t root_brick,
                            cudaTextureObject_t brick_pool,
                            cudaSurfaceObject_t backbuffer,
                            float3 vol_dims,
                            float3 brick_size,
                            dim3 tex_dims,
                            unsigned int fill_color)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= tex_dims.x && y >= tex_dims.y)
    {
        surf2Dwrite(fill_color, backbuffer, x * 4, y);
        return;
    }
        
    float2 tex_dims_f = make_float2(float(tex_dims.x), float(tex_dims.y));

    float x_tc = float(x)*vol_dims.x/tex_dims_f.x;
    float y_tc = float(y)*vol_dims.z/tex_dims_f.y;
    unsigned brick_ptr = tex3D<unsigned>(root_brick, x_tc, 1.0, y_tc);

    if(brick_ptr == ~0)
    {
        surf2Dwrite(fill_color, backbuffer, x * 4, y);
        return;
    }

    unsigned brick_x = brick_ptr & 0xFFFF;
    unsigned brick_y = (brick_ptr >> 16) & 0xFFFF;

    float x_brick_tc = frac(x_tc);
    float y_brick_tc = frac(y_tc);

    unsigned char density_value = tex3D<unsigned char>(brick_pool, brick_size.x*(brick_x + x_brick_tc), brick_size.y*brick_y, brick_size.z*y_brick_tc);

    unsigned density_mask = density_value < 128;
    unsigned color = density_mask*fill_color;

    surf2Dwrite(color, backbuffer, x * 4, y);
}

struct GenerateCameraRaysInfo
{
    Matrix4             ViewProjectionInverse;
    dim3                ScreenDims;
    unsigned            RayCount;
    RTRayCuda*          Rays;
    unsigned int        SamplesCamera;
};

struct RayTraceData
{
    unsigned            PassSeed;
    RTRayCuda*          Rays;
    unsigned            RayCount;
    RTNodeCuda*         Boxes;
    uint8_t*            Geometry;
    unsigned int        BoxCount;
	uint32_t*           LightSources;
	unsigned int        LightSourceCount;
    Spectrum            Background;
};

struct GenerateCameraRays
{
    GenerateCameraRaysInfo Info;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Info.RayCount)
        {
            return;
        }

        //unsigned camera_ray_idx = ray_idx % Info.SamplesCamera;
        unsigned ray_screen_idx = ray_idx / Info.SamplesCamera;
        unsigned ray_x = ray_screen_idx % Info.ScreenDims.x;
        unsigned ray_y = ray_screen_idx / Info.ScreenDims.x;

        Vector4 screen_tc;
        float offset_weight = Info.SamplesCamera > 1 ? 1.0f : 0.0f;

        unsigned seed = ray_idx;

        auto& ray = Info.Rays[ray_idx];
        screen_tc = Vector4{2.0f*(ray_x + offset_weight*FastFloatRand(seed))/Info.ScreenDims.x - 1.0f,
                            2.0f*(ray_y + offset_weight*FastFloatRand(seed))/Info.ScreenDims.y - 1.0f,
                            -1.0f, 1.0};

        Vector4 pos_start = Info.ViewProjectionInverse*screen_tc;
	    screen_tc.z = 1.0f;
	    Vector4 pos_end = Info.ViewProjectionInverse*screen_tc;

        Vector3 start_ray_pos = ToVector3(pos_start);
	    Vector3 end_ray_pos = ToVector3(pos_end);

	    Vector3 inc_light = end_ray_pos - start_ray_pos;
	    NormalizeSelf(&inc_light);

        ray.Origin = start_ray_pos;
        ray.Data.IncidentLight = inc_light;
        ray.State = RayActive;
        ray.Throughput = ToSpectrum(1.0f/Info.SamplesCamera);
        ray.Radiance = {};
        /*
        ray.X = ray_x;
        ray.Y = ray_y;
        //*/
        ray.Near = 0.0f;
        ray.Far = INFINITY;
        ray.GeometryID = INVALID_GEOMETRY;
    }
};

SHARED_CODE bool OcclusionBVH(RTNodeCuda* boxes, unsigned box_count, uint8_t* geom_data, unsigned ignore_id, RTRayCuda* inout_ray)
{
    for(unsigned i = 0; i < box_count; ++i)
    {
        auto& box = boxes[i];
        
        if(box.GeometryOffset == ignore_id)
            continue;
        
        float tmin, tmax;
        if(!IntersectRayAABB(inout_ray->Data.IncidentLight, inout_ray->Origin, inout_ray->Near, inout_ray->Far, box.AABB.MinCorner, box.AABB.MaxCorner, &tmin, &tmax) ||
           inout_ray->Far < tmin)
            continue;
        
        GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(geom_data + box.GeometryOffset);
        if(geom.GeometryType == (unsigned)RTGeometryType::Rect)
            return false;

        if(OCCLUSION_TESTS[geom.GeometryType](geom, inout_ray))
        {
           inout_ray->GeometryID = box.GeometryOffset;
           return true;
        }
    }
    return false;
}

struct GenericGeometrySampler
{
    inline EXPORT_CUDA static bool intersectTestFunction(const GeometryDescriptionHeader& geom, RTRayCuda* ray)
    {
        return INTERSECT_TESTS[geom.GeometryType](geom, ray);
    }

    inline EXPORT_CUDA static void geometrySampleFunction(const GeometryDescriptionHeader& geom, RTRayCuda* ray)
    {
        return GEOMETRY_SAMPLE[(unsigned)geom.GeometryType](geom, ray);
    }
};


// TODO: Make it into proper hierarchy, i.e. SBVH or LBVH
template<class TSampler>
SHARED_CODE bool IntersectBVH(RTNodeCuda* boxes, unsigned box_count, uint8_t* geom_data, RTRayCuda* inout_ray)
{
    bool status = false;
    for(unsigned i = 0; i < box_count; ++i)
    {
        auto& box = boxes[i];
        float tmin, tmax;
        if(!IntersectRayAABB(inout_ray->Data.IncidentLight, inout_ray->Origin, inout_ray->Near, inout_ray->Far, box.AABB.MinCorner, box.AABB.MaxCorner, &tmin, &tmax) ||
           inout_ray->Far < tmin)
            continue;
        
        GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(geom_data + box.GeometryOffset);
        float tprev = inout_ray->Far;
        if(TSampler::intersectTestFunction(geom, inout_ray) && inout_ray->Far < tprev)
        {
           inout_ray->GeometryID = box.GeometryOffset;
           status = true;
        }
        else
        {
            inout_ray->Far = tprev;
        }
    }
    return status;
}

template<class TSampler>
inline EXPORT_DEVICE void RayTraceImpl(RayTraceData rt_data, RTRayCuda& ray, unsigned& seed)
{
    if(IntersectBVH<TSampler>(rt_data.Boxes, rt_data.BoxCount, rt_data.Geometry, &ray))
    {
        const GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(rt_data.Geometry + ray.GeometryID);
		TSampler::geometrySampleFunction(geom, &ray);

        auto material = ray.Data.Material;
        if(material)
        {
            auto cache_function = MaterialCacheLookup[(size_t)material->Model];
			if(cache_function)
				cache_function(ray.Data, seed);

            // Move to new position and then sample new direction
            ray.Origin += ray.Data.IncidentLight*ray.Far;
            ray.Near = 0.0f;
            ray.Far = INFINITY;
            ray.Data.OutgoingLight = -ray.Data.IncidentLight;

            if(SampleIncidentLightLookup[(size_t)material->Model] == nullptr)
            {
                auto transmittance_func = TransmittanceLookup[(size_t)material->Model];
                if(transmittance_func)
                {
                    ray.Radiance += ray.Throughput*transmittance_func(ray.Data);
                }
                else
                {
				    ray.Radiance += ray.Throughput*static_cast<const RTMicrofacetMaterial*>(material)->Diffuse;
                }
                ray.State = RayIdle;
            }
        }
        else
        {
            ray.State = RayIdle;
        }
    }
    else
    {
        ray.State = RayIdle;
        ray.Radiance += ray.Throughput*rt_data.Background;
    }
}

template<class TSampler = GenericGeometrySampler>
struct DebugRenderTangents
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

        if(IntersectBVH<TSampler>(Data.Boxes, Data.BoxCount, Data.Geometry, &ray))
        {
            const GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(Data.Geometry + ray.GeometryID);
		    TSampler::geometrySampleFunction(geom, &ray);

            ray.Radiance += ray.Throughput*RGBToSpectrum(ray.Data.Tangent*0.5f + 0.5f);
        }
    }
};

template<class TSampler = GenericGeometrySampler>
struct DebugRenderNormals
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

        if(IntersectBVH<TSampler>(Data.Boxes, Data.BoxCount, Data.Geometry, &ray))
        {
            const GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(Data.Geometry + ray.GeometryID);
		    TSampler::geometrySampleFunction(geom, &ray);

            ray.Radiance += ray.Throughput*RGBToSpectrum(ray.Data.Normal*0.5f + 0.5f);
        }
    }
};

template<class TSampler = GenericGeometrySampler>
struct DebugRenderBinormals
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

        if(IntersectBVH<TSampler>(Data.Boxes, Data.BoxCount, Data.Geometry, &ray))
        {
            const GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(Data.Geometry + ray.GeometryID);
		    TSampler::geometrySampleFunction(geom, &ray);

            ray.Radiance += ray.Throughput*RGBToSpectrum(ray.Data.Binormal*0.5f + 0.5f);
        }
    }
};

template<class TSampler = GenericGeometrySampler>
struct DebugLighting
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

        if(IntersectBVH<TSampler>(Data.Boxes, Data.BoxCount, Data.Geometry, &ray))
        {
            const GeometryDescriptionHeader& geom = *reinterpret_cast<GeometryDescriptionHeader*>(Data.Geometry + ray.GeometryID);
		    TSampler::geometrySampleFunction(geom, &ray);

            ray.Radiance += ray.Throughput*ToSpectrum(Maxf(0.0f, Dot(ray.Data.Normal, -ray.Data.IncidentLight)));
        }
    }
};

template<class TSampler = GenericGeometrySampler>
struct RayTraceFirst
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

	    unsigned seed = Hash32(~(ray_idx ^ (ray.GeometryID << 16)));

        RayTraceImpl<TSampler>(Data, ray, seed);
    }
};

template<class TSampler = GenericGeometrySampler>
struct RayTrace
{
    RayTraceData Data;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
        {
            return;
        }

        auto& ray = Data.Rays[ray_idx];

        if(ray.State == RayIdle)
        {
            return;
        }

        unsigned seed = Hash32((ray.GeometryID << 16) ^ ray_idx);

        // TODO: Properly stratify
        Stratification strata;
        strata.XStrata = strata.YStrata = 0;
        strata.TotalXStrata = strata.TotalYStrata = 1;
        SampleIncidentLightLookup[(size_t)ray.Data.Material->Model](strata, &ray.Data, seed);

        ray.Throughput *= TransmittanceLookup[(size_t)ray.Data.Material->Model](ray.Data) / ray.Data.PDF;

        RayTraceImpl<TSampler>(Data, ray, seed);
    }
};

struct SampleLightSources
{
    RayTraceData Data;
    unsigned     MaxLightSamples;
    float        MaxSurfaceSamples;
    bool         ApplyBalance;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned ray_idx)
    {
        if(ray_idx >= Data.RayCount)
            return;

        auto& ray = Data.Rays[ray_idx];

        if(ray.State == RayIdle)
            return;

        unsigned seed = Hash32((ray_idx << 16) ^ ray.GeometryID);

        // TODO: Is random shuffle better?
	    Spectrum light_source_radiance{};

        RTRayCuda light_ray = ray;

        Stratification strata;
        strata.XStrata = strata.YStrata = 0;
        strata.TotalXStrata = strata.TotalYStrata = 1;

	    for(unsigned k = 0; k < MaxLightSamples; ++k)
	    {
		    unsigned light_idx = FastUintRand(0, Data.LightSourceCount, seed);

		    auto* light_src = reinterpret_cast<const LightSource*>(Data.Geometry + Data.LightSources[light_idx]);

            auto light_smp = SampleLightSourceFunctionLookup[(size_t)light_src->Type]({ Data.Geometry }, *light_src, ray.Origin, strata, seed);
            light_ray.Near = 0.0f;            
            light_ray.Far = light_smp.Distance;
            light_ray.GeometryID = INVALID_GEOMETRY;
            light_ray.Throughput = light_smp.Radiance * ray.Throughput / light_smp.PDF;
            light_ray.Data = ray.Data;
		    light_ray.Data.IncidentLight = light_smp.IncidentLight;

		    bool intersect = OcclusionBVH(Data.Boxes, Data.BoxCount, Data.Geometry, light_smp.GeometryID, &light_ray);
            // Restore original data - we don't want to sample the light source
            light_ray.Data.Normal = ray.Data.Normal;
            light_ray.Data.TexCoord = ray.Data.TexCoord;

		    if(!intersect)
		    {
                if(ApplyBalance)
                {
				    float balance = 0.5f;
			    #ifdef ENABLE_MIS
                    // TODO: Multiple light sources?
				    float light_pdf = 0.0f;
				    for(uint32_t light_idx = 0; light_idx < Data.LightSourceCount; ++light_idx)
				    {
                        light_pdf += BalanceBias(LightSourceComputePDFLookup[(unsigned)light_src->Type]({ Data.Geometry }, *light_src, ray.Origin, light_ray.Data.IncidentLight));
				    }
				    light_pdf *= BalanceBias(MaxLightSamples/Data.LightSourceCount);
                    float total_probability = light_pdf + BalanceBias(PDFLookup[(size_t)ray.Data.Material->Model](light_ray.Data) * MaxSurfaceSamples);
                    balance = light_pdf / total_probability;
                #endif
				    light_ray.Throughput *= balance;
                }
			    light_source_radiance += light_ray.Throughput * TransmittanceLookup[(size_t)light_ray.Data.Material->Model](light_ray.Data);
		    }
	    }

	    ray.Radiance += light_source_radiance/(float)MaxLightSamples;
    }
};

struct AcummulateAndConvertRadiance
{
    void*               Backbuffer;
    RTRayCuda*          Rays;
    dim3                ScreenDims;
    unsigned int        SamplesCamera;

    EXPORT_DEVICE void operator()(unsigned worker_id, unsigned x, unsigned y)
    {
        if(x >= ScreenDims.x ||
           y >= ScreenDims.y)
        {
            return;
        }

        Spectrum total_radiance{};
        for(unsigned idx = (y*ScreenDims.x + x)*SamplesCamera, end_idx = idx + SamplesCamera;
            idx < end_idx; ++idx)
        {
            total_radiance += Rays[idx].Radiance;
        }

        unsigned color = ToColor(XYZToSRGB(SpectrumToXYZ(total_radiance)));

        Tempest::Surface2DWrite(color, Backbuffer, x * 4, y);
    }
};

const void* RayTracerCuda::bindTexture(const Texture* tex)
{
#ifdef CPU_DEBUG
    return tex;
#else
    auto& hdr = tex->getHeader();
    auto fmt = hdr.Format;

    auto tex_obj = CreateCudaTexture(tex, fmt == DataFormat::RGBA8UNorm ? TEMPEST_CUDA_TEXTURE_SRGB : 0);

    m_OccupiedTextureMemorySize += hdr.Width*hdr.Height*Tempest::DataFormatElementSize(fmt);

    m_InternalTextures.push_back(tex_obj);

    return reinterpret_cast<void*>(tex_obj);
#endif
}

void RayTracerCuda::bindSurfaceAndTexture(Texture* tex, const void** out_tex_obj, void** out_surf)
{
#ifdef CPU_DEBUG
    *out_tex_obj = tex;
    *out_surf = tex;
#else
    *out_tex_obj = nullptr;
    *out_surf = nullptr;

    cudaArray_t array;

    auto& hdr = tex->getHeader();
    auto chan_desc = DataFormatToCuda(hdr.Format);
    auto err = cudaMallocArray(&array, &chan_desc, hdr.Width, hdr.Height);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create surface: failed to allocate memory: ", cudaGetErrorString(err));
		return;
    }

    size_t pitch = hdr.Width*DataFormatElementSize(hdr.Format);
    err = cudaMemcpy2DToArray(array, 0, 0, tex->getData(), pitch, pitch, hdr.Height, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to copy surface data: ", cudaGetErrorString(err));
		cudaFreeArray(array);
        return;
    }

    m_OccupiedTextureMemorySize += hdr.Height*pitch;

    cudaTextureObject_t tex_obj;
	cudaResourceDesc res_desc;
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = TranslateFilterMode(hdr.Sampling);
    tex_desc.maxAnisotropy = 1;
    tex_desc.normalizedCoords = true;
    tex_desc.sRGB = true;
    tex_desc.readMode = TranslateReadMode(hdr.Format);

    err = cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
	if(err != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to create surface: failed to create texture object: ", cudaGetErrorString(err));
		cudaFreeArray(array);
		return;
	}

    m_InternalTextures.push_back(tex_obj);

	cudaSurfaceObject_t surf_obj;
    err = cudaCreateSurfaceObject(&surf_obj, &res_desc);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create surface object: ", cudaGetErrorString(err));
		cudaDestroyTextureObject(tex_obj);
		cudaFreeArray(array);
		return;
    }

	m_InternalSurfaces.push_back(surf_obj);

    *out_tex_obj = reinterpret_cast<void*>(tex_obj);
	*out_surf = reinterpret_cast<void*>(surf_obj);
#endif
}

void* RayTracerCuda::bindBuffer(void* buf, size_t size)
{
#ifdef CPU_DEBUG
    return buf;
#else
	void* mem;
	auto err = cudaMalloc(&mem, size);
	if(err != cudaSuccess)
	{
		Log(LogLevel::Error, "Failed to allocate buffer: ", cudaGetErrorString(err));
		return nullptr;
	}

    err = cudaMemcpy(mem, buf, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to copy buffer: ", cudaGetErrorString(err));
    }

    m_OccupiedTextureMemorySize += size;

	m_InternalBuffers.push_back(mem);
	return mem;
#endif
}


void RayTracerCuda::addHierarchicalVolume(VolumeRoot* hi_volume, RTMaterial* material)
{
#ifndef CPU_DEBUG
	size_t volume_count = hi_volume->Dimensions.X*hi_volume->Dimensions.Y*hi_volume->Dimensions.Z;

	size_t non_empty_vol = 0;
	Box dims{ 0, 0, 0 };
	for(auto vol_idx = 0; vol_idx < volume_count; ++vol_idx)
	{
		auto& volume = hi_volume->Volumes[vol_idx];
		if(volume.Data == nullptr)
			continue;

		TGE_ASSERT((dims.X | dims.Y | dims.Z) == 0 ||
				   (volume.Dimensions.X == dims.X &&
				    volume.Dimensions.Y == dims.Y &&
					volume.Dimensions.Z == dims.Z), "Invalid volume. Must have equal size cells");
		dims = volume.Dimensions;
		++non_empty_vol;
	}

	std::unique_ptr<RTVolumeCuda> rt_volume(new RTVolumeCuda);

    size_t x_size = size_t(sqrtf((float)non_empty_vol));
    size_t y_size = (non_empty_vol + x_size - 1) / x_size;

    TGE_ASSERT(non_empty_vol <= x_size*y_size, "Invalid 3D texture subdivision");

	auto brick_pool_dev = CREATE_SCOPED(cudaArray*, [](cudaArray* arr){ cudaFreeArray(arr); });
	cudaExtent brick_ext{ dims.X * x_size, dims.Y * y_size, dims.Z };
    auto fmt = CUDAChannelFormats + (size_t)DataFormat::uR8;
	auto err = cudaMalloc3DArray(&brick_pool_dev, fmt, brick_ext, 0);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to allocate memory for brick set: ", cudaGetErrorString(err));
        return;
    }
	
	cudaResourceDesc brick_pool_desc;
	brick_pool_desc.resType = cudaResourceTypeArray;
	brick_pool_desc.res.array.array = brick_pool_dev;

	cudaTextureDesc brick_pool_tex_desc = {};
	brick_pool_tex_desc.addressMode[0] = cudaAddressModeClamp;
	brick_pool_tex_desc.addressMode[1] = cudaAddressModeClamp;
	brick_pool_tex_desc.addressMode[2] = cudaAddressModeClamp;
	brick_pool_tex_desc.filterMode = cudaFilterModePoint; // Non-float
	brick_pool_tex_desc.readMode = cudaReadModeElementType;

	err = cudaCreateTextureObject(&rt_volume->BrickPool, &brick_pool_desc, &brick_pool_tex_desc, nullptr);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create texture object for brick set: ", cudaGetErrorString(err));
        return;
    }

    auto root_brick_dev = CREATE_SCOPED(cudaArray*, [](cudaArray* arr){ cudaFreeArray(arr); });
	cudaExtent root_ext = { hi_volume->Dimensions.X, hi_volume->Dimensions.Y, hi_volume->Dimensions.Z };
	err = cudaMalloc3DArray(&root_brick_dev, CUDAChannelFormats + (size_t)DataFormat::uR32, root_ext, 0);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to allocate memory for root brick: ", cudaGetErrorString(err));
        return;
    }

    cudaResourceDesc root_brick_desc;
    root_brick_desc.resType = cudaResourceTypeArray;
    root_brick_desc.res.array.array = root_brick_dev;

    cudaTextureDesc root_brick_tex_desc = {};
    root_brick_tex_desc.addressMode[0] = cudaAddressModeClamp;
    root_brick_tex_desc.addressMode[1] = cudaAddressModeClamp;
    root_brick_tex_desc.addressMode[2] = cudaAddressModeClamp;
    root_brick_tex_desc.filterMode = cudaFilterModePoint;
    root_brick_tex_desc.readMode = cudaReadModeElementType;

    err = cudaCreateTextureObject(&rt_volume->RootBrick, &root_brick_desc, &root_brick_tex_desc, nullptr);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to create texture object for root brick: ", cudaGetErrorString(err));
        return;
    }
    
    cudaMemcpy3DParms memcpy_params = {};

    auto root_brick_host = CREATE_SCOPED(uint32_t*, [](uint32_t* ptr){ cudaFreeHost(ptr); });
    err = cudaHostAlloc(reinterpret_cast<void**>(&root_brick_host), volume_count*sizeof(uint32_t), cudaHostAllocWriteCombined);
    
    size_t brick_size = dims.X*dims.Y*dims.Z*sizeof(uint8_t);
    auto brick_pool_host = CREATE_SCOPED(uint8_t*, [](uint8_t* ptr){ cudaFreeHost(ptr); });
    err = cudaHostAlloc(reinterpret_cast<void**>(&brick_pool_host), x_size*y_size*brick_size, cudaHostAllocWriteCombined);

    size_t brick_pool_pitch = dims.X*x_size*sizeof(uint8_t);
	size_t brick_pool_slice = dims.Y*y_size*brick_pool_pitch;

    size_t vol_pitch = dims.X*sizeof(uint8_t);
    size_t vol_slice = dims.Y*vol_pitch;

    size_t node_idx = 0;
    for(size_t vol_idx = 0; vol_idx < volume_count; ++vol_idx)
	{
		auto& volume = hi_volume->Volumes[vol_idx];
		if(volume.Data == nullptr)
        {
            root_brick_host[vol_idx] = ~0;
			continue;
        }

		size_t node_x = node_idx % x_size;
		size_t node_y = node_idx / x_size;

		root_brick_host[vol_idx] = node_x | (node_y << 16);
        
        auto * vol = (uint8_t*)hi_volume->Volumes[vol_idx].Data;
		for(auto* ptr = brick_pool_host + node_y*dims.Y*brick_pool_pitch + node_x*dims.X,
				* end_ptr = ptr + dims.Z*brick_pool_slice; ptr < end_ptr; ptr += brick_pool_slice, vol += vol_slice)
		{
			// Here we go back
			for(size_t row = 0; row < dims.Y; ++row)
			{
                /*
                memset(ptr + brick_pool_pitch*row, 0xFF, dims.X);
                /*/
				memcpy(ptr + row*brick_pool_pitch,
					   vol + row*vol_pitch,
					   dims.X);
                //*/
			}
		}
        ++node_idx;
	}
    
    memcpy_params.kind = cudaMemcpyHostToDevice;
    memcpy_params.extent = root_ext;
    memcpy_params.srcPtr = cudaPitchedPtr{ root_brick_host, root_ext.width*sizeof(uint32_t), root_ext.width, root_ext.height };
    memcpy_params.dstArray = root_brick_dev;

    err = cudaMemcpy3D(&memcpy_params);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to tile texture: ", cudaGetErrorString(err));
        return;
    }

    memcpy_params.extent = brick_ext;
    memcpy_params.srcPtr = cudaPitchedPtr{ brick_pool_host, brick_ext.width*sizeof(uint8_t), brick_ext.width, brick_ext.height };
    memcpy_params.dstArray = brick_pool_dev;

    err = cudaMemcpy3D(&memcpy_params);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "Failed to tile texture: ", cudaGetErrorString(err));
        return;
    }

    rt_volume->Dimensions = hi_volume->Dimensions;
    rt_volume->BrickDimensions = dims;
    rt_volume->BrickPoolArray = brick_pool_dev.release();
	rt_volume->RootBrickArray = root_brick_dev.release();
	m_Volumes.push_back(rt_volume.release());

    m_SceneGeometryTypes |= 1u << (uint32_t)RTGeometryType::Volume;
#endif
}

uint64_t RayTracerCuda::addRect(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size, RTMaterial* material, const AABB2* tc, const void* tangent_texture)
{
    RTNodeCuda node;
    Rect3Geometry geom;
    geom.Rect.Center = pos;
    geom.Rect.Orientation = ToQuaternion(Matrix3(tan, Cross(norm, tan), norm));
    geom.Rect.Size = size;

    if(tc)
    {
        geom.TexCoordStart = tc->MinCorner;
        geom.TexCoordMultiplier = tc->MaxCorner - tc->MinCorner;
    }
    else
    {
        geom.TexCoordStart = { 0.0f, 0.0f };
        geom.TexCoordMultiplier = { 1.0f, 1.0f };
    }
    geom.TangentMap = tangent_texture;

    Rect3Bounds(geom.Rect, &node.AABB);
    size_t mat_size = MaterialSizeLookup[(size_t)material->Model](material);
    mat_size = AlignAddress(mat_size, sizeof(uint64_t));
	size_t mat_off = AlignAddress(sizeof(geom), sizeof(uint64_t));

    auto geom_ptr = m_DataPool.allocateAlignedMemory(mat_off + mat_size, sizeof(uint64_t));
    node.GeometryOffset = static_cast<uint32_t>(geom_ptr.PoolOffset);

    m_Boxes.push_back(node);
    
    auto pool = m_DataPool.getBase();
    memcpy(pool(geom_ptr), &geom, sizeof(geom));
    memcpy(pool(geom_ptr) + mat_off, material, mat_size);

    m_SceneGeometryTypes |= 1u << (uint32_t)RTGeometryType::Rect;

    return node.GeometryOffset;
}

void RayTracerCuda::addTriangleMesh(const Matrix4& world,
									size_t submesh_count, 
									RTSubmesh* submeshes,
									size_t tri_count, int32_t* tris,
									size_t vert_size, void* verts,
                                    MeshOptions* mesh_opts,
									uint64_t* geom_ids)
{
	size_t indices_size = 3*tri_count*sizeof(int32_t);
	size_t geometry_data_size = sizeof(MeshGeometry) + indices_size + vert_size + submesh_count*sizeof(SubmeshGeometry);

	for(uint32_t submesh_idx = 0; submesh_idx < submesh_count; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];

		auto material = submesh.Material;

		auto mat_size = MaterialSizeLookup[(size_t)material->Model](material);
		auto tri_count = submesh.VertexCount/3;

		size_t max_node_count = 2*tri_count - 1;

		geometry_data_size = AlignAddress(geometry_data_size + max_node_count*sizeof(SimpleStacklessLBVH2Node<AABBUnaligned>), 8) + mat_size;
	}

	auto data_ptr = m_DataPool.allocateAlignedMemory(geometry_data_size, sizeof(uint64_t));
	auto data_pool = m_DataPool.getBase();
    RTNodeCuda node;
	auto offset = node.GeometryOffset = data_ptr.PoolOffset;


	auto& header = reinterpret_cast<MeshGeometry&>(*data_pool(data_ptr));
	header.GeometryType = (unsigned)RTGeometryType::Mesh;
    if(mesh_opts && mesh_opts->TwoSided)
        header.GeometryType = (unsigned)RTGeometryType::TwoSidedMesh;
	header.InverseTransform = world.inverse();
	header.SubmeshCount = submesh_count;
	header.VertexSize = vert_size;
	header.IndexCount = 3*tri_count;

	offset += sizeof(MeshGeometry);
    auto ind_offset = offset;
    memcpy(data_pool(PoolPtr<uint8_t>{ offset }), tris, indices_size);
	offset += indices_size;

    auto vert_offset = offset;
	memcpy(data_pool(PoolPtr<uint8_t>{ offset }), verts, vert_size);
	offset += vert_size;

	auto submesh_offset = offset;
	offset += submesh_count*sizeof(SubmeshGeometry);
	auto start_offset = offset;

    auto expected_size = m_DataPool.getDataSize();

	for(uint32_t submesh_idx = 0; submesh_idx < submesh_count; ++submesh_idx)
	{
		auto& submesh = submeshes[submesh_idx];

		if(geom_ids)
		{
			geom_ids[submesh_idx] = ((uint64_t)submesh_idx << 32ULL) | node.GeometryOffset;
		}

		auto tri_count = submesh.VertexCount/3;

		size_t max_node_count = (2*tri_count - 1);
		size_t bvh_size = max_node_count*sizeof(SimpleStacklessLBVH2Node<AABBUnaligned>);

        auto& submesh_header = reinterpret_cast<SubmeshGeometry&>(*data_pool(PoolPtr<uint8_t>{ submesh_offset }));
		submesh_header.VertexCount = submesh.VertexCount;
		submesh_header.VertexOffset = submesh.VertexOffset;
		submesh_header.BaseIndex = submesh.BaseIndex;
		submesh_header.Stride = submesh.Stride;
		submesh_header.BVHOffset = offset - start_offset;
		
		void* submesh_vert_ptr = reinterpret_cast<uint8_t*>(verts) + submesh.VertexOffset;
		auto submesh_ind_ptr = reinterpret_cast<uint32_t*>(tris + submesh.BaseIndex);
		uint32_t submesh_vert_count = (static_cast<uint32_t>(vert_size) - submesh.VertexOffset)/submesh.Stride;
		std::unique_ptr<LBVH2Node<AABBUnaligned>> nodes(GenerateTriangleNodes<AABBUnaligned>(submesh_vert_ptr, submesh_vert_count, submesh_ind_ptr, tri_count, submesh.Stride));
	
		std::unique_ptr<SimpleStacklessLBVH2Node<AABBUnaligned>> bvh(GenerateSSLBVH(nodes.get(), tri_count));

        if(submesh_idx == 0)
        {
            node.AABB.MinCorner = bvh->Bounds.MinCorner;
    		node.AABB.MaxCorner = bvh->Bounds.MaxCorner;
        }
        else
        {
		    node.AABB.MinCorner = Vector3Min(bvh->Bounds.MinCorner, node.AABB.MinCorner);
		    node.AABB.MaxCorner = Vector3Max(bvh->Bounds.MaxCorner, node.AABB.MaxCorner);
        }

		memcpy(data_pool(PoolPtr<uint8_t>{ offset }), bvh.get(), bvh_size);

		offset = AlignAddress(offset + bvh_size, 8);

        auto mat_offset = offset;
		submesh_header.MaterialOffset = mat_offset - start_offset;
		auto material = submesh.Material;
		auto mat_size = MaterialSizeLookup[(size_t)material->Model](material);
		memcpy(data_pool(PoolPtr<uint8_t>{ offset }), submesh.Material, mat_size);

        if(IsEmissiveIlluminationModel(material->Model))
        {
            size_t light_size = AlignAddress(sizeof(MeshLight), 8);

            auto mesh_light_pool = m_DataPool.allocateAligned<MeshLight>(sizeof(uint64_t));
            auto& mesh_light = *data_pool(mesh_light_pool);
            new (&mesh_light) MeshLight;
            mesh_light.IndexBuffer = { ind_offset + sizeof(uint32_t)*submesh.BaseIndex };
            mesh_light.VertexBuffer = { vert_offset + submesh.VertexOffset };
            mesh_light.Stride = submesh.Stride;
            mesh_light.Material = { mat_offset }; 
            mesh_light.TriangleCount = tri_count;
            mesh_light.GeometryID = node.GeometryOffset;
            m_LightSources.push_back(mesh_light_pool.PoolOffset);
        }

		offset += mat_size;

		submesh_offset += sizeof(SubmeshGeometry);

        m_SceneGeometryTypes |= 1u << (uint32_t)header.GeometryType;
	}

	TGE_ASSERT(submesh_offset == start_offset, "Invalid offset after inserting submeshes");

	TGE_ASSERT(offset == expected_size, "Invalid data");

	m_Boxes.push_back(node);
}

void RayTracerCuda::addSphereLightSource(SphereAreaLight* light)
{
    std::unique_ptr<LightSource> cleanup(light);

	RTNodeCuda node;

    RTMicrofacetMaterial emissive_material;
    emissive_material.Model = IlluminationModel::Emissive;
	emissive_material.Diffuse = light->Radiance;

    size_t mat_size = MaterialSizeLookup[(size_t)emissive_material.Model](&emissive_material);
    SphereGeometry geom;
    geom.SphereShape = light->SphereShape;

    SphereBounds(light->SphereShape, &node.AABB);
    uint32_t mat_off = AlignAddress(sizeof(geom), sizeof(uint64_t));
    uint32_t light_off = AlignAddress(mat_off + static_cast<uint32_t>(mat_size), sizeof(uint64_t));
    auto geom_data = m_DataPool.allocateAlignedMemory(light_off + AlignAddress(sizeof(SphereAreaLight), sizeof(uint64_t)), sizeof(uint64_t));
    node.GeometryOffset = geom_data.PoolOffset;

	m_Boxes.push_back(node);

	light->GeometryID = node.GeometryOffset;
    auto data_pool = m_DataPool.getBase();
    memcpy(data_pool(geom_data), &geom, sizeof(geom));
    memcpy(data_pool(geom_data) + mat_off, &emissive_material, mat_size);
	memcpy(data_pool(geom_data) + light_off, light, sizeof(*light));

    m_SceneGeometryTypes |= 1u << (uint32_t)RTGeometryType::Sphere;

	m_LightSources.push_back(node.GeometryOffset + light_off);
}

void RayTracerCuda::addLightSource(LightSource* light)
{
    std::unique_ptr<LightSource> cleanup(light);

    size_t size_of_light = 0;

    switch(light->Type)
    {
    case LightSourceType::Directional:
    {
        size_of_light += sizeof(DirectionalLight);
    } break;
    case LightSourceType::Point:
    {
        size_of_light += sizeof(PointLight);
    } break;
    default:
        TGE_ASSERT(false, "Unsupported light");
        return;
    }
    
    auto light_pool = m_DataPool.allocateAlignedMemory(size_of_light, sizeof(uint64_t));
	memcpy(m_DataPool(light_pool), light, size_of_light);

	m_LightSources.push_back(light_pool.PoolOffset);
}

// Performed on the CPU to avoid readbacks
uint64_t RayTracerCuda::rayQuery(const Vector2& tc, SampleData* sample_data)
{
	Vector4 screen_tc{2.0f*tc.x - 1.0f, 1.0f - 2.0f*tc.y, -1.0f, 1.0};

	Vector4 pos_start = m_ViewProjectionInverse*screen_tc;

	screen_tc.z = 1.0f;
	Vector4 pos_end = m_ViewProjectionInverse*screen_tc;

	Vector3 start_ray_pos = ToVector3(pos_start);
	Vector3 end_ray_pos = ToVector3(pos_end);

	RTRayCuda ray;

    ray.Origin = start_ray_pos;
    ray.Data.IncidentLight = Normalize(end_ray_pos - start_ray_pos);
    ray.State = RayActive;
    ray.Throughput = ToSpectrum(1.0f);
    ray.Radiance = {};
/*    ray.X = 0;
    ray.Y = 0; */
    ray.Near = 0.0f;
    ray.Far = INFINITY;
    ray.GeometryID = INVALID_GEOMETRY;

    IntersectBVH<GenericGeometrySampler>(&m_Boxes.front(), m_Boxes.size(), m_DataPool.getBase().BaseAddress, &ray);

	if(ray.GeometryID == INVALID_GEOMETRY)
	{
		return INVALID_GEOMETRY;
	}	
	memset(sample_data, 0, sizeof(*sample_data));

    const GeometryDescriptionHeader& geometry = *m_DataPool(PoolPtr<GeometryDescriptionHeader>{ ray.GeometryID });
    GEOMETRY_SAMPLE[(unsigned)geometry.GeometryType](geometry, &ray);
	
    *sample_data = ray.Data;
	return ((uint64_t)ray.InstanceID << 32ULL) | ray.GeometryID;
}

void RayTracerCuda::commitScene()
{
#ifndef CPU_DEBUG
    if(m_GPUBoxes)
    {
        cudaFree(m_GPUBoxes);
        m_GPUBoxes = nullptr;
    }

    if(m_GPUGeometry)
    {
        cudaFree(m_GPUGeometry);
        m_GPUGeometry = nullptr;
    }

    if(m_GPULightSources)
    {
        cudaFree(m_GPULightSources);
        m_GPULightSources = nullptr;
    }

    m_OccupiedTotalMemorySize = m_OccupiedTextureMemorySize;

    size_t box_size = sizeof(m_Boxes[0])*m_Boxes.size();
    auto err = cudaMalloc(reinterpret_cast<void**>(&m_GPUBoxes), box_size);
    if(err != cudaSuccess)
    {
        Log(LogLevel::Error, "failed to allocate bounding volume hierarchy: ", cudaGetErrorString(err));
        return;
    }

    m_OccupiedTotalMemorySize += box_size;

    if(!m_Boxes.empty())
    {
        err = cudaMemcpy(m_GPUBoxes, &m_Boxes.front(), m_Boxes.size()*sizeof(m_Boxes[0]), cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to copy to BVH to GPU buffer: ", cudaGetErrorString(err));
            return;
        }


        size_t geometry_size = m_DataPool.getDataSize();
        err = cudaMalloc(reinterpret_cast<void**>(&m_GPUGeometry), geometry_size);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to allocate geometry data: ", cudaGetErrorString(err));
            return;
        }
        m_OccupiedTotalMemorySize += geometry_size;

        err = cudaMemcpy(m_GPUGeometry, m_DataPool.getBase().BaseAddress, geometry_size, cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to copy to geometry to GPU buffer: ", cudaGetErrorString(err));
            return;
        }
    }
    
    if(!m_LightSources.empty())
    {
        size_t light_size = sizeof(m_LightSources[0])*m_LightSources.size();
	    err = cudaMalloc(reinterpret_cast<void**>(&m_GPULightSources), light_size);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to allocate light sources: ", cudaGetErrorString(err));
            return;
        }
        m_OccupiedTotalMemorySize += light_size;

        err = cudaMemcpy(m_GPULightSources, &m_LightSources.front(), m_LightSources.size()*sizeof(m_LightSources[0]), cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to copy to light sources to GPU buffer: ", cudaGetErrorString(err));
            return;
        }
    }

    if(m_OccupiedTotalMemorySize > m_DeviceMemorySize)
    {
        Log(LogLevel::Debug, "running low on device memory - possible residency related performance issues");
    }
#endif
}

void RayTracerCuda::draw(CUDASurfaceResource* backbuffer, const Matrix4& view_proj_inv)
{
    m_ViewProjectionInverse = view_proj_inv;
    m_Backbuffer = backbuffer;
    drawOnce();
}

void RayTracerCuda::setGlobalCubeMap(const Tempest::CubeMap* cube_map)
{
    TGE_ASSERT(false, "Stub");
}

void RayTracerCuda::drawOnce()
{
    dim3 tex_dims(m_Backbuffer->Texture.Description.Width, m_Backbuffer->Texture.Description.Height, 0);

    RayTraceData rt_data;

#ifdef CPU_DEBUG
    auto& debug_hdr = m_DebugBackbuffer->getHeader();
    if(tex_dims.x != debug_hdr.Width ||
       tex_dims.y != debug_hdr.Height)
    {
        auto new_hdr = debug_hdr;
        new_hdr.Width = tex_dims.x;
        new_hdr.Height = tex_dims.y;
        m_DebugBackbuffer->realloc(new_hdr);
    }

    rt_data.Boxes = &m_Boxes.front();
    rt_data.Geometry = &m_GeometryData.front();
    rt_data.LightSources = !m_LightSources.empty() ? &m_LightSources.front() : nullptr;
#else
    rt_data.Boxes = m_GPUBoxes;
    rt_data.Geometry = m_GPUGeometry;
    rt_data.LightSources = m_GPULightSources;
#endif

    auto area = tex_dims.x*tex_dims.y;

    unsigned pass_memory = (m_DeviceMemorySize - m_OccupiedTotalMemorySize)*2/3; // safe-guard to make sure that we are not consuming too much memory

    unsigned pass_samples = pass_memory/(area*sizeof(m_GPURays[0]));

    if(pass_samples == 0)
    {
        Tempest::Log(Tempest::LogLevel::Fatal, "Resolution is too high. Disabling ray tracing! Block ray tracing must be implemented");
        return;
    }
    
    pass_samples = Minf(pass_samples, m_SamplesCamera);

    rt_data.Background = m_Background;
    rt_data.RayCount = area*pass_samples;
    rt_data.BoxCount = static_cast<unsigned>(m_Boxes.size());

    rt_data.LightSourceCount = static_cast<unsigned>(m_LightSources.size());

    // TODO: Reallocation
    auto ray_buf_size = sizeof(m_GPURays[0])*rt_data.RayCount;
    if(m_PrevRayCount != ray_buf_size && m_GPURays)
    {
        GENERIC_FREE(m_GPURays);
        m_GPURays = nullptr;
    }

    if(!m_GPURays)
    {
        TGE_ASSERT(m_OccupiedTotalMemorySize + ray_buf_size <= m_DeviceMemorySize, "not enough memory");
         
        auto err = GENERIC_MALLOC(reinterpret_cast<void**>(&m_GPURays), ray_buf_size);
        if(err != cudaSuccess)
        {
            Log(LogLevel::Error, "failed to allocate GPU rays buffer: ", cudaGetErrorString(err));
            return;
        }
        m_PrevRayCount = ray_buf_size;
    }

    GENERIC_MEMSET(m_GPURays, 0, ray_buf_size);

    rt_data.Rays = m_GPURays;

    // TODO: Less than ideal - put everything in accelerated structure
    /*
    for(auto* rt_vol : m_Volumes)
    {
        auto root_brick = rt_vol->RootBrick;
        auto brick_pool = rt_vol->BrickPool;
        float3 vol_dims = float3{ (float)rt_vol->Dimensions.X, (float)rt_vol->Dimensions.Y, (float)rt_vol->Dimensions.Z };
        float3 brick_size = float3{ (float)rt_vol->BrickDimensions.X, (float)rt_vol->BrickDimensions.Y, (float)rt_vol->BrickDimensions.Z };
        VolumeDebug<<<thread_groups, group_size>>>(root_brick, brick_pool, vol_dims, brick_size, tex_dims, fill_color);
    }
    */

    GenerateCameraRaysInfo info;
    info.SamplesCamera = pass_samples;
    info.ViewProjectionInverse = m_ViewProjectionInverse;
    info.ScreenDims = tex_dims;
    info.Rays = m_GPURays;
    info.RayCount = rt_data.RayCount;

#if !defined(NDEBUG) && !defined(CPU_DEBUG)
    {
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TGE_ASSERT(err == cudaSuccess, "Something was broken before we even started");
    }
#endif

    GenerateCameraRays generate_camera_ray{ info };
    Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, generate_camera_ray);
    
    auto max_light_samples = m_SamplesLocalAreaLight;

    switch(m_RenderMode)
    {
    case RenderMode::Normal:
    {
        if((m_SceneGeometryTypes & ~(1u << (uint32_t)RTGeometryType::Mesh)) == 0)
        {
            RayTraceFirst<DirectGeometryMesh> ray_trace_first{ rt_data };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, ray_trace_first);
        }
        else
        {
            RayTraceFirst<> ray_trace_first{ rt_data };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, ray_trace_first);
        }

        if(!m_LightSources.empty())
        {
            SampleLightSources sample_light_first_bounce{ rt_data, max_light_samples, 1.0f, m_MaxRayDepth != 0 };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, sample_light_first_bounce);
        }

        for(uint32_t ray_depth = 0; ray_depth < m_MaxRayDepth; ++ray_depth)
        {
            RayTrace<> ray_trace{ rt_data };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, ray_trace);

            if(!m_LightSources.empty())
            {
                SampleLightSources sample_light{ rt_data, max_light_samples, 1.0f, ray_depth + 1 != m_MaxRayDepth };
                Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, sample_light);
            }   
            max_light_samples = 1;
        }
    } break;
    case RenderMode::DebugTangents:
    {
        DebugRenderTangents<> debug_render_tangents{ rt_data };
        Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, debug_render_tangents);
    } break;
	case RenderMode::DebugBinormals:
	{
        DebugRenderBinormals<> debug_render_binormals{ rt_data };
        Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, debug_render_binormals);
	} break;
    case RenderMode::DebugNormals:
    {
        DebugRenderNormals<> debug_render_normals{ rt_data };
        Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, debug_render_normals);
    } break;
	case RenderMode::DebugLighting:
	{
        if((m_SceneGeometryTypes & ~(1u << (uint32_t)RTGeometryType::Mesh)) == 0)
        {
            DebugLighting<DirectGeometryMesh> debug_lighting{ rt_data };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, debug_lighting);
        }
        else
        {
            DebugLighting<> debug_lighting{ rt_data };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP(m_ThreadId, m_Pool, rt_data.RayCount, debug_lighting);
        }
	} break;
    }

#ifdef CPU_DEBUG
    auto backbuffer = m_DebugBackbuffer.get();
#else
    auto backbuffer = reinterpret_cast<void*>(m_Backbuffer->Surface);
#endif

    AcummulateAndConvertRadiance accumulate{ backbuffer, rt_data.Rays, tex_dims, pass_samples };
    Tempest::EXECUTE_PARALLEL_FOR_LOOP_2D(m_ThreadId, m_Pool, tex_dims.x, tex_dims.y, accumulate);

#ifndef CPU_DEBUG
#   ifndef NDEBUG
    {
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TGE_ASSERT(err == cudaSuccess, "Invalid kernel launch");
    }
#   endif

#else
    auto pitch = tex_dims.x*sizeof(uint32_t);
    auto status = cudaMemcpy2DToArray(m_Backbuffer->Array, 0, 0, m_DebugBackbuffer->getData(), pitch, pitch, tex_dims.y, cudaMemcpyHostToDevice);
    TGE_ASSERT(status == cudaSuccess, "Failed to copy backbuffer");

    cudaThreadSynchronize();
#endif
    // Stupid trick to force synchronization on the device
    //cudaMemcpyFromArray(m_GPURays, m_Backbuffer->Array, 0, 0, 1, cudaMemcpyDeviceToDevice);
}
}
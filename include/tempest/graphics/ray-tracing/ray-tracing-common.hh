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

#ifndef _TEMPEST_RAY_TRACING_COMMON_HH_
#define _TEMPEST_RAY_TRACING_COMMON_HH_

#include "tempest/math/matrix2.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/math/shapes.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/mesh/lbvh2.hh"
#include "tempest/mesh/sslbvh2.hh"
#include "tempest/mesh/essbvh2.hh"
#include "tempest/utils/memory.hh"

#ifndef __CUDACC__
#   include "tempest/graphics/texture.hh"
#endif

#include <memory>

namespace Tempest
{
enum
{
    UNINITIALIZED_STATE = 1 << 0,
	DIRTY_RECT = 1 << 1
};

#define INVALID_GEOMETRY ((unsigned)~0u)

const size_t ScratchMemoryPerThread = 65536;

class Texture;

enum class PicturePostProcess
{
    Linear,
    SRGB,
    AutoExposureSRGB,
    AutoExposureHDR,
    ACES
};

enum class IlluminationModel: uint32_t
{
    Emissive,
    BlinnPhong,
    KajiyaKay,
    AshikhminShirley,
    MicroFlake,
    _Free1,
    _Free2,
    _Free3,
    Patchwork,
    Mirror,
    _Free4,
    GGXMicrofacet,
    GGXMicrofacetConductor,
    GGXMicrofacetDielectric,
    Mix,
    GGXMicrofacetAnisotropic,
    GGXMicrofacetConductorAnisotropic,
    GGXMicrofacetDielectricAnisotropic,
	_Free5,
    _Free6,
    StochasticRotator,
    _Free7,
    SGGXMicroFlake,
    SpatiallyVaryingEmissive,
    SGGXSurface,
    SGGXPseudoVolume,
    _Free8,
    BTF,
    BeckmannMicrofacet,
    BeckmannMicrofacetConductor,
    BeckmannMicrofacetDielectric,
    BeckmannMicrofacetAnisotropic,
    BeckmannMicrofacetConductorAnisotropic,
    BeckmannMicrofacetDielectricAnisotropic,
    Count
};

enum class VolumeIlluminationModel: uint32_t
{
    MicroFlake,
    SGGXMicroFlake,
    SGGXMicroFlakeSurface,
    Count
};

enum YarnDirection
{
    YARN_TANGENT,
    YARN_BINORMAL
};

inline float ComputeReflectionCoefficient(float refr_index_0, float refr_index_1 = 1.0f)
{
    float ratio = (refr_index_0 - refr_index_1) / (refr_index_0 + refr_index_1);
    return ratio*ratio;
}

struct RTObjectSettings
{
    bool                    MultiBounceTransparent;
};

enum class RenderMode
{
    Normal,
    DebugTangents,
	DebugBinormals,
    DebugNormals,
	DebugLighting
};

struct RTSettings
{
    uint32_t                 DataPoolSize = 32*1024*1024;
};

// TODO: This doesn't make sense for many material types
struct RTMaterial
{
    IlluminationModel        Model = IlluminationModel::BlinnPhong;

    void setup();
};

struct RTMirrorMaterial: public RTMaterial
{
    Spectrum                 Specular;
};

struct RTMicrofacetMaterial: public RTMaterial
{
    Spectrum                 Specular;
    union
    {
        Vector2                  StandardDeviation;
        Vector2                  SpecularPower;
    };
    Spectrum                 Diffuse;
    Vector2   	    		 Fresnel = Vector2{ 1.0f, 0.0f };
    float                    Normalization;
    
    const void*              DiffuseMap = nullptr;
    const void*              SpecularMap = nullptr;

};

struct Stratification
{
    float             XStrata;
    float             YStrata;
    float             TotalXStrata;
    float             TotalYStrata;
};

struct SampleData
{
    const RTMaterial* Material;
    Vector2           TexCoord;
    Vector3           OutgoingLight;
    Vector3           IncidentLight;
    Vector3           Tangent;
	Vector3			  Binormal;
    Vector3           Normal;
    Vector3           Position;
    float             TotalDensity;
    mutable float     DirectionalDensity; // TODO: Figure out why the cache doesn't work
    float             PDF;
	mutable uint8_t*  ScratchMemory; // Have fun - ScratchMemoryPerThread of memory to go nuts with materials that do complicated stuff
};

template <class T>
struct MaterialSubData
{
    size_t Offset;
    MaterialSubData(size_t* offset)
        :   Offset(*offset) { *offset += sizeof(T); }
};

template <class T>
struct MaterialVaryingSubData
{
    size_t Offset;
    MaterialVaryingSubData(size_t elem_size, size_t* offset)
        :   Offset(*offset) { *offset += elem_size; }
};

template <class T, size_t size>
struct MaterialArraySubData
{
    size_t Offset;
    MaterialArraySubData(size_t* offset)
        :   Offset(*offset) { *offset += size*sizeof(T); }
};

template <class T, size_t size>
struct MaterialArrayVaryingSubData
{
    size_t Offset[size];
    MaterialArrayVaryingSubData(size_t* elem_size, size_t* offset)
    {
        for(size_t i = 0; i < size; ++i)
        {
            Offset[i] = *offset; 
			*offset += elem_size[i];
        }
    }

    inline EXPORT_CUDA size_t getOffset(size_t idx) const { return Offset[idx]; }
};

struct RTSubmesh
{
    RTMaterial*              Material;
    uint32_t                 VertexCount;
    uint32_t                 VertexOffset;
    uint32_t                 BaseIndex;
	uint32_t				 Stride;
};

typedef RTSubmesh RTSubhair;

// Wastes a little bit of space, but you can pretty much reorder initialization without breaking anything
#define MATERIAL_SUBDATA(type, name) \
    MaterialSubData<type> __MaterialSubdata##name; \
    inline EXPORT_CUDA type& get##name() { return *reinterpret_cast<type*>(reinterpret_cast<uint8_t*>(this) + __MaterialSubdata##name.Offset); } \
    inline EXPORT_CUDA const type& get##name() const { return *reinterpret_cast<const type*>(reinterpret_cast<const uint8_t*>(this) + __MaterialSubdata##name.Offset); }
#define MATERIAL_VARYING_SUBDATA(type, name) \
    MaterialVaryingSubData<type> __MaterialSubdata##name; \
    inline EXPORT_CUDA type& get##name() { return *reinterpret_cast<type*>(reinterpret_cast<uint8_t*>(this) + __MaterialSubdata##name.Offset); } \
    inline EXPORT_CUDA const type& get##name() const { return *reinterpret_cast<const type*>(reinterpret_cast<const uint8_t*>(this) + __MaterialSubdata##name.Offset); }
#define MATERIAL_ARRAY_VARYING_SUBDATA(type, name, size) \
    MaterialArrayVaryingSubData<type, size> __MaterialSubdata##name; \
    inline EXPORT_CUDA type& get##name(size_t idx) { return *reinterpret_cast<type*>(reinterpret_cast<uint8_t*>(this) + __MaterialSubdata##name.getOffset(idx)); } \
    inline EXPORT_CUDA const type& get##name(size_t idx) const { return *reinterpret_cast<const type*>(reinterpret_cast<const uint8_t*>(this) + __MaterialSubdata##name.getOffset(idx)); }
#define MATERIAL_SUBDATA_NAME(name) __MaterialSubdata##name

struct RTMixMaterial: public RTMaterial
{
    const void*              MixTexture;

    MATERIAL_ARRAY_VARYING_SUBDATA(RTMicrofacetMaterial, SubMaterial, 2)

    inline static RTMixMaterial* create(size_t* mat_size)
    {
		size_t data_size = sizeof(RTMixMaterial) + mat_size[0] + mat_size[1];
		auto mem_ptr = malloc(data_size);
		memset(mem_ptr, 0, data_size);
		RTMixMaterial* ptr = new (mem_ptr) RTMixMaterial(mat_size);
		return ptr;
    }

protected:
    RTMixMaterial(size_t* mat_size, size_t offset = sizeof(RTMixMaterial))
        :   MATERIAL_SUBDATA_NAME(SubMaterial)(mat_size, &offset) {}
};

struct RTStochasticRotatorMaterial: public RTMaterial
{
    Vector2                    StandardDeviation;

    uint32_t                   SampleCount;
    MATERIAL_VARYING_SUBDATA(RTMaterial, SubMaterial)

    inline static RTStochasticRotatorMaterial* create(size_t mat_size)
    {
        size_t data_size = sizeof(RTStochasticRotatorMaterial) + mat_size;
        auto mem_ptr = malloc(data_size);
        memset(mem_ptr, 0, data_size);
        auto* ptr = new (mem_ptr) RTStochasticRotatorMaterial(mat_size);
        return ptr;
    }

protected:
    RTStochasticRotatorMaterial(size_t mat_size, size_t offset = sizeof(RTStochasticRotatorMaterial))
        :   MATERIAL_SUBDATA_NAME(SubMaterial)(mat_size, &offset)
    {
        Model = IlluminationModel::StochasticRotator;
    }
};

typedef float (*PDFFunction)(const SampleData&);
typedef Spectrum (*TransmittanceFunction)(const SampleData&);
typedef void (*SampleIncidentLightFunction)(const Stratification& strata, SampleData*, unsigned& seed);
typedef void (*MaterialSetupFunction)(RTMaterial* material);
typedef bool (*MaterialIsMirrorFunction)(const SampleData&);
typedef void (*MaterialCacheFunction)(const SampleData&, unsigned& seed);
typedef size_t (*MaterialSizeFunction)(const RTMaterial* material); // Collapsed size, so that it can be conveniently placed in linear memory
typedef float (*VolumeDensityFunction)(const SampleData&);

struct RTSpatiallyVaryingEmitter: public RTMaterial
{
    TransmittanceFunction EmitFunction;
    uint32_t              MaterialSize = sizeof(RTSpatiallyVaryingEmitter);
};

template<class T>
struct SimpleDeleter
{
    inline void operator()(T* ptr) { free(ptr); }
};

template<class T>
using UniqueMaterial = std::unique_ptr<T, SimpleDeleter<T>>;

// TODO: Hard associate materials with structs
inline UniqueMaterial<RTMixMaterial> CreateRTMixMaterial(size_t mat_size0, size_t mat_size1)
{
    size_t mat_size[2] =
    {
        mat_size0,
        mat_size1
    };

    return UniqueMaterial<RTMixMaterial>(RTMixMaterial::create(mat_size));
}

struct Patch
{
    Matrix2                  RotateScale;
    Matrix2                  RotateScaleInverse;
    Matrix2                  Rotate;
    Vector2                  Translate;
    union
    {
        uint8_t              MiniScratchMemory[8]; // Scratch memory overload
        uint64_t             MiniScratchMemoryUInt64;
    };
    /*
    Vector2                  Repeat;
    */
};

struct RTPatchworkMaterial: public RTMicrofacetMaterial
{
    IlluminationModel        BaseModel;
    RTMaterial*              PatchMaterial;
    uint32_t                 PatchCount;
    Patch*                   Patches = nullptr;
    LBVH2Node<AABB2>*        BVH = nullptr;

    ~RTPatchworkMaterial()
    {
        delete[] Patches;
        delete[] BVH;
    }
};

const size_t MicroFlakeMaxDirectionSamples = 100;

struct RTVolumeMaterial: public RTMaterial
{
    VolumeIlluminationModel VolumeModel;
    Vector3                 Albedo;
};

struct RTMicroFlakeMaterial: public RTVolumeMaterial
{
    float                   StandardDeviation;
    float                   Normalization;
    float                   DirectionalDensity[MicroFlakeMaxDirectionSamples];
};

struct RTSGGXMicroFlakeMaterial: public RTVolumeMaterial
{
    Vector3                 SGGXStandardDeviation = Vector3{ 0.0f, 0.0f, 0.0f };
};

struct RTSGGXSurface: public RTMicrofacetMaterial
{
	uint32_t				Depth = 2, // Only applicable for the pseudo volume approach
							SampleCount = 256,
                            BasisMapWidth = 0,
                            BasisMapHeight = 0;
	Vector4					SGGXBasis;
	const void*				BasisMap = nullptr;
	const void*				StandardDeviationMap = nullptr;
};

typedef void (*GeometrySamplerFunction)(void* v0, void* v1, void* v2, uint32_t stride, const Vector3& barycentric, SampleData* data);

struct MeshOptions
{
    bool					TwoSided = false;
	GeometrySamplerFunction GeometrySampler = nullptr;
    void*                   UserData = nullptr;
};

void DefaultMeshSample(void* v0, void* v1, void* v2, uint32_t stride, const Vector3& barycentric, SampleData* data);

struct BTF;

struct RTBTF: public RTMaterial
{
    const BTF*              BTFData;
};

struct PNFormat
{
    Vector3                 Position;
    Vector3                 Normal;
};

struct PTNFormat
{
    Vector3                 Position;
    Vector2                 TexCoord;
    Vector3                 Normal;
};

struct PTTNFormat
{
    Vector3                 Position;
    Vector2                 TexCoord;
	Vector3					Tangent;
	Vector3                 Normal;
};

struct PcNFormat
{
    Vector3                 Position;
	float					InterpolationConstant;
	Vector3                 Normal;
};

struct PTcNFormat
{
    Vector3                 Position;
    Vector2                 TexCoord;
	float					InterpolationConstant;
	Vector3                 Normal;
};

struct PTTcNFormat
{
    Vector3                 Position;
    Vector2                 TexCoord;
	Vector3					Tangent;
	float					InterpolationConstant;
	Vector3                 Normal;
};

struct HairFormat
{
    Vector3                 Position;
    float                   Radius;
};

struct LightSample
{
    unsigned        GeometryID;
    Vector3         IncidentLight;
    float           Distance;
    Spectrum        Radiance;
    float           PDF;
};

enum class LightSourceType
{
	Directional,
	SphereArea,
    MeshArea,
    Point,
	Count
};

struct LightSource
{
	LightSourceType				  Type;

	LightSource(LightSourceType _type)
		:	Type(_type) {}
};

struct SphereAreaLight: public LightSource
{
    Sphere                        SphereShape;
    Spectrum                      Radiance;
    unsigned                      GeometryID;

	SphereAreaLight()
		:	LightSource(LightSourceType::SphereArea) {}
};

struct DirectionalLight: public LightSource
{
    Vector3                      Direction;
    Spectrum                     Radiance;

	DirectionalLight()
		:	LightSource(LightSourceType::Directional) {}
};

struct PointLight: public LightSource
{
    Vector3                      Position;
    Spectrum                     Radiance;

	PointLight()
		:	LightSource(LightSourceType::Point) {}
};

struct RTMesh;

struct MeshLight: public LightSource
{
    PoolPtr<int32_t>		IndexBuffer;
    PoolPtr<void>			VertexBuffer;
    uint32_t				Stride;
    uint32_t                TriangleCount;
    unsigned                GeometryID;
    PoolPtr<RTMicrofacetMaterial> Material;

    MeshLight()
		:	LightSource(LightSourceType::MeshArea) {}
};

EXPORT_CUDA_CONSTANT uint64_t AreaLightModels = 1ULL << (uint64_t)LightSourceType::SphereArea;

inline EXPORT_CUDA bool IsAreaLight(LightSourceType type)
{
    return (AreaLightModels & (1ULL << (uint64_t)type)) != 0;
}

#if defined(CPU_DEBUG) && defined(RAY_TRACER)
#   define EXPORT_TABLE
#endif

#ifdef EXPORT_TABLE
EXPORT_CUDA LightSample SphereAreaSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed);
EXPORT_CUDA float SphereAreaLightComputePDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir);
EXPORT_CUDA LightSample DirectionalSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed);
EXPORT_CUDA float PunctualLightPDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir);
EXPORT_CUDA LightSample MeshAreaSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed);
EXPORT_CUDA float MeshAreaLightComputePDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir);
EXPORT_CUDA LightSample PointSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed);

#if defined(EXPORT_TABLE) && (defined(__CUDACC__) || defined(DISABLE_CUDA))
EXPORT_CUDA LightSample SphereAreaSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed)
{
	auto& light = static_cast<const SphereAreaLight&>(light_source);
	LightSample smp;
    auto intersect = SampleSphereArea(enlit_pos, light.SphereShape.Center, light.SphereShape.Radius,
                                      (strata.XStrata + FastFloatRand(seed))/strata.TotalXStrata,
                                      (strata.YStrata + FastFloatRand(seed))/strata.TotalYStrata);
    smp.IncidentLight = intersect.Direction;
    smp.Distance = intersect.Distance;
    smp.Radiance = light.Radiance;
    smp.PDF = intersect.PDF; // TODO: Emission?
    smp.GeometryID = light.GeometryID;
    return smp;
};

EXPORT_CUDA float SphereAreaLightComputePDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir)
{
	auto& light = static_cast<const SphereAreaLight&>(light_source);
	return ComputeSphereAreaPDF(pos, dir, light.SphereShape.Center, light.SphereShape.Radius);
}

EXPORT_CUDA LightSample DirectionalSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed)
{
	auto& light = static_cast<const DirectionalLight&>(light_source);
    LightSample smp;
    smp.Distance = INFINITY;
    smp.IncidentLight = light.Direction;
    smp.Radiance = light.Radiance;
    smp.PDF = 1.0f;
    smp.GeometryID = INVALID_GEOMETRY;
    return smp;
}

EXPORT_CUDA LightSample PointSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed)
{
	auto& light = static_cast<const PointLight&>(light_source);
    LightSample smp;
    smp.Distance = INFINITY;

    auto dist_vec = light.Position - enlit_pos;
    float dist = Length(dist_vec);

    smp.IncidentLight = dist_vec/dist;
    smp.Radiance = light.Radiance/(Tempest::MathTau*dist*dist);
    smp.PDF = 1.0f;
    smp.GeometryID = INVALID_GEOMETRY;
    return smp;
}

EXPORT_CUDA float PunctualLightPDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir)
{
	return 1.0f;
}

EXPORT_CUDA LightSample MeshAreaSampleLightSource(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& strata, unsigned& seed)
{
    auto& light = static_cast<const MeshLight&>(light_source);

    auto tri_idx = FastUintRand(0, light.TriangleCount, seed);

    auto vert_buffer = reinterpret_cast<uint8_t*>(memory_pool(light.VertexBuffer));
    auto index_buffer = memory_pool(light.IndexBuffer);

    auto i0 = index_buffer[tri_idx*3],
         i1 = index_buffer[tri_idx*3 + 1],
         i2 = index_buffer[tri_idx*3 + 2];

    auto v0 = reinterpret_cast<Vector3*>(vert_buffer + i0*light.Stride),
         v1 = reinterpret_cast<Vector3*>(vert_buffer + i1*light.Stride),
         v2 = reinterpret_cast<Vector3*>(vert_buffer + i2*light.Stride);

    auto intersect = UniformSampleTriangleArea(enlit_pos, *v0, *v1, *v2, FastFloatRand(seed), FastFloatRand(seed));

    LightSample smp;
    smp.Distance = intersect.Distance;
    smp.IncidentLight = intersect.Direction;
    smp.Radiance = memory_pool(light.Material)->Diffuse;
    smp.PDF = intersect.PDF/light.TriangleCount;
    smp.GeometryID = light.GeometryID;
    return smp;
}

// Somewhat good estimate for convex objects
EXPORT_CUDA float MeshAreaLightComputePDF(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& dir)
{
    auto& light = static_cast<const MeshLight&>(light_source);

    float total_area = 0.0f;

    auto tri_count = light.TriangleCount;
    auto stride = light.Stride;
    auto vertex_buffer = reinterpret_cast<uint8_t*>(memory_pool(light.VertexBuffer));
    auto index_buffer = memory_pool(light.IndexBuffer);

    for(uint32_t tri_idx = 0; tri_idx < tri_count; ++tri_idx)
    {
        auto index0 = index_buffer[tri_idx*3],
             index1 = index_buffer[tri_idx*3 + 1],
             index2 = index_buffer[tri_idx*3 + 2];

        uint8_t* v0 = vertex_buffer + index0*stride;
	    uint8_t* v1 = vertex_buffer + index1*stride;
	    uint8_t* v2 = vertex_buffer + index2*stride;

        auto* p0 = reinterpret_cast<Vector3*>(v0);
        auto* p1 = reinterpret_cast<Vector3*>(v1);
        auto* p2 = reinterpret_cast<Vector3*>(v2);

        auto d0 = *p0 - *p1;
        auto d1 = *p2 - *p1;

        auto unorm = Cross(d1, d0);

        if(Dot(dir, unorm) >= 0.0f)
            continue;

        total_area += ProjectedTriangleSphereAreaApproximate(pos, *p0, *p1, *p2);
    }

    return total_area ? 1.0f/total_area : 0.0f;
}
#endif
#endif

typedef LightSample (*SampleLightSourceFunction)(MemoryPool memory_pool, const LightSource& light_source, const Vector3& enlit_pos, const Stratification& sample_data, unsigned& mirand);
typedef float (*LightSourceComputePDFFunction)(MemoryPool memory_pool, const LightSource& light_source, const Vector3& pos, const Vector3& vec);

#ifdef EXPORT_TABLE
EXPORT_TABLE const SampleLightSourceFunction SampleLightSourceFunctionLookup[(size_t)LightSourceType::Count]
{
	DirectionalSampleLightSource, //DirectionalLight,
	SphereAreaSampleLightSource, //SphereAreaLight,
    MeshAreaSampleLightSource, //MeshAreaLight,
    PointSampleLightSource, //PointLight,
};

EXPORT_TABLE const LightSourceComputePDFFunction LightSourceComputePDFLookup[(size_t)LightSourceType::Count]
{
	PunctualLightPDF, //DirectionalLight,
	SphereAreaLightComputePDF, //SphereAreaLight,
    MeshAreaLightComputePDF, //MeshAreaLight,
    PunctualLightPDF, //PointLight,
};
#else
extern const SampleLightSourceFunction SampleLightSourceFunctionLookup[(size_t)LightSourceType::Count];
extern const LightSourceComputePDFFunction LightSourceComputePDFLookup[(size_t)LightSourceType::Count];
#endif

#if defined(CPU_DEBUG) && defined(RAY_TRACER)
#   undef EXPORT_TABLE
#endif
}

#endif // _TEMPEST_RAY_TRACING_COMMON_HH_

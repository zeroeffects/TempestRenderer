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

#ifndef _TEMPEST_ILLUM_MODELS_HH_
#define _TEMPEST_ILLUM_MODELS_HH_

#include "tempest/math/windowing.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/vector3.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/functions.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/math/intersect.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/morton.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/mesh/lbvh2.hh"
#include "tempest/image/exr-image.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/image/btf.hh"

#include <climits>

#ifdef DISABLE_CUDA
namespace thrust
{
    using namespace std;
}
#else
#ifdef ILLUMINATION_MODEL_IMPLEMENTATION
#include <thrust/complex.h>
#endif
#endif

#include "tempest/graphics/ray-tracing/ggx-models-impl.hh"
#include "tempest/graphics/ray-tracing/beckmann-models-impl.hh"

namespace Tempest
{
struct DirectionalDensityCache
{
    float                DirectionalDensity;
};

struct MixCacheHeader
{
    float                Alpha;
};

struct StochasticRotatorHeader
{
    Quaternion           Rotation[1];
};

template<class T>
static size_t SizeOfMaterial(const RTMaterial*);

namespace Cpp
{
float BlinnPhongPDF(const SampleData&);
float UniformHemispherePDF(const SampleData&);
float AshikhminShirleyPDF(const SampleData&);
float WrongPDF(const SampleData&);
float PatchworkPDF(const SampleData&);
float MirrorPDF(const SampleData&);
float GGXMicrofacetPDF(const SampleData&);
float MixPDF(const SampleData&);
float GGXMicrofacetAnisotropicPDF(const SampleData&);
float StochasticRotatorPDF(const SampleData&);
float SGGXMicroFlakePDF(const SampleData&);
float SGGXMicroFlakeSurfacePDF(const SampleData&);

Spectrum BlinnPhongBRDF(const SampleData&);
Spectrum KajiyaKayBRDF(const SampleData&);
Spectrum AshikhminShirleyBRDF(const SampleData&);
Spectrum MicroFlakeTransmittance(const SampleData&);
Spectrum PatchworkBRDF(const SampleData&);
Spectrum MirrorBRDF(const SampleData&);
Spectrum GGXMicrofacetBRDF(const SampleData&);
Spectrum GGXMicrofacetConductorBRDF(const SampleData&);
Spectrum GGXMicrofacetDielectricBRDF(const SampleData&);
Spectrum MixBRDF(const SampleData&);
Spectrum GGXMicrofacetAnisotropicBRDF(const SampleData&);
Spectrum GGXMicrofacetAnisotropicConductorBRDF(const SampleData&);
Spectrum GGXMicrofacetAnisotropicDielectricBRDF(const SampleData&);
Spectrum StochasticRotatorBRDF(const SampleData&);
Spectrum SGGXMicroFlakeBRDF(const SampleData& sample_data);
Spectrum SpatiallyVaryingEmit(const SampleData& sample_data);
Spectrum SGGXMicroFlakeSurfaceBRDF(const SampleData& sample_data);
Spectrum SGGXMicroFlakePseudoVolumeBRDF(const SampleData& sample_data);
Spectrum BTFTransmittance(const SampleData& sample_data);

void BlinnPhongSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void UniformSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void AshikhminShirleySampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void MicroFlakeSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void PatchworkSampleIncidentLightFunction(const Stratification& strata, SampleData*, unsigned& seed);
void MirrorSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void GGXMicrofacetSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void MixSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void GGXMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void StochasticRotatorSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void SGGXMicroFlakeSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
void SGGXMicroFlakeSurfaceSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);

float MicroFlakeDensity(const SampleData& sample_data);
float SGGXMicroFlakeDensity(const SampleData& sample_data);
float SGGXMicroFlakeSurfaceDensity(const SampleData& sample_data);
}
#if defined(__CUDACC__) && !defined(ILLUMINATION_MODEL_STATIC_IMPLEMENTATION)
namespace Cuda
{
__device__ float BlinnPhongPDF(const SampleData&);
__device__ float UniformHemispherePDF(const SampleData&);
__device__ float AshikhminShirleyPDF(const SampleData&);
__device__ float WrongPDF(const SampleData&);
__device__ float PatchworkPDF(const SampleData&);
__device__ float MirrorPDF(const SampleData&);
__device__ float GGXMicrofacetPDF(const SampleData&);
__device__ float MixPDF(const SampleData&);
__device__ float GGXMicrofacetAnisotropicPDF(const SampleData&);
__device__ float StochasticRotatorPDF(const SampleData&);
__device__ float SGGXMicroFlakePDF(const SampleData&);
__device__ float SGGXMicroFlakeSurfacePDF(const SampleData&);

__device__ Spectrum BlinnPhongBRDF(const SampleData&);
__device__ Spectrum KajiyaKayBRDF(const SampleData&);
__device__ Spectrum AshikhminShirleyBRDF(const SampleData&);
__device__ Spectrum MicroFlakeTransmittance(const SampleData&);
__device__ Spectrum PatchworkBRDF(const SampleData&);
__device__ Spectrum MirrorBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetConductorBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetDielectricBRDF(const SampleData&);
__device__ Spectrum MixBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetAnisotropicBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetAnisotropicConductorBRDF(const SampleData&);
__device__ Spectrum GGXMicrofacetAnisotropicDielectricBRDF(const SampleData&);
__device__ Spectrum StochasticRotatorBRDF(const SampleData&);
__device__ Spectrum SGGXMicroFlakeBRDF(const SampleData& sample_data);
__device__ Spectrum SpatiallyVaryingEmit(const SampleData& sample_data);
__device__ Spectrum SGGXMicroFlakeSurfaceBRDF(const SampleData& sample_data);
__device__ Spectrum SGGXMicroFlakePseudoVolumeBRDF(const SampleData& sample_data);
__device__ Spectrum BTFTransmittance(const SampleData& sample_data);

__device__ void BlinnPhongSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void UniformSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void AshikhminShirleySampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void MicroFlakeSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void PatchworkSampleIncidentLightFunction(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void MirrorSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void GGXMicrofacetSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void MixSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void GGXMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void StochasticRotatorSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void SGGXMicroFlakeSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);
__device__ void SGGXMicroFlakeSurfaceSampleIncidentLight(const Stratification& strata, SampleData*, unsigned& seed);

__device__ float MicroFlakeDensity(const SampleData& sample_data);
__device__ float SGGXMicroFlakeDensity(const SampleData& sample_data);
__device__ float SGGXMicroFlakeSurfaceDensity(const SampleData& sample_data);
}
#endif


namespace MODEL_NAMESPACE
{
size_t MixMaterialSize(const RTMaterial* material);
size_t StochasticRotatorMaterialSize(const RTMaterial* material);
size_t SpatiallyVaryingEmissiveSize(const RTMaterial* material);

template<class T>
size_t SizeOfMaterial(const RTMaterial*)
{
    return sizeof(T);
}

SPLIT_EXPORT_CUDA bool PatchworkIsMirrorMaterial(const SampleData&);
SPLIT_EXPORT_CUDA bool DefaultMirrorMaterial(const SampleData&);
SPLIT_EXPORT_CUDA bool MixIsMirrorMaterial(const SampleData&);
SPLIT_EXPORT_CUDA bool StochasticRotatorIsMirrorMaterial(const SampleData&);

void MixCache(const SampleData& sample_data, unsigned& seed);
void StochasticRotatorCache(const SampleData& sample_data, unsigned& seed);
SPLIT_EXPORT_CUDA void PatchworkCache(const SampleData& sample_data, unsigned& seed);
SPLIT_EXPORT_CUDA void SGGXSurfaceCache(const SampleData& sample_data, unsigned& seed);

void DefaultMicrofacetMaterialSetup(RTMaterial* material);
void KajiyaKaySetup(RTMaterial* material);
void MicroFlakeSetup(RTMaterial* material);
void PatchworkSetup(RTMaterial* material);
void MixMaterialSetup(RTMaterial* material);
void StochasticRotatorSetup(RTMaterial* material);
void SGGXMicroFlakeSetup(RTMaterial *material);

#ifndef EXPORT_TABLE
extern SPLIT_TABLE_EXPORT_CUDA PDFFunction PDFLookup[(size_t)IlluminationModel::Count];
extern SPLIT_TABLE_EXPORT_CUDA TransmittanceFunction TransmittanceLookup[(size_t)IlluminationModel::Count];
extern SPLIT_TABLE_EXPORT_CUDA SampleIncidentLightFunction SampleIncidentLightLookup[(size_t)IlluminationModel::Count];
extern MaterialSetupFunction MaterialSetupLookup[(size_t)IlluminationModel::Count];
extern SPLIT_TABLE_EXPORT_CUDA MaterialIsMirrorFunction IsMirrorLookup[(size_t)IlluminationModel::Count];
extern SPLIT_TABLE_EXPORT_CUDA MaterialCacheFunction MaterialCacheLookup[(size_t)IlluminationModel::Count];
extern MaterialSizeFunction MaterialSizeLookup[(size_t)IlluminationModel::Count];
#endif

#ifdef EXPORT_TABLE
// Now we might start arguing what is better for I$
EXPORT_TABLE PDFFunction PDFLookup[(size_t)IlluminationModel::Count] =
{
    nullptr, //Emissive,
    BlinnPhongPDF, //BlinnPhong,
    UniformHemispherePDF, //KajiyaKay,
    AshikhminShirleyPDF, //AshikhminShirley,
    WrongPDF, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
#ifndef __CUDACC__
    PatchworkPDF, //Patchwork,
#else
    nullptr,
#endif
    MirrorPDF, //Mirror,
    nullptr, //Free4,
    GGXMicrofacetPDF, //GGXMicrofacet,
    GGXMicrofacetPDF, //GGXMicrofacetConductor,
    GGXMicrofacetPDF, //GGXMicrofacetDielectric,
    MixPDF, //Mix,
    GGXMicrofacetAnisotropicPDF, //GGXMicrofacetAnisotropic,
    GGXMicrofacetAnisotropicPDF, //GGXMicrofacetAnisotropicConductor,
    GGXMicrofacetAnisotropicPDF, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
#ifndef __CUDACC__
    StochasticRotatorPDF, //StochasticRotator,
#else
    nullptr,
#endif
    nullptr, //_Free7,
    SGGXMicroFlakePDF, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    UniformHemispherePDF, //SGGXMicroFlakeSurfacePDF, //SGGXSurface,
    SGGXMicroFlakeSurfacePDF, //SGGXPseudoVolume,
    nullptr, //_Free8,
    UniformHemispherePDF, //BTF,
    BeckmannMicrofacetPDF, //BeckmannMicrofacet,
    BeckmannMicrofacetPDF, //BeckmannMicrofacetConductor,
    BeckmannMicrofacetPDF, //BeckmannMicrofacetDielectric,
    BeckmannMicrofacetAnisotropicPDF, //BeckmannMicrofacetAnisotropic,
    BeckmannMicrofacetAnisotropicPDF, //BeckmannMicrofacetConductorAnisotropic,
    BeckmannMicrofacetAnisotropicPDF, //BeckmannMicrofacetDielectricAnisotropic,
};

EXPORT_TABLE TransmittanceFunction TransmittanceLookup[(size_t)IlluminationModel::Count] =
{
    nullptr, //Emissive,
    BlinnPhongBRDF, //BlinnPhong,
    KajiyaKayBRDF, //KajiyaKay,
    AshikhminShirleyBRDF, //AshikhminShirley,
    MicroFlakeTransmittance, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
#ifndef __CUDACC__
    PatchworkBRDF, //Patchwork,
#else
    nullptr,
#endif
    MirrorBRDF, //Mirror,
    nullptr, //Free4,
    GGXMicrofacetBRDF, //GGXMicrofacet,
    GGXMicrofacetConductorBRDF, //GGXMicrofacetConductor,
    GGXMicrofacetDielectricBRDF, //GGXMicrofacetDielectric,
    MixBRDF, //Mix,
    GGXMicrofacetAnisotropicBRDF, // GGXMicrofacetAnisotropic,
    GGXMicrofacetAnisotropicConductorBRDF, //GGXMicrofacetAnisotropicConductor,
    GGXMicrofacetAnisotropicDielectricBRDF, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
#ifndef __CUDACC__
    StochasticRotatorBRDF, //StochasticRotator,
#else
    nullptr,
#endif
    nullptr, //_Free7,
    SGGXMicroFlakeBRDF, //SGGXMicroFlake,
    SpatiallyVaryingEmit, //SpatiallyVaryingEmissive,
    SGGXMicroFlakeSurfaceBRDF, //SGGXSurface,
    SGGXMicroFlakePseudoVolumeBRDF, //SGGXPseudoVolume,
    nullptr, //_Free8,
    BTFTransmittance, //BTF,
    BeckmannMicrofacetBRDF, //BeckmannMicrofacet,
    BeckmannMicrofacetConductorBRDF, //BeckmannMicrofacetConductor,
    BeckmannMicrofacetDielectricBRDF, //BeckmannMicrofacetDielectric,
    BeckmannMicrofacetAnisotropicBRDF, //BeckmannMicrofacetAnisotropic,
    BeckmannMicrofacetAnisotropicConductorBRDF, //BeckmannMicrofacetConductorAnisotropic,
    BeckmannMicrofacetAnisotropicDielectricBRDF, //BeckmannMicrofacetDielectricAnisotropic,
};

EXPORT_TABLE SampleIncidentLightFunction SampleIncidentLightLookup[(size_t)IlluminationModel::Count]
{
    nullptr, //Emissive,
    BlinnPhongSampleIncidentLight, //BlinnPhong,
    UniformSampleIncidentLight, //KajiyaKay,
    AshikhminShirleySampleIncidentLight, //AshikhminShirley,
    MicroFlakeSampleIncidentLight, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
#ifndef __CUDACC__
    PatchworkSampleIncidentLightFunction, //Patchwork,
#else
    nullptr,
#endif
    MirrorSampleIncidentLight, //Mirror,
    nullptr, //Free4,
    GGXMicrofacetSampleIncidentLight, //GGXMicrofacet,
    GGXMicrofacetSampleIncidentLight, //GGXMicrofacetConductor,
    GGXMicrofacetSampleIncidentLight, //GGXMicrofacetDielectric,
    MixSampleIncidentLight, //Mix,
    GGXMicrofacetAnisotropicSampleIncidentLight, //GGXMicrofacetAnisotropic,
    GGXMicrofacetAnisotropicSampleIncidentLight, //GGXMicrofacetAnisotropicConductor,
    GGXMicrofacetAnisotropicSampleIncidentLight, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
#ifndef __CUDACC__
    StochasticRotatorSampleIncidentLight, //StochasticRotator,
#else
    nullptr,
#endif
    nullptr, //_Free7,
    SGGXMicroFlakeSampleIncidentLight, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    UniformSampleIncidentLight, //SGGXMicroFlakeSurfaceSampleIncidentLight, //SGGXSurface,
    SGGXMicroFlakeSurfaceSampleIncidentLight, //SGGXPseudoVolume,
    nullptr, //_Free8,
    UniformSampleIncidentLight, //BTF,
    BeckmannMicrofacetSampleIncidentLight, //BeckmannMicrofacet,
    BeckmannMicrofacetSampleIncidentLight, //BeckmannMicrofacetConductor,
    BeckmannMicrofacetSampleIncidentLight, //BeckmannMicrofacetDielectric,
    BeckmannMicrofacetAnisotropicSampleIncidentLight, //BeckmannMicrofacetAnisotropic,
    BeckmannMicrofacetAnisotropicSampleIncidentLight, //BeckmannMicrofacetConductorAnisotropic,
    BeckmannMicrofacetAnisotropicSampleIncidentLight, //BeckmannMicrofacetDielectricAnisotropic,
};

#ifndef __CUDACC__
MaterialSetupFunction MaterialSetupLookup[(size_t)IlluminationModel::Count]
{
    nullptr, //Emissive,
    DefaultMicrofacetMaterialSetup, //BlinnPhong,
    KajiyaKaySetup, //KajiyaKay,
    DefaultMicrofacetMaterialSetup, //AshikhminShirley,
    MicroFlakeSetup, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
    PatchworkSetup, //Patchwork,
    nullptr, //Mirror,
    nullptr, //Free4,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacet,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacetConductor,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacetDielectric,
    MixMaterialSetup, //Mix,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacetAnisotropic,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacetAnisotropicConductor,
    DefaultMicrofacetMaterialSetup, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
    StochasticRotatorSetup, //StochasticRotator,
    nullptr, //_Free7,
    SGGXMicroFlakeSetup, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    DefaultMicrofacetMaterialSetup, //SGGXSurface,
    DefaultMicrofacetMaterialSetup, //SGGXPseudoVolume,
    nullptr, //_Free8,
    nullptr, //BTF,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacet,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacetConductor,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacetDielectric,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacetAnisotropic,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacetConductorAnisotropic,
    DefaultMicrofacetMaterialSetup, //BeckmannMicrofacetDielectricAnisotropic,
};
#endif

EXPORT_TABLE MaterialIsMirrorFunction IsMirrorLookup[(size_t)IlluminationModel::Count]
{
    nullptr, //Emissive,
    nullptr, //BlinnPhong,
    nullptr, //KajiyaKay,
    nullptr, //AshikhminShirley,
    nullptr, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
    PatchworkIsMirrorMaterial, //Patchwork,
    DefaultMirrorMaterial, //Mirror,
    nullptr, //Free4,
    nullptr, //GGXMicrofacet,
    nullptr, //GGXMicrofacetConductor,
    nullptr, //GGXMicrofacetDielectric,
    MixIsMirrorMaterial, //Mix,
    nullptr, //GGXMicrofacetAnisotropic,
    nullptr, //GGXMicrofacetAnisotropicConductor,
    nullptr, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
    StochasticRotatorIsMirrorMaterial, //StochasticRotator,
    nullptr, //_Free7,
    nullptr, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    nullptr, //SGGXSurface,
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

EXPORT_TABLE MaterialCacheFunction MaterialCacheLookup[(size_t)IlluminationModel::Count]
{
	nullptr, //Emissive,
    nullptr, //BlinnPhong,
    nullptr, //KajiyaKay,
    nullptr, //AshikhminShirley,
    nullptr, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
    PatchworkCache, //Patchwork,
    nullptr, //Mirror,
    nullptr, //Free4,
    nullptr, //GGXMicrofacet,
    nullptr, //GGXMicrofacetConductor,
    nullptr, //GGXMicrofacetDielectric,
#ifndef __CUDACC__
    MixCache, //Mix,
#else
    nullptr,
#endif
    nullptr, //GGXMicrofacetAnisotropic,
    nullptr, //GGXMicrofacetAnisotropicConductor,
    nullptr, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
#ifndef __CUDACC__
    StochasticRotatorCache, //StochasticRotator,
#else
    nullptr,
#endif
    nullptr, //_Free7,
    nullptr, //SGGXMicroFlake,
    nullptr, //SpatiallyVaryingEmissive,
    SGGXSurfaceCache, //SGGXSurface,
    SGGXSurfaceCache, //SGGXPseudoVolume,
    nullptr, //_Free8,
    nullptr, //BTF,
    nullptr, //BeckmannMicrofacet,
    nullptr, //BeckmannMicrofacetConductor,
    nullptr, //BeckmannMicrofacetDielectric,
    nullptr, //BeckmannMicrofacetAnisotropic,
    nullptr, //BeckmannMicrofacetConductorAnisotropic,
    nullptr, //BeckmannMicrofacetDielectricAnisotropic,
};

EXPORT_TABLE VolumeDensityFunction DensityLookup[(size_t)VolumeIlluminationModel::Count]
{
    MicroFlakeDensity, //MicroFlake,
    SGGXMicroFlakeDensity, //SGGXMicroFlake,
    SGGXMicroFlakeSurfaceDensity, //SGGXMicroFlakeSurface,
};
#endif

EXPORT_CUDA_CONSTANT uint64_t VolumeModels = (1ULL << (uint64_t)IlluminationModel::MicroFlake) |
                                             (1ULL << (uint64_t)IlluminationModel::SGGXMicroFlake);

EXPORT_CUDA_CONSTANT uint64_t EmissiveModels = (1ULL << (uint64_t)IlluminationModel::Emissive) |
                                               (1ULL << (uint64_t)IlluminationModel::SpatiallyVaryingEmissive);

inline EXPORT_CUDA bool IsVolumeIlluminationModel(IlluminationModel model)
{
    return (VolumeModels & (1ULL << (uint64_t)model)) != 0;
}

inline EXPORT_CUDA bool IsEmissiveIlluminationModel(IlluminationModel model)
{
    return (EmissiveModels & (1ULL << (uint64_t)model)) != 0;
}

#if (defined(__CUDACC__) && defined(ILLUMINATION_MODEL_IMPLEMENTATION) && defined(EXPORT_TABLE)) || defined(CPU_DEBUG)
MaterialSizeFunction MaterialSizeLookup[(size_t)IlluminationModel::Count]
{
    SizeOfMaterial<RTMicrofacetMaterial>, //Emissive,
    SizeOfMaterial<RTMicrofacetMaterial>, //BlinnPhong,
    SizeOfMaterial<RTMicrofacetMaterial>, //KajiyaKay,
    SizeOfMaterial<RTMicrofacetMaterial>, //AshikhminShirley,
    nullptr, //MicroFlake,
    nullptr, //_Free1,
    nullptr, //_Free2,
    nullptr, //_Free3,
    nullptr, //Patchwork,
    SizeOfMaterial<RTMirrorMaterial>, //Mirror,
     nullptr, //Free4,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacet,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacetConductor,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacetDielectric,
    MixMaterialSize, //Mix,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacetAnisotropic,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacetAnisotropicConductor,
    SizeOfMaterial<RTMicrofacetMaterial>, //GGXMicrofacetAnisotropicDielectric,
	nullptr, //_Free5,
    nullptr, //_Free6,
    StochasticRotatorMaterialSize, //StochasticRotator,
    nullptr, //_Free7,
    nullptr, //SGGXMicroFlake,
    SpatiallyVaryingEmissiveSize, //SpatiallyVaryingEmissive,
    SizeOfMaterial<RTSGGXSurface>, //SGGXSurface,
    SizeOfMaterial<RTSGGXSurface>, //SGGXPseudoVolume,
    nullptr, //_Free8,
    SizeOfMaterial<RTBTF>, //BTF,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacet,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacetConductor,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacetDielectric,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacetAnisotropic,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacetConductorAnisotropic,
    SizeOfMaterial<RTMicrofacetMaterial>, //BeckmannMicrofacetDielectricAnisotropic,
};

size_t MixMaterialSize(const RTMaterial* material)
{
    auto* mix_material = static_cast<const RTMixMaterial*>(material);
    auto& mat0 = mix_material->getSubMaterial(0);
    size_t mat0_model = (size_t)mat0.Model;
    size_t mat0_size = MaterialSizeLookup[mat0_model](&mat0);
    auto& mat1 = mix_material->getSubMaterial(1);
    size_t mat1_model = (size_t)mat1.Model;
    size_t mat1_size = MaterialSizeLookup[mat1_model](&mat1);
    return sizeof(*mix_material) + mat0_size + mat1_size;
}

size_t StochasticRotatorMaterialSize(const RTMaterial* material)
{
    auto* rot_material = static_cast<const RTStochasticRotatorMaterial*>(material);
    auto& mat = rot_material->getSubMaterial();
    size_t mat_model = (size_t)mat.Model;
    size_t mat_size = MaterialSizeLookup[mat_model](&mat);
    return sizeof(*rot_material) + mat_size;
}

size_t SpatiallyVaryingEmissiveSize(const RTMaterial* material)
{
    return static_cast<const RTSpatiallyVaryingEmitter*>(material)->MaterialSize;
}
#endif

#ifdef ILLUMINATION_MODEL_IMPLEMENTATION
SPLIT_EXPORT_CUDA bool DefaultMirrorMaterial(const SampleData&)
{
    return true;
}

inline SPLIT_EXPORT_CUDA bool MirrorCheck(const SampleData& sample_data)
{
    auto material = sample_data.Material;
    auto is_mirror = IsMirrorLookup[(size_t)material->Model];
    return is_mirror != nullptr && (is_mirror == DefaultMirrorMaterial || is_mirror(sample_data));
}

#ifndef __CUDACC__
void DefaultMicrofacetMaterialSetup(RTMaterial* material)
{
    auto microfacet_material = static_cast<RTMicrofacetMaterial*>(material);

    if(SampleIncidentLightLookup[(size_t)material->Model])
    {
        float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    
	    if(max_value > 1.0f)
        {
            microfacet_material->Diffuse /= max_value;
            microfacet_material->Specular /= max_value;
        }
    }
    microfacet_material->Normalization = 1.0f;
}
#endif

// Blinn-Phong
SPLIT_EXPORT_CUDA float BlinnPhongPDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    Vector3 half_vec = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);

    Tempest::Spectrum albedo = microfacet_material->Diffuse;
    auto albedo_map = microfacet_material->DiffuseMap;
    if(albedo_map)
    {
        albedo = SampleSpectrum(albedo_map, sample_data.TexCoord);
    }

    float max_value = MaxValue(albedo + microfacet_material->Specular);
    float dot_light_norm = Dot(sample_data.IncidentLight, sample_data.Normal);
    if(dot_light_norm < 0.0f)
        return 0.0f; // TODO: Fix energy loss when generating samples

    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
    float pdf0 = (microfacet_material->SpecularPower.x + 1)*powf(Dot(half_vec, sample_data.Normal), microfacet_material->SpecularPower.x) * (1.0f / (2.0f*MathPi));
    float transform = 4.0f * Dot(half_vec, sample_data.OutgoingLight);
    pdf0 = transform > 0.0f && pdf0 > 0.0f ? pdf0 / transform : 0.0f;
    float pdf1 = dot_light_norm * (1.0f/MathPi);
    TGE_ASSERT(std::isfinite(pdf0) && std::isfinite(pdf1), "Bad Blinn-Phong PDF");
    return (pdf0 - pdf1)*ratio + pdf1;
}

SPLIT_EXPORT_CUDA void BlinnPhongSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data->Material);

	float x_ray = (strata.XStrata + FastFloatRand(mirand))/strata.TotalXStrata;
	float y_ray = (strata.YStrata + FastFloatRand(mirand))/strata.TotalYStrata;
	Tempest::Spectrum albedo = microfacet_material->Diffuse;
    auto albedo_map = microfacet_material->DiffuseMap;
    if(albedo_map)
    {
        albedo = SampleSpectrum(albedo_map, sample_data->TexCoord);
    }

    float max_value = MaxValue(albedo + microfacet_material->Specular);
	float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);
	if(FastFloatRand(mirand) <= ratio)
	{
		auto micro_norm = PowerCosineSampleHemisphere(x_ray, y_ray, microfacet_material->SpecularPower.x);
        micro_norm = Normalize(surface_space.transform(micro_norm));

		sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, micro_norm);
	}
	else
	{
		sample_data->IncidentLight = CosineSampleHemisphere(x_ray, y_ray);
        sample_data->IncidentLight = Normalize(surface_space.transform(sample_data->IncidentLight));
	}

    sample_data->PDF = BlinnPhongPDF(*sample_data);
}

SPLIT_EXPORT_CUDA Spectrum BlinnPhongBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    Tempest::Spectrum albedo = microfacet_material->Diffuse;
    Tempest::Spectrum specular_str = microfacet_material->Specular;
    float specular_power = microfacet_material->SpecularPower.x;

    auto albedo_map = microfacet_material->DiffuseMap;
    if(albedo_map)
    {
        albedo = SampleSpectrum(albedo_map, sample_data.TexCoord);
    }

    float dotNormLight = Maxf(0.0f, Dot(sample_data.Normal, sample_data.IncidentLight));
    Vector3 micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);
    specular_str *= pow(Maxf(0.0f, Dot(sample_data.Normal, micro_norm)), specular_power);
    float cos_theta = Clamp(Dot(sample_data.IncidentLight, micro_norm), 0.0f, 1.0f);
    float fresnel = FresnelSchlick(microfacet_material->Fresnel.x, cos_theta);
	auto result = dotNormLight * (albedo + fresnel*specular_str*(specular_power + 8.0f)*(1.0f/8.0f)) * (1.0f / MathPi);
    TGE_ASSERT(std::isfinite(Array(result)[0]), "Invalid data");
    return result;
}

#ifndef __CUDACC__
// Kajiya-Kay
void KajiyaKaySetup(RTMaterial* material)
{
    auto* microfacet_material = static_cast<RTMicrofacetMaterial*>(material);

    const uint32_t integral_samples = 1024;

    // Simpson's composite rule
    float spec_power = microfacet_material->SpecularPower.x;
	float integral = Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, MathPi*0.5f, integral_samples,
	                    [spec_power](float theta)
                        {
		                    return std::pow(std::sin(theta), spec_power + 1.0f /* integration in polar coordinates */);
	                    });

	microfacet_material->Normalization = 1/(integral*2.0f*MathPi);
}
#endif

SPLIT_EXPORT_CUDA float UniformHemispherePDF(const SampleData& sample_data)
{
    float dot_norm_dir = Dot(sample_data.Normal, sample_data.IncidentLight);
    return dot_norm_dir >= 0.0f ? 1.0f/(2.0f*MathPi) : 0.0f;
}

SPLIT_EXPORT_CUDA void UniformSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);

    sample_data->IncidentLight = UniformSampleHemisphere((strata.XStrata + FastFloatRand(mirand))/strata.TotalXStrata, (strata.YStrata + FastFloatRand(mirand))/strata.TotalYStrata);
    sample_data->PDF = 1.0f/(2.0f*MathPi);

    sample_data->IncidentLight = Normalize(surface_space.transform(sample_data->IncidentLight));
}

SPLIT_EXPORT_CUDA Spectrum KajiyaKayBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    float dotTangentLight = Dot(sample_data.Tangent, sample_data.IncidentLight);
    float dotTangentEye = Dot(sample_data.Tangent, sample_data.OutgoingLight);
    float sineTangentLight = sqrt(Maxf(0.0f, 1.0f - dotTangentLight*dotTangentLight));
    float sineTangentEye = sqrt(Maxf(0.0f, 1.0f - dotTangentEye*dotTangentEye));
    float specular_str = pow(Maxf(0.0f, dotTangentLight * dotTangentEye + sineTangentLight * sineTangentEye), microfacet_material->SpecularPower.x);
    Spectrum diffuse = microfacet_material->Diffuse;
    Spectrum specular = microfacet_material->Specular * specular_str;
    return sineTangentLight * (diffuse * (1.0f/(2.0f*MathPi*MathPi*0.25f)) + microfacet_material->Normalization * specular); // TODO: adjust for whole hemisphere
}

// Ashikhmin-Shirley
SPLIT_EXPORT_CUDA float AshikhminShirleyPDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    Vector3 half_vec = sample_data.IncidentLight + sample_data.OutgoingLight;
    NormalizeSelf(&half_vec);
    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float dot_light_norm = Dot(sample_data.IncidentLight, sample_data.Normal);
    if(dot_light_norm < 0.0f)
        return 0.0f;
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
    float dot_hv_tan = Dot(half_vec, sample_data.Tangent);
    float dot_hv_bin = Dot(half_vec, sample_data.Binormal);
    float dot_hv_norm = Dot(half_vec, sample_data.Normal);
    float spec_power = (microfacet_material->SpecularPower.x*dot_hv_tan*dot_hv_tan + microfacet_material->SpecularPower.y*dot_hv_bin*dot_hv_bin)/(1 - dot_hv_norm*dot_hv_norm);

	float pdf0 = sqrtf((microfacet_material->SpecularPower.x + 1.0f)*(microfacet_material->SpecularPower.y + 1.0f)) * powf(Dot(sample_data.Normal, half_vec), spec_power) / (2.0f * MathPi);
    float transform = 4.0f * Dot(half_vec, sample_data.OutgoingLight);
    pdf0 = transform >= 0.0f && pdf0 >= 0.0f ? pdf0 / transform : 0.0f;
    float pdf1 = dot_light_norm * (1.0f/MathPi);
    return (pdf0 - pdf1)*ratio + pdf1;
}

SPLIT_EXPORT_CUDA void AshikhminShirleySampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data->Material);

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);

	float x_ray = (strata.XStrata + FastFloatRand(mirand))/strata.TotalXStrata;
	float y_ray = (strata.YStrata + FastFloatRand(mirand))/strata.TotalYStrata;
	float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
	float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
	if(FastFloatRand(mirand) <= ratio)
	{
		auto micro_norm = AnisoPowerCosineSampleHemisphere(x_ray, y_ray, microfacet_material->SpecularPower);
        micro_norm = Normalize(surface_space.transform(micro_norm));

		sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, micro_norm);
	}
	else
	{
		sample_data->IncidentLight = CosineSampleHemisphere(x_ray, y_ray);
        sample_data->IncidentLight = Normalize(surface_space.transform(sample_data->IncidentLight));
	}

	sample_data->PDF = AshikhminShirleyPDF(*sample_data);
}

SPLIT_EXPORT_CUDA Spectrum AshikhminShirleyBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

	Spectrum diffuse = microfacet_material->Diffuse;
    Spectrum specular = microfacet_material->Specular;

	Vector3 micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);

	float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
	float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
	float dot_hv_tan = Dot(micro_norm, sample_data.Tangent);
	float dot_hv_bin = Dot(micro_norm, sample_data.Binormal);
	float dot_hv_norm = Dot(micro_norm, sample_data.Normal);
	float spec_power = (microfacet_material->SpecularPower.x*dot_hv_tan*dot_hv_tan + microfacet_material->SpecularPower.y*dot_hv_bin*dot_hv_bin)/(1 - dot_hv_norm*dot_hv_norm);
	float dot_half_light = Dot(micro_norm, sample_data.IncidentLight);
	float dot_norm_inc_light = Dot(sample_data.Normal, sample_data.IncidentLight);
    float dot_norm_out_light = Dot(sample_data.Normal, sample_data.OutgoingLight);
    if(dot_norm_inc_light <= 0.0f || dot_norm_out_light <= 0.0f)
        return Tempest::Spectrum{};
	float fresnel_term = FresnelSchlick(microfacet_material->Fresnel.x, dot_half_light);
	specular *= sqrtf((microfacet_material->SpecularPower.x + 1.0f)*(microfacet_material->SpecularPower.y + 1.0f)) * powf(Dot(sample_data.Normal, micro_norm), spec_power) /
				(8.0f * MathPi * dot_half_light * Maxf(dot_norm_inc_light, dot_norm_out_light));
	float diffuse_factors = (1.0f - pow(1.0f - 0.5f*Dot(sample_data.Normal, sample_data.IncidentLight), 5.0f)) * (1.0f - pow(1.0f - 0.5f*Dot(sample_data.Normal, sample_data.OutgoingLight), 5.0f));
	diffuse *= 28.0f * (1.0f - microfacet_material->Fresnel.x) * diffuse_factors / (23.0f * MathPi);
    auto ret = dot_norm_inc_light*(diffuse + specular);
    TGE_ASSERT(Array(ret)[0] >= 0.0f, "Invalid BRDF");
	return ret;
}

// MicroFlake Model
const size_t MicroFlakeMaxIntegralSamples = 16*1024;

SPLIT_EXPORT_CUDA float WrongPDF(const SampleData& sample_data)
{
    return 1.0f;
}

SPLIT_EXPORT_CUDA void SampleMicroflakeIncidentLight(SampleData* sample_data, unsigned int& mirand)
{
    auto* microflake_material = static_cast<const RTMicroFlakeMaterial*>(sample_data->Material);

    float rcp_sqrt2_stdev = 1.0f / (Sqrt2*microflake_material->StandardDeviation);
    for(;;)
    {
        float r1 = FastFloatRand(mirand);
        float theta = BrentMethod(0, MathPi, 1e-3f,
            [rcp_sqrt2_stdev, r1](float theta)
            {
                return 0.5f - 0.5f * std::erff(cosf(theta) * rcp_sqrt2_stdev) / std::erff(rcp_sqrt2_stdev) - r1;
            });
        float omega = 2*MathPi*FastFloatRand(mirand);
        float sin_omega, cos_omega;
        FastSinCos(omega, &sin_omega, &cos_omega);
        float cos_theta = cos(theta);
        float sin_theta = sqrtf(1 - cos_theta*cos_theta);
		Vector3 micro_flake{cos_omega*sin_theta, sin_omega*sin_theta, cos_theta};
        float dot_mf = Dot(sample_data->OutgoingLight, micro_flake);
        if(FastFloatRand(mirand) < fabsf(dot_mf))
        {
            Vector3 result =  Normalize(micro_flake * dot_mf * 2.0f - sample_data->OutgoingLight);
            sample_data->IncidentLight = result;
            sample_data->PDF = 1.0f; // not trivial to compute
            return;
        }
    }
}

SPLIT_EXPORT_CUDA void MicroFlakeSampleIncidentLight(const Stratification&, SampleData* sample_data, unsigned& mirand)
{
    if(sample_data->Tangent.x == 0.0f && sample_data->Tangent.y == 0.0f && sample_data->Tangent.z == 0.0f)
    {
        sample_data->IncidentLight = sample_data->OutgoingLight;
        sample_data->PDF = 0.0f;
        return;
    }

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);
                      
    Vector3 out_light = sample_data->OutgoingLight;
    sample_data->OutgoingLight = surface_space.transformRotationInverse(out_light);

    SampleMicroflakeIncidentLight(sample_data, mirand);
    
    sample_data->IncidentLight = surface_space.transform(sample_data->IncidentLight);
    sample_data->OutgoingLight = out_light;
}

SPLIT_EXPORT_CUDA Spectrum MicroFlakeTransmittance(const SampleData& sample_data)
{
    auto* microflake_material = static_cast<const RTMicroFlakeMaterial*>(sample_data.Material);

	Vector3 micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);

    float dir_density = sample_data.DirectionalDensity;

	float cos_theta_h = Dot(micro_norm, sample_data.Tangent);
	float stdev = microflake_material->StandardDeviation;
	float pdf_cos_theta = microflake_material->Normalization * exp(-cos_theta_h*cos_theta_h / (2.0f*stdev*stdev));
	return ToSpectrum(0.5f * pdf_cos_theta / dir_density);
}

SPLIT_EXPORT_CUDA float MicroFlakeDensity(const SampleData& sample_data)
{
    auto micro_flake_material = static_cast<const RTMicroFlakeMaterial*>(sample_data.Material);

    float dot_dir_orient = Clamp(Dot(sample_data.IncidentLight, sample_data.Tangent), -1.0f, 1.0f);
    float angle = acosf(dot_dir_orient) * ((MicroFlakeMaxDirectionSamples - 1) / Tempest::MathPi);
    float idx_f = Maxf(angle, 0.0f);

    int idx0 = Mini(FastFloorToInt(idx_f), (int)MicroFlakeMaxDirectionSamples - 1);
    int idx1 = Mini((int)FastCeil(idx_f), (int)MicroFlakeMaxDirectionSamples - 1);

    return micro_flake_material->DirectionalDensity[idx0] + (idx_f - idx0)*(micro_flake_material->DirectionalDensity[idx1] - micro_flake_material->DirectionalDensity[idx0]);
}

#ifndef __CUDACC__
void MicroFlakeSetup(RTMaterial* material)
{
    auto micro_flake_material = static_cast<RTMicroFlakeMaterial*>(material);
    micro_flake_material->VolumeModel = VolumeIlluminationModel::MicroFlake;

    float stddev =  micro_flake_material->StandardDeviation;
    float norm_factor = pow((2.0f*MathPi), 3.0f/2.0f)*stddev*std::erff(1.0f/(Sqrt2*stddev));
    float c1 = micro_flake_material->Normalization = 1.0f/norm_factor;
    float* precomp_table = micro_flake_material->DirectionalDensity;
    unsigned mirand = 1;
    float step = MathPi / (MicroFlakeMaxDirectionSamples - 1);
    float angle = 0.0f;

    float c0 = 2.0f*stddev*stddev;

    for(size_t i = 0; i < MicroFlakeMaxDirectionSamples; ++i, angle += step)
    { 
        const float pdf = 1.0f / MathPi;

        float sin_angle, cos_angle;
        FastSinCos(angle, &sin_angle, &cos_angle);

        float result = Tempest::StratifiedMonteCarloIntegratorSphere(MicroFlakeMaxIntegralSamples,
            [sin_angle, cos_angle, c0, c1](const Tempest::Vector3& dir)
            {
                return fabsf(Dot(dir, Vector3{0.0f, sin_angle, cos_angle }))*ZhaoMicroFlakeNDF(c0, c1, dir.z); // Assume wf = (0, 0, 1)
            });

        /*
        // Highly symmetrical
        float result = StratifiedMonteCarloIntegrator(0.0f, MathPi, MicroFlakeMaxIntegralSamples,
            [const_factor, angle](float rand_angle)
            {
                float sin_ra, cos_ra;
                Tempest::FastSinCos(rand_angle, &sin_ra, &cos_ra);

                return sin_ra * fabsf(cos(angle - rand_angle)) * exp(-cos_ra*cos_ra*const_factor); // Assume wf = (0, 0, 1)
            });

        result *= material->Normalization;
        result *= 2.0f * MathPi;
        */
        
        precomp_table[i] = result;
    }
}
#endif

// Patchwork
struct PatchworkCacheRecord
{
	Vector3 Tangent;
	Vector3 Binormal;
	Vector2 TexCoord;
    union
    {
        uint8_t   MiniScratchMemory[2*4];
        uint64_t  MiniScratchMemoryUInt64;
    };
};

struct PatchworkCacheHeader
{
	size_t				 IntersectedPatchCount;
	PatchworkCacheRecord IntersectedPatches[]; 
};

// :)
inline EXPORT_CUDA void PatchSampleData(SampleData& sample_data, RTMaterial* material, const PatchworkCacheRecord& patch_info)
{
	sample_data.Material = material;
	sample_data.Tangent  = patch_info.Tangent;
	sample_data.Binormal = patch_info.Binormal;
	sample_data.TexCoord = patch_info.TexCoord;
    sample_data.ScratchMemory = const_cast<uint8_t*>(patch_info.MiniScratchMemory);
}

SPLIT_EXPORT_CUDA float PatchworkPDF(const SampleData& sample_data)
{
    auto material = reinterpret_cast<const RTPatchworkMaterial*>(sample_data.Material);
	auto intersection_data = reinterpret_cast<PatchworkCacheHeader*>(sample_data.ScratchMemory);

    float pdf = 0.0f;
    for(uint32_t i = 0; i < intersection_data->IntersectedPatchCount; ++i)
    {
        SampleData patch_sample_data = sample_data;
        PatchSampleData(patch_sample_data, material->PatchMaterial, intersection_data->IntersectedPatches[i]);
        pdf += PDFLookup[(size_t)material->PatchMaterial->Model](patch_sample_data);
    }
    
    return intersection_data->IntersectedPatchCount ? pdf/intersection_data->IntersectedPatchCount : PDFLookup[(size_t)material->BaseModel](sample_data);
}

SPLIT_EXPORT_CUDA Spectrum PatchworkBRDF(const SampleData& sample_data)
{
    auto material = reinterpret_cast<const RTPatchworkMaterial*>(sample_data.Material);
	auto intersection_data = reinterpret_cast<PatchworkCacheHeader*>(sample_data.ScratchMemory);

    Spectrum radiance{};
    for(uint32_t i = 0; i < intersection_data->IntersectedPatchCount; ++i)
    {
        SampleData patch_sample_data = sample_data;
        PatchSampleData(patch_sample_data, material->PatchMaterial, intersection_data->IntersectedPatches[i]);
        radiance += TransmittanceLookup[(size_t)material->PatchMaterial->Model](patch_sample_data);
    }

    return intersection_data->IntersectedPatchCount ? radiance/(float)intersection_data->IntersectedPatchCount : TransmittanceLookup[(size_t)material->BaseModel](sample_data);
}

SPLIT_EXPORT_CUDA void PatchworkSampleIncidentLightFunction(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    auto material = reinterpret_cast<const RTPatchworkMaterial*>(sample_data->Material);
	auto intersection_data = reinterpret_cast<PatchworkCacheHeader*>(sample_data->ScratchMemory);

    if(intersection_data->IntersectedPatchCount == 0)
    {
        return SampleIncidentLightLookup[(size_t)material->BaseModel](strata, sample_data, mirand);
    }
    
    uint32_t chosen_idx = (uint32_t)(FastFloatRand(mirand)*(intersection_data->IntersectedPatchCount - 1));

    SampleData cur_patch_sample_data = *sample_data;
	PatchSampleData(cur_patch_sample_data, material->PatchMaterial, intersection_data->IntersectedPatches[chosen_idx]);
    SampleIncidentLightLookup[(size_t)material->PatchMaterial->Model](strata, &cur_patch_sample_data, mirand);

    float total_pdf = 0.0f;
    size_t idx = 0;
    for(uint32_t i = 0; i < intersection_data->IntersectedPatchCount; ++i)
    {
        SampleData patch_sample_data = *sample_data;
        PatchSampleData(patch_sample_data, material->PatchMaterial, intersection_data->IntersectedPatches[i]);
        if(idx != chosen_idx++ && !MirrorCheck(patch_sample_data))
        {
            total_pdf += PDFLookup[(size_t)material->PatchMaterial->Model](patch_sample_data);
        }
    }

    sample_data->IncidentLight = cur_patch_sample_data.IncidentLight;
    sample_data->PDF = (cur_patch_sample_data.PDF + total_pdf)/intersection_data->IntersectedPatchCount;
}

#ifndef __CUDACC__
void PatchworkSetup(RTMaterial* material)
{
    auto patchwork_material = reinterpret_cast<RTPatchworkMaterial*>(material);

    auto patch_count = patchwork_material->PatchCount;

	uint32_t amp_count = 1;
	uint32_t total_node_count = amp_count*patch_count;
    std::unique_ptr<LBVH2Node<AABB2>[]> interm_nodes(new LBVH2Node<AABB2>[total_node_count]);

    for(uint32_t i = 0; i < patch_count; ++i)
    {
        auto& patch = patchwork_material->Patches[i];
        patch.RotateScaleInverse = patch.RotateScale.inverse();
        patch.Rotate = patch.RotateScale;
		NormalizeSelf(&patch.Rotate.column(0));
		NormalizeSelf(&patch.Rotate.column(1));

		auto& interm_node = interm_nodes[i];
		interm_node.Patch = LBVH_LEAF_DECORATION | i;
		Rect2Bounds(patch.RotateScale, patch.Translate, &interm_node.Bounds);
    }

    patchwork_material->BVH = GenerateLBVH(interm_nodes.get(), total_node_count);

    MaterialSetupLookup[(size_t)patchwork_material->BaseModel](material);
}
#endif

SPLIT_EXPORT_CUDA bool PatchworkIsMirrorMaterial(const SampleData& sample_data)
{
    auto patchwork_material = reinterpret_cast<const RTPatchworkMaterial*>(sample_data.Material);
	auto intersection_data = reinterpret_cast<PatchworkCacheHeader*>(sample_data.ScratchMemory);
    auto is_patch_mirror = IsMirrorLookup[(size_t)patchwork_material->PatchMaterial->Model];
    if(is_patch_mirror && intersection_data->IntersectedPatchCount)
    {
		if(is_patch_mirror == DefaultMirrorMaterial)
                return true;
        for(uint32_t i = 0; i < intersection_data->IntersectedPatchCount; ++i)
		{
			SampleData patch_sample_data = sample_data;
			PatchSampleData(patch_sample_data, patchwork_material->PatchMaterial, intersection_data->IntersectedPatches[i]);
            is_patch_mirror(patch_sample_data);
        }
    }
    auto is_base_mirror = IsMirrorLookup[(size_t)patchwork_material->BaseModel];
    return is_base_mirror != nullptr && (is_base_mirror == DefaultMirrorMaterial || is_base_mirror(sample_data));
}

struct PatchIntersectTest
{
	Patch* Patches;
	const SampleData& Data;
	PatchworkCacheHeader* IntersectionData;

	EXPORT_CUDA PatchIntersectTest(Patch* patches, const SampleData& sample_data,	PatchworkCacheHeader* intersection_data)
		:	Patches(patches),
			Data(sample_data),
			IntersectionData(intersection_data) {}

	inline EXPORT_CUDA void operator()(uint32_t geom_id, const Vector2& pos)
	{
		auto& patch = Patches[geom_id];
		Vector2 pos_patch = pos - patch.Translate;
		/*
		if(patch.Repeat.x)
			pos_patch.x = fmodf(pos_patch.x, patch.Repeat.x);
		if(patch.Repeat.y)
			pos_patch.y = fmodf(pos_patch.y, patch.Repeat.y);
		*/
		pos_patch = patch.RotateScaleInverse.transform(pos_patch);

		if(-1.0f <= pos_patch.x && pos_patch.x <= 1.0f &&
			-1.0f <= pos_patch.y && pos_patch.y <= 1.0f)
		{
			if(sizeof(PatchworkCacheHeader) + (IntersectionData->IntersectedPatchCount + 1)*sizeof(PatchworkCacheRecord) > ScratchMemoryPerThread)
				return;
			auto& intersect = IntersectionData->IntersectedPatches[IntersectionData->IntersectedPatchCount++];
			intersect.Tangent  = patch.Rotate.column(0).x*Data.Tangent + patch.Rotate.column(0).y*Data.Binormal;
			intersect.Binormal = patch.Rotate.column(1).x*Data.Tangent + patch.Rotate.column(1).y*Data.Binormal;
			intersect.TexCoord = pos_patch*0.5f + Vector2{0.5f, 0.5f};
			intersect.MiniScratchMemoryUInt64 = patch.MiniScratchMemoryUInt64;
		}
	}
};

SPLIT_EXPORT_CUDA void PatchworkCache(const SampleData& sample_data, unsigned& seed)
{
	auto patchwork_material = reinterpret_cast<const RTPatchworkMaterial*>(sample_data.Material);
	auto intersection_data = reinterpret_cast<PatchworkCacheHeader*>(sample_data.ScratchMemory);
	intersection_data->IntersectedPatchCount = 0;
 
    Vector2 pos = 2.0f*sample_data.TexCoord - Vector2{1.0f, 1.0f};

	PatchIntersectTest box_intersect(patchwork_material->Patches, sample_data, intersection_data);
    /*
	for(uint32_t i = 0; i < patchwork_material->PatchCount; ++i)
    {
        box_intersect(i, pos);
    }
    /*/
    IntersectLBVHNode(patchwork_material->BVH, 0, pos, box_intersect);
    //*/
}

// Mirror
SPLIT_EXPORT_CUDA float MirrorPDF(const SampleData& sample_data)
{
    TGE_ASSERT(false, "Don't ask about the chance of having the correct pair of outgoing and incident light vectors when you use a mirror");
    return 0.0f;
}

SPLIT_EXPORT_CUDA Spectrum MirrorBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMirrorMaterial*>(sample_data.Material);

    TGE_ASSERT(1.0f - Dot(Reflect(sample_data.OutgoingLight, sample_data.Normal), sample_data.IncidentLight) < 1e-3f, "Invalid light");
    return microfacet_material->Specular;
}

SPLIT_EXPORT_CUDA void MirrorSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, sample_data->Normal);
    NormalizeSelf(&sample_data->IncidentLight);
    sample_data->PDF = 1.0f;
}

inline SPLIT_EXPORT_CUDA float SampleMixAlpha(const RTMixMaterial* mix_material, const SampleData& sample_data)
{
#ifndef __CUDACC__
TGE_ASSERT(0.0f <= reinterpret_cast<MixCacheHeader*>(sample_data.ScratchMemory)->Alpha && reinterpret_cast<MixCacheHeader*>(sample_data.ScratchMemory)->Alpha <= 1.0f, "Invalid mixing factor");
    return reinterpret_cast<MixCacheHeader*>(sample_data.ScratchMemory)->Alpha;
#else
    // Not much advantage in caching it when working with CUDA.
    return SampleRed(mix_material->MixTexture, sample_data.TexCoord);
#endif
}

template<class TFunc>
EXPORT_CUDA auto EvaluateFunctionForMaterial(TFunc func, const RTMaterial* material, const SampleData& sample_data) -> decltype(func(sample_data))
{
    auto mtl_sample_data = sample_data;
    mtl_sample_data.Material = material;
    return func(mtl_sample_data);
}

template<class TFuncTable>
EXPORT_CUDA auto EvaluateTableFunctionForMaterial(TFuncTable table, const RTMaterial* material, const SampleData& sample_data) -> decltype(table[0](sample_data))
{
    return EvaluateFunctionForMaterial(table[(size_t)material->Model], material, sample_data);
}

SPLIT_EXPORT_CUDA float MixPDF(const SampleData& sample_data)
{
    auto* mix_material = static_cast<const RTMixMaterial*>(sample_data.Material);
    float alpha = SampleMixAlpha(mix_material, sample_data);

    if(alpha == 0.0f)
    {
        return EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(0), sample_data);
    }
    else if(alpha == 1.0f)
    {
        return EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(1), sample_data);
    }

    float pdf0 = EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(0), sample_data);
    float pdf1 = EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(1), sample_data);

    return pdf0 + alpha*(pdf1 - pdf0);
}

SPLIT_EXPORT_CUDA Spectrum MixBRDF(const SampleData& sample_data)
{
    auto* mix_material = static_cast<const RTMixMaterial*>(sample_data.Material);
    float alpha = SampleMixAlpha(mix_material, sample_data);

    if(alpha == 0.0f)
    {
        return EvaluateTableFunctionForMaterial(TransmittanceLookup, &mix_material->getSubMaterial(0), sample_data);
    }
    else if(alpha == 1.0f)
    {
        return EvaluateTableFunctionForMaterial(TransmittanceLookup, &mix_material->getSubMaterial(1), sample_data);
    }

    auto spec0 = EvaluateTableFunctionForMaterial(TransmittanceLookup, &mix_material->getSubMaterial(0), sample_data);
    auto spec1 = EvaluateTableFunctionForMaterial(TransmittanceLookup, &mix_material->getSubMaterial(1), sample_data);

    return spec0*(1 - alpha) + spec1*alpha;
}

SPLIT_EXPORT_CUDA void MixSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
    auto* mix_material = static_cast<const RTMixMaterial*>(sample_data->Material);
    float alpha = SampleMixAlpha(mix_material, *sample_data);

    if(alpha == 0.0f)
    {
        auto mtl0_sample_data = *sample_data;
        auto mtl0 = mtl0_sample_data.Material = &mix_material->getSubMaterial(0);
        SampleIncidentLightLookup[(size_t)mtl0->Model](strata, &mtl0_sample_data, seed);
        sample_data->IncidentLight = mtl0_sample_data.IncidentLight;
        sample_data->PDF = mtl0_sample_data.PDF;
    }
    else if(alpha == 1.0f)
    {
        auto mtl1_sample_data = *sample_data;
        auto mtl1 = mtl1_sample_data.Material = &mix_material->getSubMaterial(1);
        SampleIncidentLightLookup[(size_t)mtl1->Model](strata, &mtl1_sample_data, seed);
        sample_data->IncidentLight = mtl1_sample_data.IncidentLight;
        sample_data->PDF = mtl1_sample_data.PDF;
    }
    else if(FastFloatRand(seed) <= alpha)
    {
        float pdf0 = EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(0), *sample_data);

        auto mtl1_sample_data = *sample_data;
        auto mtl1 = mtl1_sample_data.Material = &mix_material->getSubMaterial(1);
        SampleIncidentLightLookup[(size_t)mtl1->Model](strata, &mtl1_sample_data, seed);
        float pdf1 = mtl1_sample_data.PDF;
        sample_data->IncidentLight = mtl1_sample_data.IncidentLight;

        sample_data->PDF = pdf0 + alpha*(pdf1 - pdf0);
    }
    else
    {
        auto mtl0_sample_data = *sample_data;
        auto mtl0 = mtl0_sample_data.Material = &mix_material->getSubMaterial(0);
        SampleIncidentLightLookup[(size_t)mtl0->Model](strata, &mtl0_sample_data, seed);
        float pdf0 = mtl0_sample_data.PDF;
        sample_data->IncidentLight = mtl0_sample_data.IncidentLight;

        float pdf1 = EvaluateTableFunctionForMaterial(PDFLookup, &mix_material->getSubMaterial(1), *sample_data);
        
        sample_data->PDF = pdf0 + alpha*(pdf1 - pdf0);
    }
}

#ifndef __CUDACC__
void MixMaterialSetup(RTMaterial* material)
{
    auto* mix_material = static_cast<RTMixMaterial*>(material);
    auto& mat0 = mix_material->getSubMaterial(0);
    MaterialSetupLookup[(size_t)mat0.Model](&mat0);
    auto& mat1 = mix_material->getSubMaterial(1);
    MaterialSetupLookup[(size_t)mat1.Model](&mat1);
}
#endif

SPLIT_EXPORT_CUDA bool MixIsMirrorMaterial(const SampleData& sample_data)
{
    // NOTE: Not really correct for mixed cases
    auto* mix_material = static_cast<const RTMixMaterial*>(sample_data.Material);
    bool mirror0 = EvaluateFunctionForMaterial(MirrorCheck, &mix_material->getSubMaterial(0), sample_data);
    bool mirror1 = EvaluateFunctionForMaterial(MirrorCheck, &mix_material->getSubMaterial(1), sample_data);

    return mirror0 && mirror1;
}

SPLIT_EXPORT_CUDA bool StochasticRotatorIsMirrorMaterial(const SampleData& sample_data)
{
    // NOTE: Not really correct for mixed cases
    auto* rotator_material = static_cast<const RTStochasticRotatorMaterial*>(sample_data.Material);
    return EvaluateFunctionForMaterial(MirrorCheck, &rotator_material->getSubMaterial(), sample_data);
}

#ifndef __CUDACC__
void MixCache(const SampleData& sample_data, unsigned& seed)
{
    auto* mix_material = static_cast<const RTMixMaterial*>(sample_data.Material);
    auto* mix_cache = reinterpret_cast<MixCacheHeader*>(sample_data.ScratchMemory);

    mix_cache->Alpha = SampleRed(mix_material->MixTexture, sample_data.TexCoord);

    TGE_ASSERT(MaterialCacheLookup[(size_t)mix_material->getSubMaterial(0).Model] == nullptr, "subcaches not supported!");
    TGE_ASSERT(MaterialCacheLookup[(size_t)mix_material->getSubMaterial(1).Model] == nullptr, "subcaches not supported!");
}
#endif

// Stochastic Rotator
// If you replace the patching function you can use it to generate all sorts of BRDF out of
// distribution of geometric features
SPLIT_EXPORT_CUDA float StochasticRotatorPDF(const SampleData& sample_data)
{
    auto* rot_material = static_cast<const RTStochasticRotatorMaterial*>(sample_data.Material);
    auto* cache = reinterpret_cast<StochasticRotatorHeader*>(sample_data.ScratchMemory);
    auto mtl_sample_data = sample_data;
    auto sub_material = mtl_sample_data.Material = &rot_material->getSubMaterial();

    float pdf = 0.0f;
    float weight = 1.0f/rot_material->SampleCount;

    for(uint32_t samp_idx = 0; samp_idx < rot_material->SampleCount; ++samp_idx)
    {
        mtl_sample_data.Tangent = Transform(cache->Rotation[samp_idx], sample_data.Tangent);
        mtl_sample_data.Binormal = Transform(cache->Rotation[samp_idx], sample_data.Binormal);
        mtl_sample_data.Normal = Transform(cache->Rotation[samp_idx], sample_data.Normal);
     
        if(Dot(mtl_sample_data.Normal, sample_data.Normal) < 0.0f)
            continue;
        
        pdf += weight*PDFLookup[(size_t)sub_material->Model](mtl_sample_data);
    }
    return pdf;
}

SPLIT_EXPORT_CUDA Spectrum StochasticRotatorBRDF(const SampleData& sample_data)
{
    auto* rot_material = static_cast<const RTStochasticRotatorMaterial*>(sample_data.Material);
    auto* cache = reinterpret_cast<StochasticRotatorHeader*>(sample_data.ScratchMemory);
    auto mtl_sample_data = sample_data;
    auto sub_material = mtl_sample_data.Material = &rot_material->getSubMaterial();

    Spectrum contrib{};
    float weight = 1.0f/rot_material->SampleCount;

    for(uint32_t samp_idx = 0; samp_idx < rot_material->SampleCount; ++samp_idx)
    {
        mtl_sample_data.Tangent = Transform(cache->Rotation[samp_idx], sample_data.Tangent);
        mtl_sample_data.Binormal = Transform(cache->Rotation[samp_idx], sample_data.Binormal);
        mtl_sample_data.Normal = Transform(cache->Rotation[samp_idx], sample_data.Normal);

        if(Dot(mtl_sample_data.Normal, sample_data.Normal) < 0.0f)
            continue;

        contrib += weight*TransmittanceLookup[(size_t)sub_material->Model](mtl_sample_data);
    }
    return contrib;
}

SPLIT_EXPORT_CUDA void StochasticRotatorSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
    auto* rot_material = static_cast<const RTStochasticRotatorMaterial*>(sample_data->Material);
    auto* cache = reinterpret_cast<StochasticRotatorHeader*>(sample_data->ScratchMemory);
    auto mtl_sample_data = *sample_data;
    auto sub_material = mtl_sample_data.Material = &rot_material->getSubMaterial();

    float weight = 1.0f/rot_material->SampleCount;

    uint32_t chosen_idx = FastRandRange(0, rot_material->SampleCount, seed);
    mtl_sample_data.Tangent = Transform(cache->Rotation[chosen_idx], sample_data->Tangent);
    mtl_sample_data.Binormal = Transform(cache->Rotation[chosen_idx], sample_data->Binormal);
    mtl_sample_data.Normal = Transform(cache->Rotation[chosen_idx], sample_data->Normal);
    SampleIncidentLightLookup[(size_t)sub_material->Model](strata, &mtl_sample_data, seed);

    if(Dot(mtl_sample_data.Normal, sample_data->Normal) < 0.0f)
    {
        sample_data->PDF = 0.0f;
        return;
    }

    auto pdf = weight*mtl_sample_data.PDF;
    sample_data->IncidentLight = mtl_sample_data.IncidentLight;

    for(uint32_t samp_idx = 0; samp_idx < rot_material->SampleCount; ++samp_idx)
    {
        if(samp_idx == chosen_idx)
            continue;

        mtl_sample_data.Tangent = Transform(cache->Rotation[samp_idx], sample_data->Tangent);
        mtl_sample_data.Binormal = Transform(cache->Rotation[samp_idx], sample_data->Binormal);
        mtl_sample_data.Normal = Transform(cache->Rotation[samp_idx], sample_data->Normal);

        if(Dot(mtl_sample_data.Normal, sample_data->Normal) < 0.0f)
            continue;

        pdf += weight*PDFLookup[(size_t)sub_material->Model](mtl_sample_data);
    }

    sample_data->PDF = pdf;
}

#ifndef __CUDACC__
void StochasticRotatorSetup(RTMaterial* material)
{
    auto* rot_material = static_cast<RTStochasticRotatorMaterial*>(material);
    auto& mat = rot_material->getSubMaterial();
    MaterialSetupLookup[(size_t)mat.Model](&mat);
}

void StochasticRotatorCache(const SampleData& sample_data, unsigned& seed)
{
    float x_ray = FastFloatRand(seed);
	float y_ray = FastFloatRand(seed);

    auto* rot_material = static_cast<const RTStochasticRotatorMaterial*>(sample_data.Material);
    auto* cache = reinterpret_cast<StochasticRotatorHeader*>(sample_data.ScratchMemory);

    // Because basically we are going to multiply by the same pdf and then divide, it doesn't matter, so just generate some direction
    float geom_factor_x_2 = SpecularPowerToGeometryFactorSq(rot_material->StandardDeviation.x);
	float geom_factor_y_2 = SpecularPowerToGeometryFactorSq(rot_material->StandardDeviation.y);
    float geom_factor_x = sqrtf(geom_factor_x_2);
	float geom_factor_y = sqrtf(geom_factor_y_2);

    for(uint32_t samp_idx = 0; samp_idx < rot_material->SampleCount; ++samp_idx)
    {
        auto micro_norm = GGXSampleMicrofacetAnisotropicDistribution(geom_factor_x, geom_factor_y, x_ray, y_ray);
	    NormalizeSelf(&micro_norm);

        cache->Rotation[samp_idx] = ConservativeRotationBetweenVectorQuaternion(micro_norm, Vector3{ 0.0f, 0.0f, 1.0f });
    }
}
#endif


//SpatiallyVaryingEmissive
SPLIT_EXPORT_CUDA Spectrum SpatiallyVaryingEmit(const SampleData& sample_data)
{
    auto* material = reinterpret_cast<const RTSpatiallyVaryingEmitter*>(sample_data.Material);
    return material->EmitFunction(sample_data);
}

//BTF
SPLIT_EXPORT_CUDA Spectrum BTFTransmittance(const SampleData& sample_data)
{
    auto* material = reinterpret_cast<const RTBTF*>(sample_data.Material);
    Tempest::Matrix3 basis(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);

	auto inc_light = basis.transformRotationInverse(sample_data.IncidentLight),
		 out_light = basis.transformRotationInverse(sample_data.OutgoingLight);
    return BTFSampleSpectrum(material->BTFData, inc_light, out_light, sample_data.TexCoord);
}
#endif
}

using namespace MODEL_NAMESPACE;
}

#endif // _TEMPEST_ILLUM_MODELS_HH_

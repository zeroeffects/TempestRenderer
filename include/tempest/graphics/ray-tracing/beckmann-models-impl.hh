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

#ifndef _TEMPEST_BECKMANN_MODELS_HH_
#define _TEMPEST_BECKMANN_MODELS_HH_

#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/graphics/ray-tracing/microfacet-common.hh"
#include "tempest/math/functions.hh"

namespace Tempest
{
inline EXPORT_CUDA float G1_Beckmann(float geom_factor_2, float cos_vec_micronorm, float cos_vec_norm)
{
    float ratio = cos_vec_micronorm/cos_vec_norm;
    if(ratio <= 0.0f)
        return 0.0f;

    float cos_vec_norm_2 = cos_vec_norm*cos_vec_norm;
    float tan_2 = (1.0f - cos_vec_norm_2)/cos_vec_norm_2;

    float a_2 = 1.0f/(geom_factor_2*tan_2);
    float a = sqrtf(a_2);

    /*
    float masking_beckmann = 2.0f/(1.0f + erf(a) + expf(-a_2)/(a*Tempest::SqrtPi));
    /*/
    float masking_beckmann = a < 1.6f ? (3.535f*a + 2.181f*a_2)/(1.0f + 2.276f*a + 2.577f*a_2) : 1.0f;
    //*/

    return masking_beckmann;
}

inline EXPORT_CUDA float BeckmannMicrofacetDistribution(float geom_factor_2, float cos_norm_half)
{
    if(cos_norm_half <= 0.0f)
        return 0.0f;

    float cos_norm_half_2 = cos_norm_half*cos_norm_half;
    float cos_norm_half_4 = cos_norm_half_2*cos_norm_half_2;
    float tan_norm_half_2 = (1.0f - cos_norm_half_2)/cos_norm_half_2;

    float distro = expf(-tan_norm_half_2/geom_factor_2)/(Tempest::MathPi*geom_factor_2*cos_norm_half_4);
    
    return distro;
}

inline EXPORT_CUDA float BeckmannMicrofacetDistributionAnisotropic(float geom_factor_x, float geom_factor_y, float cos_norm_half, float cos_tan_micro, float sin_tan_micro)
{
    float cos_norm_half_2 = cos_norm_half*cos_norm_half;
    float cos_norm_half_4 = cos_norm_half_2*cos_norm_half_2;
    float tan_norm_half_2 = (1.0f - cos_norm_half_2)/cos_norm_half_2;

    TGE_ASSERT(cos_tan_micro*cos_tan_micro + sin_tan_micro*sin_tan_micro > 0.99f, "Bad sin cos");

    float ratio1 = cos_tan_micro/geom_factor_x;
    float ratio2 = sin_tan_micro/geom_factor_y;

    float aniso_term = (ratio1*ratio1 + ratio2*ratio2);

    float distro = expf(-tan_norm_half_2*aniso_term)/(Tempest::MathPi*geom_factor_x*geom_factor_y*cos_norm_half_4);

    return distro;
}

namespace Cpp
{
float BeckmannSpecularBRDF(const SampleData& sample_data, const MicrofacetAngles& angles);
Spectrum BeckmannMicrofacetBRDF(const SampleData& sample_data);
Spectrum BeckmannMicrofacetConductorBRDF(const SampleData& sample_data);
Spectrum BeckmannMicrofacetDielectricBRDF(const SampleData& sample_data);
Spectrum BeckmannMicrofacetAnisotropicBRDF(const SampleData& sample_data);
Spectrum BeckmannMicrofacetAnisotropicConductorBRDF(const SampleData& sample_data);
Spectrum BeckmannMicrofacetAnisotropicDielectricBRDF(const SampleData& sample_data);
float BeckmannMicrofacetPDF(const SampleData& sample_data);
void BeckmannMicrofacetSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed);
float BeckmannMicrofacetAnisotropicPDF(const SampleData& sample_data);
void BeckmannMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed);
}
#if defined(__CUDACC__) && !defined(ILLUMINATION_MODEL_STATIC_IMPLEMENTATION)
namespace Cuda
{
__device__ float BeckmannSpecularBRDF(const SampleData& sample_data, const MicrofacetAngles& angles);
__device__ Spectrum BeckmannMicrofacetBRDF(const SampleData& sample_data);
__device__ Spectrum BeckmannMicrofacetConductorBRDF(const SampleData& sample_data);
__device__ Spectrum BeckmannMicrofacetDielectricBRDF(const SampleData& sample_data);
__device__ Spectrum BeckmannMicrofacetAnisotropicBRDF(const SampleData& sample_data);
__device__ Spectrum BeckmannMicrofacetAnisotropicConductorBRDF(const SampleData& sample_data);
__device__ Spectrum BeckmannMicrofacetAnisotropicDielectricBRDF(const SampleData& sample_data);
__device__ float BeckmannMicrofacetPDF(const SampleData& sample_data);
__device__ void BeckmannMicrofacetSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed);
__device__ float BeckmannMicrofacetAnisotropicPDF(const SampleData& sample_data);
__device__ void BeckmannMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed);
}
#endif

namespace MODEL_NAMESPACE
{
#ifdef ILLUMINATION_MODEL_IMPLEMENTATION
SPLIT_EXPORT_CUDA float BeckmannSpecularBRDF(const SampleData& sample_data, const MicrofacetAngles& angles)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);
    float geom_factor_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
    float shadow_mask = G1_Beckmann(geom_factor_2, angles.CosIncidentMicroNorm, angles.CosIncidentNorm)*
                        G1_Beckmann(geom_factor_2, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);

    float micro_facet_distro = BeckmannMicrofacetDistribution(geom_factor_2, angles.CosMicroNormNorm);

    return shadow_mask*micro_facet_distro/(4.0f*fabsf(angles.CosIncidentNorm)*fabsf(angles.CosOutgoingNorm));
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);

    float fresnel = FresnelSchlick(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);

    auto brdf_beckmann = fresnel*BeckmannSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetConductorBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);

    float fresnel = FresnelConductor(microfacet_material->Fresnel.x, microfacet_material->Fresnel.y, angles.CosIncidentMicroNorm);

    auto brdf_beckmann = fresnel*BeckmannSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetDielectricBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);
    
    float cos_trans = CosTransmittance(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);
    float fresnel = FresnelDielectric(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm, cos_trans);

    auto brdf_beckmann = fresnel*BeckmannSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

inline EXPORT_CUDA float BeckmannSpecularAnisotropicBRDF(const SampleData& sample_data, const MicrofacetAnglesAnisotropic& angles)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

	Vector2 tangent_plane = Vector2{ Dot(sample_data.OutgoingLight, sample_data.Tangent), Dot(sample_data.OutgoingLight, sample_data.Binormal) };
	NormalizeSelf(&tangent_plane);

    float geom_factor_x_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
	float geom_factor_y_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.y);
    float geom_factor_x = sqrtf(geom_factor_x_2);
	float geom_factor_y = sqrtf(geom_factor_y_2);
    float geom_factor_2_out = ComputeRoughnessProjectedOnDirectionSq(geom_factor_x, geom_factor_y,
																     tangent_plane.x, tangent_plane.y);
    float shadow_mask = G1_Beckmann(geom_factor_2_out, angles.CosIncidentMicroNorm, angles.CosIncidentNorm)*
                        G1_Beckmann(geom_factor_2_out, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);

    float micro_facet_distro = BeckmannMicrofacetDistributionAnisotropic(geom_factor_x, geom_factor_y, angles.CosMicroNormNorm, angles.CosMicroNormTangent, angles.CosMicroNormBinorm);
	
    return shadow_mask*micro_facet_distro/(4.0f*fabsf(angles.CosIncidentNorm)*fabsf(angles.CosOutgoingNorm));
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetAnisotropicBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float fresnel = FresnelSchlick(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);

    auto brdf_beckmann = fresnel*BeckmannSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse* (1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetAnisotropicConductorBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float fresnel =  FresnelConductor(microfacet_material->Fresnel.x, microfacet_material->Fresnel.y, angles.CosIncidentMicroNorm);

    auto brdf_beckmann = fresnel*BeckmannSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum BeckmannMicrofacetAnisotropicDielectricBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float cos_trans = CosTransmittance(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);
    float fresnel = FresnelDielectric(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm, cos_trans);

    auto brdf_beckmann = fresnel*BeckmannSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_beckmann)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA float BeckmannMicrofacetPDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    float cos_light_norm = Dot(sample_data.IncidentLight, sample_data.Normal);
    if(cos_light_norm < 0.0f)
        return 0.0f;
    
    auto half_vec = sample_data.IncidentLight + sample_data.OutgoingLight; // or microsurface normal
    NormalizeSelf(&half_vec);

    float cos_half_norm = Dot(half_vec, sample_data.Normal);

    float geom_factor_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
    float norm_pdf = BeckmannMicrofacetDistribution(geom_factor_2, cos_half_norm)*cos_half_norm;
    if(cos_half_norm <= 0.0f || norm_pdf <= 0.0f)
        return 0.0f;

    float jacobian = 1.0f/(4.0f*Dot(half_vec, sample_data.OutgoingLight)); // Because we are generating normals and then performing reflection
    float pdf0 = norm_pdf*jacobian;

    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
    float pdf1 = cos_light_norm * (1.0f/MathPi);
    TGE_ASSERT(std::isfinite(pdf0) && std::isfinite(pdf1), "Bad Blinn-Phong PDF");
    return (pdf0 - pdf1)*ratio + pdf1;
}

SPLIT_EXPORT_CUDA void BeckmannMicrofacetSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data->Material);

    float x_ray = (strata.XStrata + FastFloatRand(seed))/strata.TotalXStrata;
	float y_ray = (strata.YStrata + FastFloatRand(seed))/strata.TotalYStrata;

    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);

    if(FastFloatRand(seed) <= ratio)
    {
        float geom_factor2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
        float geom_factor = sqrtf(geom_factor2);

        float phi = 2.0f*MathPi*y_ray;

        float cos_phi, sin_phi;
        FastSinCos(phi, &sin_phi, &cos_phi);

        float ratio = -logf(x_ray);
		float sqrt_ratio = sqrtf(ratio);

        float rcp_len = 1.0f/sqrtf(ratio*geom_factor2 + 1.0f);
        float common_term = sqrt_ratio*geom_factor*rcp_len;

        auto micro_norm = rcp_len > 1e-6f ? Vector3{-common_term*cos_phi, -common_term*sin_phi, rcp_len} : Vector3{0.0f, 0.0f, 1.0f};
        micro_norm = surface_space.transform(micro_norm);
        TGE_ASSERT(fabsf(Length(micro_norm) - 1.0f) < 1e-3f, "Invalid length");

        sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, micro_norm);
    }
    else
    {
        sample_data->IncidentLight = CosineSampleHemisphere(x_ray, y_ray);
        sample_data->IncidentLight = surface_space.transform(sample_data->IncidentLight);
		NormalizeSelf(&sample_data->IncidentLight);
    }

    sample_data->PDF = BeckmannMicrofacetPDF(*sample_data);
}

SPLIT_EXPORT_CUDA float BeckmannMicrofacetAnisotropicPDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    float cos_light_norm = Dot(sample_data.IncidentLight, sample_data.Normal);
    if(cos_light_norm < 0.0f)
        return 0.0f;
    
    auto half_vec = sample_data.IncidentLight + sample_data.OutgoingLight; // or microsurface normal
    NormalizeSelf(&half_vec);

    float cos_half_norm = Dot(half_vec, sample_data.Normal);

    Vector2 tangent_plane_angle{ Dot(sample_data.Tangent, half_vec), Dot(sample_data.Binormal, half_vec) };
    NormalizeSelf(&tangent_plane_angle);

    float geom_factor_x_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
	float geom_factor_y_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.y);
    float geom_factor_x = sqrtf(geom_factor_x_2);
	float geom_factor_y = sqrtf(geom_factor_y_2);
    float norm_pdf = BeckmannMicrofacetDistributionAnisotropic(geom_factor_x, geom_factor_y, cos_half_norm, tangent_plane_angle.x, tangent_plane_angle.y)*cos_half_norm;
    if(norm_pdf <= 0.0f)
        return 0.0f;

    float jacobian = 1.0f/(4.0f*Dot(half_vec, sample_data.OutgoingLight)); // Because we are generating normals and then performing reflection
    float pdf0 = norm_pdf*jacobian;

    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;
    float pdf1 = cos_light_norm * (1.0f/MathPi);
    TGE_ASSERT(std::isfinite(pdf0) && std::isfinite(pdf1), "Bad Blinn-Phong PDF");
    return (pdf0 - pdf1)*ratio + pdf1;
}

SPLIT_EXPORT_CUDA Vector3 BeckmannSampleMicrofacetAnisotropicDistribution(float geom_factor_x, float geom_factor_y, float r0, float r1)
{
    float phi = 2.0f*MathPi*r1;

    float cos_phi, sin_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);

	float sqrt_ratio = sqrtf(-logf(r0));
	auto x = geom_factor_x*sqrt_ratio*cos_phi;
	auto y = geom_factor_y*sqrt_ratio*sin_phi;

    auto micro_norm = Vector3{-x, -y, 1.0f};
	NormalizeSelf(&micro_norm);
    return micro_norm;
}

SPLIT_EXPORT_CUDA void BeckmannMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data->Material);

    float x_ray = (strata.XStrata + FastFloatRand(seed))/strata.TotalXStrata;
	float y_ray = (strata.YStrata + FastFloatRand(seed))/strata.TotalYStrata;

    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);

    if(FastFloatRand(seed) <= ratio)
    {
        float geom_factor_x_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
		float geom_factor_y_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.y);
        float geom_factor_x = sqrtf(geom_factor_x_2);
		float geom_factor_y = sqrtf(geom_factor_y_2);

        float x_ray = (strata.XStrata + FastFloatRand(seed))/strata.TotalXStrata;
	    float y_ray = (strata.YStrata + FastFloatRand(seed))/strata.TotalYStrata;

        auto micro_norm = BeckmannSampleMicrofacetAnisotropicDistribution(geom_factor_x, geom_factor_y, x_ray, y_ray);
		NormalizeSelf(&micro_norm);
        
		micro_norm = surface_space.transform(micro_norm);
        NormalizeSelf(&micro_norm);
        sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, micro_norm);
    }
    else
    {
        sample_data->IncidentLight = CosineSampleHemisphere(x_ray, y_ray);
        sample_data->IncidentLight = surface_space.transform(sample_data->IncidentLight);
		NormalizeSelf(&sample_data->IncidentLight);
    }

    sample_data->PDF = BeckmannMicrofacetAnisotropicPDF(*sample_data);
}
#endif // ILLUMINATION_MODEL_IMPLEMENTATION
}
}

#endif // _TEMPEST_BECKMANN_MODELS_HH_
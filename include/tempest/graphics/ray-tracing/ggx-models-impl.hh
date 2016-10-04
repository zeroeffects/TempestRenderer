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

#ifndef _TEMPEST_GGX_MODELS_IMPL_HH_
#define _TEMPEST_GGX_MODELS_IMPL_HH_

#include "tempest/graphics/custom-samplers.hh"
#include "tempest/graphics/ray-tracing/microfacet-common.hh"

//#define SGGX_QUATERNION_SLERP

namespace Tempest
{
inline EXPORT_CUDA float ZhaoMicroFlakeNDF(float c0, float c1, float cos_theta)
{
    return expf(-(cos_theta*cos_theta)/c0)*c1;
}

inline EXPORT_CUDA float SGGXMicroFlakeNDF(const Matrix3& smatrix, const Vector3& normal)
{
    auto adj_tempest = smatrix.adjugate();
    float det_tempest = smatrix.determinant();
    float den_tempest = Dot(normal, adj_tempest.transform(normal));

    return 0.5f*powf(fabsf(det_tempest), 1.5f) / (MathPi*den_tempest*den_tempest);
}

inline EXPORT_CUDA float SGGXMicroFlakeNDF(const Vector3& stddev, const Matrix3& basis, const Vector3& normal)
{
    auto inv_cov = basis.transformCovariance(1.0f/(stddev*stddev));
    float det_alt = inv_cov.determinant();
	if(det_alt < 1e-6f)
		return {};
    float den_alt = Dot(normal, inv_cov.transform(normal));
    return 0.5f*sqrtf(det_alt) / (Tempest::MathPi*den_alt*den_alt);
}


inline EXPORT_CUDA Spectrum SGGXMicroFlakeBRDF(float dir_density, const Vector3& micro_norm, const Vector3& sggx_stddev, const Matrix3& basis)
{
    float debug_coef = 2.0f; // It seems that the original microflake also has this. Why it doesn't work straight out of the box is interesting
                             // question.

    float surface_contrib = SGGXMicroFlakeNDF(sggx_stddev, basis, micro_norm)*debug_coef/(4.0f*dir_density);

    return ToSpectrum(surface_contrib);
}

inline EXPORT_CUDA Vector3 ConvertZhaoMicroFlakeToSGGX(float stddev)
{
    const uint32_t integrator_steps = 1024;

    const float c0 = 2.0f*stddev*stddev;
    const float c1 = 1/((powf(2.0f*Tempest::MathPi, 1.5f))*stddev*erf(1.0f/(sqrt(2.0f)*stddev)));

    /*
    float norm_value = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, integrator_steps,
                                                                                               [c0, c1](float t)
                                                                                               {
                                                                                                   float sin_theta, cos_theta;
                                                                                                   Tempest::FastSinCos(t, &sin_theta, &cos_theta);
                                                                                                   return fabsf(cos_theta)*Tempest::ZhaoNDF(c0, c1, cos_theta)*sin_theta;
                                                                                               });
                                                                                               */
    float norm_value = 2.0f*Tempest::MathPi*c1*c0*(1.0f - expf(-1.0f/c0));

    // integral abs(cos(0 .. pi))
    float cos_int = 4.0f;

    float tan_value = cos_int*SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, integrator_steps,
																		[c0, c1](float t)
																		{
																		    float sin_theta, cos_theta;
																		    FastSinCos(t, &sin_theta, &cos_theta);
																		    return sin_theta*Tempest::ZhaoMicroFlakeNDF(c0, c1, cos_theta)*sin_theta;
																		});

    return { norm_value, tan_value, tan_value };
}

inline EXPORT_CUDA float SGGXProjectedArea(const Vector3& stddev, const Matrix3& basis, const Vector3& incident_light)
{
    return Length(basis.transformRotationInverse(incident_light)*stddev);
}

SPLIT_EXPORT_CUDA Spectrum SGGXMicroFlakePseudoVolumeBRDF(const SampleData& sample_data, const Vector3& sggx_stddev, const Matrix3& sggx_basis);

inline EXPORT_CUDA float GGXMicrofacetDistributionAnisotropic(float geom_factor_x, float geom_factor_y, float cos_norm_half, float cos_tan_micro, float sin_tan_micro)
{
    float cos_norm_half_2 = cos_norm_half*cos_norm_half;
    float cos_norm_half_4 = cos_norm_half_2*cos_norm_half_2;
    float tan_norm_half_2 = (1.0f - cos_norm_half_2)/cos_norm_half_2;

    TGE_ASSERT(cos_tan_micro*cos_tan_micro + sin_tan_micro*sin_tan_micro > 0.99f, "Bad sin cos");

    float ratio1 = cos_tan_micro/geom_factor_x;
    float ratio2 = sin_tan_micro/geom_factor_y;

    float aniso_term = (ratio1*ratio1 + ratio2*ratio2);

    float denom_factor = (1.0f + tan_norm_half_2*aniso_term);

    return 1.0f/(MathPi*cos_norm_half_4*geom_factor_x*geom_factor_y*denom_factor*denom_factor);
}

inline EXPORT_CUDA float GGXMicrofacetDistribution(float geom_factor_2, float cos_norm_half)
{
    if(cos_norm_half <= 0.0f)
        return 0.0f;

    float cos_norm_half_2 = cos_norm_half*cos_norm_half;
    float cos_norm_half_4 = cos_norm_half_2*cos_norm_half_2;
    float tan_norm_half_2 = (1.0f - cos_norm_half_2)/cos_norm_half_2;

    float denom_factor = (geom_factor_2 + tan_norm_half_2);

    return geom_factor_2/(MathPi*cos_norm_half_4*denom_factor*denom_factor);
}

inline EXPORT_CUDA float G1_GGX(float geom_factor_2, float cos_vec_micronorm, float cos_vec_norm)
{
    float ratio = cos_vec_micronorm/cos_vec_norm;
    if(ratio <= 0.0f)
        return 0.0f;

    float cos_vec_norm_2 = cos_vec_norm*cos_vec_norm;
    float tan_2 = (1.0f - cos_vec_norm_2)/cos_vec_norm_2;

    return 2.0f/(1.0f + sqrtf(1.0f + geom_factor_2*tan_2));
}

#ifdef ILLUMINATION_MODEL_IMPLEMENTATION
namespace MODEL_NAMESPACE
{
SPLIT_EXPORT_CUDA float GGXSpecularBRDF(const SampleData& sample_data, const MicrofacetAngles& angles)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);
    float geom_factor_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
    float shadow_mask = G1_GGX(geom_factor_2, angles.CosIncidentMicroNorm, angles.CosIncidentNorm)*
                        G1_GGX(geom_factor_2, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);

    float micro_facet_distro = GGXMicrofacetDistribution(geom_factor_2, angles.CosMicroNormNorm);

    return shadow_mask*micro_facet_distro/(4.0f*fabsf(angles.CosIncidentNorm)*fabsf(angles.CosOutgoingNorm));
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);

    float fresnel = FresnelSchlick(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);

    auto brdf_ggx = fresnel*GGXSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetConductorBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);

    float fresnel = FresnelConductor(microfacet_material->Fresnel.x, microfacet_material->Fresnel.y, angles.CosIncidentMicroNorm);

    auto brdf_ggx = fresnel*GGXSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetDielectricBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAngles angles;
    ComputeMicrofacetAngles(sample_data, &angles);
    
    float cos_trans = CosTransmittance(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);
    float fresnel = FresnelDielectric(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm, cos_trans);

    auto brdf_ggx = fresnel*GGXSpecularBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA float GGXMicrofacetPDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    float cos_light_norm = Dot(sample_data.IncidentLight, sample_data.Normal);
    if(cos_light_norm < 0.0f)
        return 0.0f;
    
    auto half_vec = sample_data.IncidentLight + sample_data.OutgoingLight; // or microsurface normal
    NormalizeSelf(&half_vec);

    float cos_half_norm = Dot(half_vec, sample_data.Normal);

    float geom_factor_2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
    float norm_pdf = GGXMicrofacetDistribution(geom_factor_2, cos_half_norm)*cos_half_norm;
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

SPLIT_EXPORT_CUDA void GGXMicrofacetSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& mirand)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data->Material);

    float x_ray = (strata.XStrata + FastFloatRand(mirand))/strata.TotalXStrata;
	float y_ray = (strata.YStrata + FastFloatRand(mirand))/strata.TotalYStrata;

    float max_value = MaxValue(microfacet_material->Diffuse + microfacet_material->Specular);
    float ratio = max_value ? MaxValue(microfacet_material->Specular) / max_value : 0.0f;

    Tempest::Matrix3 surface_space;
    surface_space.makeBasis(sample_data->Normal);

    if(FastFloatRand(mirand) <= ratio)
    {
        float geom_factor2 = SpecularPowerToGeometryFactorSq(microfacet_material->SpecularPower.x);
        float geom_factor = sqrtf(geom_factor2);

        float phi = 2.0f*MathPi*y_ray;

        float cos_phi, sin_phi;
        FastSinCos(phi, &sin_phi, &cos_phi);

        float ratio = x_ray/(1.0f - x_ray);
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

    sample_data->PDF = GGXMicrofacetPDF(*sample_data);
}

inline EXPORT_CUDA float GGXSpecularAnisotropicBRDF(const SampleData& sample_data, const MicrofacetAnglesAnisotropic& angles)
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
    float shadow_mask = G1_GGX(geom_factor_2_out, angles.CosIncidentMicroNorm, angles.CosIncidentNorm)*
                        G1_GGX(geom_factor_2_out, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);

    float micro_facet_distro = GGXMicrofacetDistributionAnisotropic(geom_factor_x, geom_factor_y, angles.CosMicroNormNorm, angles.CosMicroNormTangent, angles.CosMicroNormBinorm);
	
    return shadow_mask*micro_facet_distro/(4.0f*fabsf(angles.CosIncidentNorm)*fabsf(angles.CosOutgoingNorm));
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetAnisotropicBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float fresnel = FresnelSchlick(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);

    auto brdf_ggx = fresnel*GGXSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse* (1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetAnisotropicConductorBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float fresnel =  FresnelConductor(microfacet_material->Fresnel.x, microfacet_material->Fresnel.y, angles.CosIncidentMicroNorm);

    auto brdf_ggx = fresnel*GGXSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA Spectrum GGXMicrofacetAnisotropicDielectricBRDF(const SampleData& sample_data)
{
    auto* microfacet_material = static_cast<const RTMicrofacetMaterial*>(sample_data.Material);

    MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(sample_data, &angles);

    float cos_trans = CosTransmittance(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm);
    float fresnel = FresnelDielectric(microfacet_material->Fresnel.x, angles.CosIncidentMicroNorm, cos_trans);

    auto brdf_ggx = fresnel*GGXSpecularAnisotropicBRDF(sample_data, angles);

    return (microfacet_material->Diffuse*(1.0f/MathPi) + microfacet_material->Specular*brdf_ggx)*angles.CosIncidentNorm;
}

SPLIT_EXPORT_CUDA float GGXMicrofacetAnisotropicPDF(const SampleData& sample_data)
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
    float norm_pdf = GGXMicrofacetDistributionAnisotropic(geom_factor_x, geom_factor_y, cos_half_norm, tangent_plane_angle.x, tangent_plane_angle.y)*cos_half_norm;
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

SPLIT_EXPORT_CUDA Vector3 GGXSampleMicrofacetAnisotropicDistribution(float geom_factor_x, float geom_factor_y, float r0, float r1)
{
    float phi = 2.0f*MathPi*r1;

    float cos_phi, sin_phi;
    FastSinCos(phi, &sin_phi, &cos_phi);

	float sqrt_ratio = sqrtf(r0/(1.0f - r0));
	auto x = geom_factor_x*sqrt_ratio*cos_phi;
	auto y = geom_factor_y*sqrt_ratio*sin_phi;

    auto micro_norm = Vector3{-x, -y, 1.0f};
	NormalizeSelf(&micro_norm);
    return micro_norm;
}

SPLIT_EXPORT_CUDA void GGXMicrofacetAnisotropicSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
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

        auto micro_norm = GGXSampleMicrofacetAnisotropicDistribution(geom_factor_x, geom_factor_y, x_ray, y_ray);
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

    sample_data->PDF = GGXMicrofacetAnisotropicPDF(*sample_data);
}

//SGGX
#ifndef __CUDACC__
void SGGXMicroFlakeSetup(RTMaterial* material)
{
    auto* sggx_material = reinterpret_cast<RTSGGXMicroFlakeMaterial*>(material);
    sggx_material->VolumeModel = VolumeIlluminationModel::SGGXMicroFlake;
}
#endif

inline EXPORT_CUDA float SGGXMicroFlakePDF(float dir_density, const Vector3& sggx_stddev, const Matrix3& basis, const Vector3& micro_norm)
{
	float ndf = SGGXMicroFlakeNDF(sggx_stddev, basis, micro_norm);
    float proj_area = /* Dot(micro_norm, sample_data.IncidentLight))/Dot(micro_norm, sample_data.OutgoingLight) identity because jacobian*/
					  1.0f/(4.0f*dir_density);

    float debug_coef = 2.0f;

    return debug_coef*ndf*proj_area;
}

SPLIT_EXPORT_CUDA float SGGXMicroFlakePDF(const SampleData& sample_data)
{
    auto micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);
    auto* sggx_material = reinterpret_cast<const RTSGGXMicroFlakeMaterial*>(sample_data.Material);
    Tempest::Matrix3 basis(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);
    return SGGXMicroFlakePDF(sample_data.DirectionalDensity, sggx_material->SGGXStandardDeviation, basis, micro_norm);
}

SPLIT_EXPORT_CUDA Spectrum SGGXMicroFlakeBRDF(const SampleData& sample_data)
{
    // TODO: Not really how it's implemented
    auto micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);
    auto* sggx_material = reinterpret_cast<const RTSGGXMicroFlakeMaterial*>(sample_data.Material);
    Tempest::Matrix3 basis(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);

    return SGGXMicroFlakeBRDF(sample_data.DirectionalDensity, micro_norm, sggx_material->SGGXStandardDeviation, basis);
}

SPLIT_EXPORT_CUDA Spectrum SGGXMicroFlakeSurfaceBRDF(const SampleData& sample_data)
{
    auto sggx_surface_material = reinterpret_cast<const RTSGGXSurface*>(sample_data.Material); 

//    float hack_energy_loss = 2.0f; // HACK: the 2x factor is massive cheat to reduce energy loss

    float dot_inc_norm = Dot(sample_data.Normal, sample_data.IncidentLight);
    float dot_out_norm = Dot(sample_data.Normal, sample_data.OutgoingLight);
    if(dot_inc_norm < 0.0f || dot_out_norm < 0.0f)
        return {};

	auto albedo_strength = sggx_surface_material->Diffuse;
	if(sggx_surface_material->DiffuseMap)
	{
		albedo_strength = SampleSpectrum(sggx_surface_material->DiffuseMap, sample_data.TexCoord);
	}

	Tempest::Spectrum result = albedo_strength*dot_inc_norm*(1.0f / MathPi);
	
	auto stddev_v2 = sggx_surface_material->StandardDeviation;
	if(sggx_surface_material->StandardDeviationMap)
	{
		stddev_v2 = SampleRG(sggx_surface_material->StandardDeviationMap, sample_data.TexCoord);
	}

    Tempest::Vector3 stddev{ stddev_v2.x, stddev_v2.y, 1.0f };

	if(stddev.x >= 1e-6f && stddev.y >= 1e-6f && stddev.z >= 1e-6f && sample_data.DirectionalDensity >= 1e-6f)
	{
		auto micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);
	
		Tempest::Matrix3 surface_orientation(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);

		Quaternion quat;
        quat.V4 = sggx_surface_material->SGGXBasis;
		if(sggx_surface_material->BasisMap)
		{
        #ifdef SGGX_QUATERNION_SLERP
			quat = SampleQuaternionSlerp(sggx_surface_material->BasisMap, sggx_surface_material->BasisMapWidth, sggx_surface_material->BasisMapHeight, sample_data.TexCoord);
        #else
            quat.V4 = SampleRGBA(sggx_surface_material->BasisMap, sample_data.TexCoord);
        #endif
        }

		auto basis = surface_orientation*Tempest::ToMatrix3(Normalize(quat));

		float cos_micro = Dot(micro_norm, sample_data.IncidentLight);
		//float fresnel = Clamp(FresnelSchlick(sggx_surface_material->Fresnel.x, cos_micro), 0.0f, 1.0f);

		float fresnel;
		float refr_idx = sggx_surface_material->Fresnel.x;
		if(refr_idx == 1.0f)
		{
			fresnel = 1.0f;
		}
		else
		{
			float cos_trans = CosTransmittance(refr_idx, cos_micro);
			fresnel = FresnelDielectric(refr_idx, cos_micro, cos_trans);
			fresnel = Clamp(fresnel, 0.0f, 1.0f);
		}

		auto spec_strength = sggx_surface_material->Specular;
		if(sggx_surface_material->SpecularMap)
		{
			spec_strength = SampleSpectrum(sggx_surface_material->SpecularMap, sample_data.TexCoord);
		}

        float out_density = sample_data.DirectionalDensity; //SGGXProjectedArea(stddev, basis, sample_data.OutgoingLight);
        float incoming_density = SGGXProjectedArea(stddev, basis, sample_data.IncidentLight);

        float contrib = 1.0f/(out_density + incoming_density*dot_out_norm/dot_inc_norm);

        result += contrib*spec_strength*SGGXMicroFlakeNDF(stddev, basis, micro_norm)*fresnel*2.0f/4.0f;
	}
	return result;
}

inline EXPORT_CUDA void SGGXMicroFlakeSampleIncidentLight(const Vector3& out_light, float dir_density, const Vector3& sggx_stddev, const Matrix3& basis, unsigned& seed, Tempest::Vector3* inc_light)
{
	auto uniform_sample = UniformSampleSphere(FastFloatRand(seed), FastFloatRand(seed));

    auto S_matrix = basis.transformCovariance(sggx_stddev*sggx_stddev);

    Matrix3 out_basis;
    out_basis.makeBasis(out_light);

    S_matrix = out_basis*S_matrix*out_basis.transpose();

    // Basically we are going towards ellipsoid space
    auto visible_norm = Normalize(S_matrix.choleskyDecomposition().transformRotationInverse(uniform_sample));
    visible_norm = out_basis.transform(visible_norm);

	*inc_light = Reflect(out_light, visible_norm);
}

SPLIT_EXPORT_CUDA void SGGXMicroFlakeSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
    auto* sggx_material = reinterpret_cast<const RTSGGXMicroFlakeMaterial*>(sample_data->Material);

    Tempest::Matrix3 basis(sample_data->Tangent, sample_data->Binormal, sample_data->Normal);
	SGGXMicroFlakeSampleIncidentLight(sample_data->OutgoingLight, sample_data->DirectionalDensity, sggx_material->SGGXStandardDeviation, basis, seed, &sample_data->IncidentLight);
    auto micro_norm = Normalize(sample_data->IncidentLight + sample_data->OutgoingLight);
    sample_data->PDF = SGGXMicroFlakePDF(sample_data->DirectionalDensity, sggx_material->SGGXStandardDeviation, basis, micro_norm);
}

SPLIT_EXPORT_CUDA float SGGXMicroFlakeDensity(const SampleData& sample_data)
{
    auto sggx_material = static_cast<const RTSGGXMicroFlakeMaterial*>(sample_data.Material);
    Matrix3 basis(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);
    return SGGXProjectedArea(sggx_material->SGGXStandardDeviation, basis, sample_data.IncidentLight);
}

SPLIT_EXPORT_CUDA float SGGXMicroFlakeSurfaceDensity(const SampleData& sample_data)
{
    auto sggx_material = static_cast<const RTSGGXSurface*>(sample_data.Material);
    auto stddev_v2 = sggx_material->StandardDeviation;
	if(sggx_material->StandardDeviationMap)
	{
		stddev_v2 = SampleRG(sggx_material->StandardDeviationMap, sample_data.TexCoord);
	}

    Tempest::Vector3 stddev{ stddev_v2.x, stddev_v2.y, 1.0f };

	Tempest::Quaternion quat;
    quat.V4 = sggx_material->SGGXBasis;
	if(sggx_material->BasisMap)
	{
	#ifdef SGGX_QUATERNION_SLERP
        quat = SampleQuaternionSlerp(sggx_material->BasisMap, sggx_material->BasisMapWidth, sggx_material->BasisMapHeight, sample_data.TexCoord);
    #else
        quat.V4 = SampleRGBA(sggx_material->BasisMap, sample_data.TexCoord);
    #endif
	}

    Tempest::Matrix3 surface_orientation(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);
    Tempest::Matrix3 basis = surface_orientation*ToMatrix3(quat);
    return SGGXProjectedArea(stddev, basis, sample_data.IncidentLight);
}

SPLIT_EXPORT_CUDA void SGGXSurfaceCache(const SampleData& sample_data, unsigned& seed)
{
    sample_data.DirectionalDensity = SGGXMicroFlakeSurfaceDensity(sample_data);
}

SPLIT_EXPORT_CUDA Spectrum SGGXMicroFlakePseudoVolumeBRDF(const SampleData& sample_data)
{
	auto sggx_material = reinterpret_cast<const RTSGGXSurface*>(sample_data.Material);

    float dot_inc_norm = Dot(sample_data.Normal, sample_data.IncidentLight);
	float dot_out_norm = Dot(sample_data.Normal, sample_data.OutgoingLight);

    if(dot_out_norm < 0.0f || dot_inc_norm < 0.0f)
    {
        return {};
    }
    
	auto stddev_v2 = sggx_material->StandardDeviation;
	if(sggx_material->StandardDeviationMap)
	{
		stddev_v2 = SampleRG(sggx_material->StandardDeviationMap, sample_data.TexCoord);
	}

    Tempest::Vector3 sggx_stddev{ stddev_v2.x, stddev_v2.y, 1.0f };

    if(sggx_stddev.x < 1e-6f || sggx_stddev.y < 1e-6f)
        return {};
    
    Tempest::Matrix3 surface_orientation(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);

    Quaternion quat;
    quat.V4 = sggx_material->SGGXBasis;
	if(sggx_material->BasisMap)
	{
	#ifdef SGGX_QUATERNION_SLERP
        quat = SampleQuaternionSlerp(sggx_material->BasisMap, sggx_material->BasisMapWidth, sggx_material->BasisMapHeight, sample_data.TexCoord);
    #else
        quat.V4 = SampleRGBA(sggx_material->BasisMap, sample_data.TexCoord);
    #endif
	}

	Tempest::Matrix3 sggx_basis = surface_orientation*Tempest::ToMatrix3(quat);

    unsigned seed = ((uint32_t)(sample_data.TexCoord.y * USHRT_MAX) << 16) + (uint32_t)(sample_data.TexCoord.x * USHRT_MAX); // TODO: get it from somewhere else
    FastFloatRand(seed);

    const uint32_t max_simulate_scattering = sggx_material->Depth;
    const uint32_t sample_count = sggx_material->SampleCount;

    float start_dir_density = sample_data.DirectionalDensity;
    float incoming_density = SGGXProjectedArea(sggx_stddev, sggx_basis, sample_data.IncidentLight);

//#define USE_PSEUDO_VOLUME_FRESNEL

	auto micro_norm = Normalize(sample_data.OutgoingLight + sample_data.IncidentLight);

    float contrib = sample_count/(start_dir_density + incoming_density*dot_out_norm/dot_inc_norm);

#ifdef USE_PSEUDO_VOLUME_FRESNEL
    float refr_idx = sample_data.Material->Fresnel.x;
    if(refr_idx != 1.0f)
    {
        float cos_micro = Dot(sample_data.IncidentLight, micro_norm);
        float cos_trans = CosTransmittance(refr_idx, cos_micro);
        float fresnel = FresnelDielectric(refr_idx, cos_micro, cos_trans);
        contrib *= Clamp(fresnel, 0.0f, 1.0f);
    }
#endif

    Spectrum radiance = contrib*sggx_material->Specular*SGGXMicroFlakeNDF(sggx_stddev, sggx_basis, micro_norm)*2.0f/4.0f;

    for(uint32_t sample_idx = 0; sample_idx < sample_count; ++sample_idx)
    {
        Tempest::Spectrum throughput = sggx_material->Specular;
        auto out_light = sample_data.OutgoingLight;
        auto dir_density = start_dir_density;
        auto inc_light = sample_data.IncidentLight;

		float t = FastFloatRand(seed);
        float log_step = -logf(t);
    
        float step_size = log_step/dir_density;

        float penetration_depth = step_size*dot_out_norm;

        for(uint32_t simulation_iter = 0; throughput > 1e-3f && simulation_iter < max_simulate_scattering; ++simulation_iter)
        {
            SGGXMicroFlakeSampleIncidentLight(out_light, dir_density, sggx_stddev, sggx_basis,
                                              seed, &inc_light);

        #ifdef USE_PSEUDO_VOLUME_FRESNEL
            if(refr_idx != 1.0f)
            {
                auto micro_norm = Normalize(out_light + inc_light);
                float cos_micro = Dot(inc_light, micro_norm);
                float cos_trans = CosTransmittance(refr_idx, cos_micro);
                float fresnel = FresnelDielectric(refr_idx, cos_micro, cos_trans);
                throughput *= Clamp(fresnel, 0.0f, 1.0f);
            }
        #endif

            dir_density = SGGXProjectedArea(sggx_stddev, sggx_basis, inc_light);
            out_light = -inc_light;
        
			float t = FastFloatRand(seed);
            float log_step = -logf(t);
    
            float step_size = log_step/dir_density;

            penetration_depth += step_size*Dot(sample_data.Normal, out_light);

            if(penetration_depth < 0.0f)
                break;

            float incoming_distance = penetration_depth / dot_inc_norm;
            float incoming_extinction = expf(-incoming_density*incoming_distance);
            //cur_extinction *= expf(-log_step);
			if(incoming_extinction < 1e-9f)
				break;
			auto micro_norm = Normalize(out_light + sample_data.IncidentLight);

            //float balance = simulation_iter - 1 != max_simulate_scattering ? 0.5f : 1.0f;
            float balance = 1.0f;

			throughput *= balance*sggx_material->Specular;

            float contrib = incoming_extinction;

        #ifdef USE_PSEUDO_VOLUME_FRESNEL
            if(refr_idx != 1.0f)
            {
                float cos_micro = Dot(sample_data.IncidentLight, micro_norm);
                float cos_trans = CosTransmittance(refr_idx, cos_micro);
                float fresnel = FresnelDielectric(refr_idx, cos_micro, cos_trans);
                contrib *= Clamp(fresnel, 0.0f, 1.0f);
            }
        #endif

            radiance += /* cur_extinction* */ contrib*throughput*SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, sggx_basis);
		}
    }
    return radiance/static_cast<float>(sample_count) + sggx_material->Diffuse*dot_inc_norm*(1.0f / MathPi);
}

SPLIT_EXPORT_CUDA float SGGXMicroFlakeSurfacePDF(const SampleData& sample_data)
{
	auto* sggx_material = reinterpret_cast<const RTSGGXSurface*>(sample_data.Material);
	if(Tempest::Dot(sample_data.IncidentLight, sample_data.Normal) < 0.0f ||
       Tempest::Dot(sample_data.OutgoingLight, sample_data.Normal) < 0.0f)
        return 0.0f;
    
    Vector3 micro_norm = Normalize(sample_data.IncidentLight + sample_data.OutgoingLight);

	auto stddev_v2 = sggx_material->StandardDeviation;
	if(sggx_material->StandardDeviationMap)
	{
		stddev_v2 = SampleRG(sggx_material->StandardDeviationMap, sample_data.TexCoord);
	}

    Tempest::Vector3 sggx_stddev{ stddev_v2.x, stddev_v2.y, 1.0f };

	Quaternion quat;
    quat.V4 = sggx_material->SGGXBasis;
	if(sggx_material->BasisMap)
	{
	#ifdef SGGX_QUATERNION_SLERP
        quat = SampleQuaternionSlerp(sggx_material->BasisMap, sggx_material->BasisMapWidth, sggx_material->BasisMapHeight, sample_data.TexCoord);
    #else
        quat.V4 = SampleRGBA(sggx_material->BasisMap, sample_data.TexCoord);
    #endif
	}

    Tempest::Matrix3 surface_orientation(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);
	Tempest::Matrix3 sggx_basis = surface_orientation*Tempest::ToMatrix3(quat);

	auto pdf = 2.0f*SGGXMicroFlakePDF(sample_data.DirectionalDensity, sggx_stddev, sggx_basis, micro_norm);
    TGE_ASSERT(std::isfinite(pdf), "Invalid pdf");
    return pdf;
}

SPLIT_EXPORT_CUDA void SGGXMicroFlakeSurfaceSampleIncidentLight(const Stratification& strata, SampleData* sample_data, unsigned& seed)
{
	auto sggx_material = reinterpret_cast<const RTSGGXSurface*>(sample_data->Material);

	auto stddev_v2 = sggx_material->StandardDeviation;
	if(sggx_material->StandardDeviationMap)
	{
		stddev_v2 = SampleRG(sggx_material->StandardDeviationMap, sample_data->TexCoord);
	}

    Tempest::Vector3 sggx_stddev{ stddev_v2.x, stddev_v2.y, 1.0f };

	Tempest::Quaternion quat;
    quat.V4 = sggx_material->SGGXBasis;
	if(sggx_material->BasisMap)
	{
	#ifdef SGGX_QUATERNION_SLERP
        quat = SampleQuaternionSlerp(sggx_material->BasisMap, sggx_material->BasisMapWidth, sggx_material->BasisMapHeight, sample_data->TexCoord);
    #else
        quat.V4 = SampleRGBA(sggx_material->BasisMap, sample_data->TexCoord);
    #endif
	}

    Tempest::Matrix3 surface_orientation(sample_data->Tangent, sample_data->Binormal, sample_data->Normal);
    Tempest::Matrix3 basis = surface_orientation*Tempest::ToMatrix3(quat);

    auto uniform_sample = UniformSampleSphere(FastFloatRand(seed), FastFloatRand(seed));

    auto S_matrix = basis.transformCovariance(sggx_stddev*sggx_stddev);

    Matrix3 out_basis;
    out_basis.makeBasis(sample_data->OutgoingLight);

    S_matrix = out_basis*S_matrix*out_basis.transpose();

    // Basically we are going towards ellipsoid space
    auto micro_norm = Normalize(S_matrix.choleskyDecomposition().transformRotationInverse(uniform_sample));
    micro_norm = out_basis.transform(micro_norm);

    // Exploit symmetry
    if(Dot(micro_norm, sample_data->Normal))
        micro_norm = -micro_norm;

    sample_data->IncidentLight = Reflect(sample_data->OutgoingLight, micro_norm);
    sample_data->PDF = Dot(sample_data->IncidentLight, sample_data->Normal) > 0.0f ? 2.0f*SGGXMicroFlakePDF(sample_data->DirectionalDensity, sggx_stddev, basis, micro_norm) : 0.0f;
    TGE_ASSERT(std::isfinite(sample_data->PDF), "Invalid pdf");
}
}
#endif
}

#endif

#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/utils/refractive-indices.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"

#include <memory>

const size_t IntergrationSampleCount = 4096;
const size_t SampleCount = 10000;

inline EXPORT_CUDA void ComputeMicrofacetAnglesAnisotropic(const Tempest::Vector3& inc_light, const Tempest::Vector3& out_light, Tempest::MicrofacetAnglesAnisotropic* angles)
{
    auto half_vec = Normalize(inc_light + out_light); // or microsurface normal

    angles->CosIncidentNorm = inc_light.z;
    angles->CosOutgoingNorm = out_light.z;
    angles->CosIncidentMicroNorm = Dot(inc_light, half_vec);
    angles->CosMicroNormNorm = half_vec.z;

    Tempest::Vector2 tangent_plane_angle{ half_vec.x, half_vec.y };
    NormalizeSelf(&tangent_plane_angle);

	angles->CosMicroNormTangent = tangent_plane_angle.x;
	angles->CosMicroNormBinorm = tangent_plane_angle.y;
}

TGE_TEST("Testing whether different implementations of GGX are working consistently")
{
    Tempest::RTMicrofacetMaterial mtl{};
    mtl.Model = Tempest::IlluminationModel::GGXMicrofacetDielectric;
    mtl.Model = Tempest::IlluminationModel::GGXMicrofacet;
    mtl.Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{ 0.1f, 0.1f, 0.1f });
    mtl.Specular = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
    mtl.SpecularPower.x = 1000.0f;
    mtl.SpecularPower.y = 1000.0f;
	mtl.Fresnel.x = 0.5f;

    unsigned seed = 1;

    std::unique_ptr<uint8_t[]> scratch_memory(new uint8_t[Tempest::ScratchMemoryPerThread]);

    Tempest::SampleData sample_data;
    sample_data.Material = &mtl;
    Tempest::Stratification strata;
    strata.XStrata = 0;
    strata.YStrata = 0;
    strata.TotalXStrata = 1;
    strata.TotalYStrata = 1;
    sample_data.TexCoord = Tempest::Vector2{};
    sample_data.Tangent = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    sample_data.Normal = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
    sample_data.ScratchMemory = scratch_memory.get();

    auto stddev = Tempest::Vector2Sqrt(2.0f/( mtl.SpecularPower + Tempest::ToVector2(2.0f)));

    auto out_light = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

    auto projected_area = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntergrationSampleCount,
                            [&stddev, &out_light](const Tempest::Vector3& norm)
                            {
                                Tempest::Vector2 tangent_plane_angle{ norm.x, norm.y };
                                NormalizeSelf(&tangent_plane_angle);

                                float ndf = Tempest::GGXMicrofacetDistributionAnisotropic(stddev.x, stddev.y, norm.z, tangent_plane_angle.x, tangent_plane_angle.y);

                                return ndf;
                            });

    TGE_CHECK(Tempest::ApproxEqual(projected_area, 1.0f, 1e-1f), "Invalid projected area");

    auto weak_furnace_iso = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntergrationSampleCount,
                            [&stddev, &out_light](const Tempest::Vector3& inc_light)
                            {
                                auto norm = Normalize(out_light + inc_light);

                                Tempest::MicrofacetAnglesAnisotropic angles;
                                ComputeMicrofacetAnglesAnisotropic(inc_light, out_light, &angles);

                                Tempest::Vector2 tangent_plane{ out_light.x, out_light.y };
	                            NormalizeSelf(&tangent_plane);

                                float geom_factor_2 = stddev.x*stddev.x;

                                float geom_term = Tempest::G1_GGX(geom_factor_2, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);
                                float ndf = Tempest::GGXMicrofacetDistribution(stddev.x, angles.CosMicroNormNorm);

                                float sample = geom_term * ndf / (4.0f*angles.CosOutgoingNorm);
	
                                return sample;
                            });

    TGE_CHECK(Tempest::ApproxEqual(weak_furnace_iso, 1.0f, 1e-1f), "Failed the furnace test for isotropic sample");

    auto weak_furnace_aniso = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntergrationSampleCount,
                            [&stddev, &out_light](const Tempest::Vector3& inc_light)
                            {
                                auto norm = Normalize(out_light + inc_light);

                                Tempest::MicrofacetAnglesAnisotropic angles;
                                ComputeMicrofacetAnglesAnisotropic(inc_light, out_light, &angles);

                                Tempest::Vector2 tangent_plane{ out_light.x, out_light.y };
	                            NormalizeSelf(&tangent_plane);

                                float geom_factor_2_out = Tempest::ComputeRoughnessProjectedOnDirectionSq(stddev.x, stddev.y,
																     tangent_plane.x, tangent_plane.y);

                                float geom_term = Tempest::G1_GGX(geom_factor_2_out, angles.CosOutgoingMicroNorm, angles.CosOutgoingNorm);
                                float ndf = Tempest::GGXMicrofacetDistributionAnisotropic(stddev.x, stddev.y, angles.CosMicroNormNorm, angles.CosMicroNormTangent, angles.CosMicroNormBinorm);

                                float sample = geom_term * ndf / (4.0f*angles.CosOutgoingNorm);
	
                                return sample;
                            });

    TGE_CHECK(Tempest::ApproxEqual(weak_furnace_aniso, 1.0f, 1e-1f), "Failed the furnace test for anisotropic sample");

    for(size_t i = 0; i < SampleCount; ++i)
    {
        //*
        float ra = Tempest::FastFloatRand(seed),
              rb = Tempest::FastFloatRand(seed);
        auto out_dir = Tempest::UniformSampleHemisphere(ra, rb);

        sample_data.OutgoingLight = out_dir;
        /*/
        sample_data.OutgoingLight = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
        //*/
        mtl.Model = Tempest::IlluminationModel::GGXMicrofacet;

        auto start_seed = seed;

        Tempest::GGXMicrofacetSampleIncidentLight(strata, &sample_data, seed);
        auto inc_iso = sample_data.IncidentLight;
        auto pdf_iso = sample_data.PDF;

        mtl.Model = Tempest::IlluminationModel::GGXMicrofacetAnisotropic;
        // NOTE: it assumes equal draws from distribution. Might fail if something changes drastically.
        seed = start_seed;

        auto pdf_ani = Tempest::GGXMicrofacetAnisotropicPDF(sample_data);

        TGE_CHECK(Tempest::ApproxEqual(pdf_ani, pdf_iso, 1e-3f), "Invalid PDF");

        auto brdf_iso = Tempest::GGXMicrofacetBRDF(sample_data);
        auto brdf_ani = Tempest::GGXMicrofacetAnisotropicBRDF(sample_data);

        TGE_CHECK(Tempest::ApproxEqual(brdf_iso, brdf_ani, 1e-3f), "Invalid direction");
    }
}
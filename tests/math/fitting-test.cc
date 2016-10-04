#include "tempest/utils/testing.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/fitting.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/image/btf.hh"

const uint32_t CommonSampleCount = 512;
const uint32_t LowRandomSamples = 10;
const uint32_t NDRReduceSphereSamples = CommonSampleCount*CommonSampleCount;
const uint32_t EigenVectorSphereSamples = CommonSampleCount*CommonSampleCount;
const uint32_t EigenVectorPlaneSamples = CommonSampleCount;
const uint32_t ConvertSamples = CommonSampleCount*CommonSampleCount;
const uint32_t DummyBTFLightSamples = 1024;
const uint32_t NDFSamples = 128;
const uint32_t NDFReduceSamples = 4096;
const uint32_t BTFBRDFComparisonSamples = 1024;
const uint32_t IntegratorSamples = 4096;

//#define TEST_SGGX

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

float GGXNDFBRDF(const Tempest::Vector2& stddev, const Tempest::Vector3& inc_light, const Tempest::Vector3& out_light)
{
    Tempest::MicrofacetAnglesAnisotropic angles;
    ComputeMicrofacetAnglesAnisotropic(inc_light, out_light, &angles);

    float micro_facet_distro = Tempest::GGXMicrofacetDistributionAnisotropic(stddev.x, stddev.y, angles.CosMicroNormNorm, angles.CosMicroNormTangent, angles.CosMicroNormBinorm);
	
    return micro_facet_distro; /* /(4.0f*fabsf(angles.CosIncidentNorm)*fabsf(angles.CosOutgoingNorm))) */
}

TGE_TEST("Testing data fitting functions")
{
    Tempest::ThreadPool pool;
    auto thread_id = pool.allocateThreadNumber();

    unsigned seed = 1;
    for(uint32_t i = 0; i < LowRandomSamples; ++i)
    {
        auto norm = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 basis;
        basis.makeBasis(norm);

        auto sggx_stddev = Tempest::Vector3{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), 1.0f };

        {
        Tempest::BTFPtr btf(Tempest::CreateDummyBTF(DummyBTFLightSamples));
        
        Tempest::Vector2 ggx_stddev = { Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };
		Tempest::Vector3 sggx_stddev = { ggx_stddev.x, ggx_stddev.y, 1.0f };
        float specular = 1.0f; //Tempest::FastFloatRand(seed);

		auto dir = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
		Tempest::Matrix3 basis;
		basis.makeBasis(dir);

		Tempest::Vector3 scaling, euler;
		basis.decompose(&scaling, &euler);

	    for(uint32_t idx = 0, idx_end = DummyBTFLightSamples*DummyBTFLightSamples; idx < idx_end; ++idx)
	    {
		    uint32_t view_idx = idx / btf->LightCount;
		    uint32_t light_idx = idx % btf->LightCount;

            auto inc_light = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]),
                 out_light = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);

        #ifdef TEST_SGGX
            auto micro_norm = Tempest::Normalize(inc_light + out_light);
            auto dir_density = Tempest::SGGXProjectedArea(sggx_stddev, basis, inc_light);
            auto brdf_value_arr = Tempest::SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, basis);
            auto brdf_value = Array(brdf_value_arr)[0];
        #else
            auto brdf_value = GGXNDFBRDF(ggx_stddev, inc_light, out_light)*inc_light.z;
        #endif

            reinterpret_cast<float*>(btf->LeftSingularU)[idx] = brdf_value;

            auto btf_value = Tempest::BTFFetchSpectrum(btf.get(), light_idx, view_idx, 0, 0);
            TGE_CHECK(brdf_value == Array(btf_value)[0], "Invalid generated BTF");
	    }

        auto brdf_integral = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples, 
                                                                  [&sggx_stddev, &basis](const Tempest::Vector3& out_light)
                                                                  {
                                                                      Tempest::Vector3 inc_light{ 0.0f, 0.0f, 1.0f };

                                                                  #ifdef TEST_SGGX
                                                                      auto micro_norm = Tempest::Normalize(inc_light + out_light);
                                                                      auto dir_density = Tempest::SGGXProjectedArea(sggx_stddev, basis, inc_light);
                                                                      auto brdf_value_arr = Tempest::SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, basis);
                                                                      auto brdf_value = Array(brdf_value_arr)[0];
                                                                  #else
                                                                      auto brdf_value = GGXNDFBRDF(Tempest::Vector2{ sggx_stddev.x, sggx_stddev.y }, inc_light, out_light);  
                                                                  #endif

                                                                      return brdf_value;
                                                                  });

        auto btf_ptr = btf.get();
        auto btf_integral = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples, 
                                                                  [btf_ptr](const Tempest::Vector3& out_light)
                                                                  {
                                                                      Tempest::Vector3 inc_light{ 0.0f, 0.0f, 1.0f };

                                                                      auto btf_value = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, inc_light, out_light, 0, 0);

                                                                      return Array(btf_value)[0];
                                                                  });

        for(uint32_t sample_idx = 0; sample_idx < BTFBRDFComparisonSamples; ++sample_idx)
        {
            auto inc_light = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
            auto out_light = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        #ifdef TEST_SGGX
            auto micro_norm = Tempest::Normalize(inc_light + out_light);

            auto dir_density = Tempest::SGGXProjectedArea(sggx_stddev, basis, inc_light);

            auto brdf_value_arr = Tempest::SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, basis);
            auto brdf_value = Array(brdf_value_arr)[0];
        #else
            auto brdf_value = GGXNDFBRDF(ggx_stddev, inc_light, out_light);
        #endif
            auto btf_value = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, inc_light, out_light, 0, 0);

            TGE_CHECK(Tempest::ApproxEqual(brdf_value, Array(btf_value)[0], 1e-1f) || Array(btf_value)[0] == 0.0f, "Bad quality sampling");
        }

        for(uint32_t sample_idx = 0; sample_idx < NDFReduceSamples; ++sample_idx)
        {
            auto norm = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };//Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

            float brdf_ndf_value = Tempest::ReduceToNDF(NDFReduceSamples, norm,
                                                   [&sggx_stddev, &basis](const Tempest::Vector3& inc_light, const Tempest::Vector3& out_light)
                                                   {
                                                   #ifdef TEST_SGGX
                                                       auto micro_norm = Tempest::Normalize(inc_light + out_light);
                                                       auto dir_density = Tempest::SGGXProjectedArea(sggx_stddev, basis, inc_light);
                                                       auto brdf_value_arr = Tempest::SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, basis);
                                                       auto brdf_value = Array(brdf_value_arr)[0];
                                                   #else
                                                       auto brdf_value = GGXNDFBRDF(Tempest::Vector2{ sggx_stddev.x, sggx_stddev.y }, inc_light, out_light);
                                                   #endif
                                                       return brdf_value;
                                                   });

            float ndf_value = Tempest::ReduceToNDF(NDFReduceSamples, norm,
                                                   [btf_ptr, &norm](const Tempest::Vector3& inc_light, const Tempest::Vector3& out_light)
                                                   {
                                                       auto spec = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_ptr, inc_light, out_light, 0, 0);
                                                       return Array(spec)[0];
                                                   });

            float actual_ndf_value = Tempest::SGGXMicroFlakeNDF(sggx_stddev, basis, norm);

            TGE_CHECK(Tempest::ApproxEqual(ndf_value, actual_ndf_value, 1e-3f), "Invalid ndf");
        }
        }

        auto recons_basis = Tempest::PerformPCA(thread_id, pool, false, EigenVectorSphereSamples, EigenVectorPlaneSamples, [&sggx_stddev, &basis](const Tempest::Vector3& norm)
                                                {
                                                    return Tempest::SGGXMicroFlakeNDF(sggx_stddev, basis, norm);
                                                });

		auto sggx_stddev_recons = Tempest::StratifiedMonteCarloIntegratorSphere<Tempest::Vector3>(ConvertSamples,
									  [&recons_basis, &sggx_stddev, &basis](const Tempest::Vector3& norm)
									  {
										  return Tempest::Vector3Abs(recons_basis.transformRotationInverse(norm))*Tempest::SGGXMicroFlakeNDF(sggx_stddev, basis, norm);
									  });

		for(uint32_t orig_axis_idx = 0; orig_axis_idx < 3; ++orig_axis_idx)
		{
			bool axis_found = false;
			for(uint32_t recons_axis_idx = 0; recons_axis_idx < 3; ++recons_axis_idx)
			{
				if(fabsf(Tempest::Dot(basis.column(orig_axis_idx), recons_basis.column(recons_axis_idx))) > 0.9f)
				{
					TGE_CHECK(Tempest::ApproxEqual(Array(sggx_stddev)[orig_axis_idx], Array(sggx_stddev_recons)[recons_axis_idx], 1e-1f), "Failed to reconstruct standard deviation");
					axis_found = true;
					break;
				}
			}
			TGE_CHECK(axis_found, "Cannot find equivalent axis");
		}
    }
}
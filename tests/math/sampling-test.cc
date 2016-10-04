#include "tempest/utils/testing.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/math/triangle.hh"
#include "tempest/math/intersect.hh"

const size_t TestSamples = 10000;
const size_t LowTestSamples = 16;
const size_t RandomSamples = 1024;
const size_t IntegratorSamples = 4*1024;

TGE_TEST("Testing drawing from different distributions")
{
    unsigned seed = 1;
    for(size_t i = 0; i < TestSamples; ++i)
    {
        auto x = Tempest::FastFloatRand(seed);
        float rand_value = Tempest::GaussianSampling(0.0f, 1.0f, x);
        auto prob = Tempest::GaussianCDF(0.0f, 1.0f, rand_value);
        TGE_CHECK(fabs(prob - x) < 1e-7f, "Invalid distribution");
    }

    float total_pdf = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSamples, 
                        [](const Tempest::Vector3& dir)
                        {
                            return Tempest::ComputeSphereAreaPDF({}, dir, { 0.0f, 0.0f, 1.0f }, 0.5f);
                        });
    
    TGE_CHECK(Tempest::ApproxEqual(total_pdf, 5e-2f, 1.0f), "Invalid sampling on a cone");

    for(size_t i = 0; i < LowTestSamples; ++i)
    {
        auto v0 = (1.0f + Tempest::FastFloatRand(seed))*Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed)),
             v1 = (1.0f + Tempest::FastFloatRand(seed))*Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed)),
             v2 = (1.0f + Tempest::FastFloatRand(seed))*Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        for(size_t j = 0; j < LowTestSamples; ++j)
        {
            auto sample_v = Tempest::UniformSampleTriangleArea({}, v0, v1, v2, Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

            float t;
            Tempest::Vector3 norm;
            Tempest::Vector2 barycentric;
            
            auto intersect = Tempest::IntersectRayTriangle(sample_v.Direction, {}, v0, v1, v2, &norm, &barycentric, &t);

            TGE_CHECK(intersect, "Invalid barycentric direction");
        }
        

        auto total_projected_area = 2.0f*Tempest::MathPi*Tempest::StratifiedMonteCarloIntegratorHemisphere(16*IntegratorSamples, 
            [&v0, &v1, &v2](const Tempest::Vector3& dir)
            {
                Tempest::Vector2 barycentric_coordinates;
                Tempest::Vector3 normal;
                float t;
                auto intersect = Tempest::IntersectRayTriangle(dir, {}, v0, v1, v2, &normal, &barycentric_coordinates, &t);
                return intersect ? 1.0f : 0.0f;
            });

        auto total_projected_area_approx = ProjectedTriangleSphereAreaApproximate({}, v0, v1, v2);

        // TODO: Implement Wang's approximation

        TGE_CHECK(total_projected_area_approx <= total_projected_area || total_projected_area_approx < 0.1f, "Invalid projected area");
    }

    for(size_t i = 0; i < LowTestSamples; ++i)
    {
        auto mean_dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        auto concentration = 1.0f + 80.0f*Tempest::FastFloatRand(seed);

        auto von_mises_int = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSamples,
                                [&mean_dir, concentration](const Tempest::Vector3& dir)
                                {
                                    auto value = Tempest::VonMisesFisherPDF(mean_dir, concentration, dir);
                                    TGE_CHECK(std::isfinite(value), "invalid value");
                                    return value;
                                });
        TGE_CHECK(Tempest::ApproxEqual(von_mises_int, 1.0f, 1e-1f), "Invalid PDF");

        auto angle = Tempest::MathPi*0.1f;
        float cos_angle = cosf(angle);

        auto vmf_integral = Tempest::StratifiedMonteCarloIntegratorSphericalCone(IntegratorSamples, cos_angle,
                                                                                 [concentration](const Tempest::Vector3& dir)
                                                                                 {
                                                                                     return Tempest::VonMisesFisherPDF(Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, concentration, dir);
                                                                                 });

        auto closed_form_integral = Tempest::VonMisesFisherConeCDF(concentration, cos_angle);
        
        TGE_CHECK(Tempest::ApproxEqual(vmf_integral, closed_form_integral, 1e-1f), "invalid integral");
    
        float required_tolerance = 0.5f + Tempest::FastFloatRand(seed)*0.5f;
    
        cos_angle = Tempest::VonMisesFisherToleranceCone(concentration, required_tolerance);

        auto vmf_tolerance_integral = Tempest::VonMisesFisherConeCDF(concentration, cos_angle);

        TGE_CHECK(Tempest::ApproxEqual(vmf_tolerance_integral, required_tolerance, 1e-1f), "invalid tolerance computation");

        angle = Tempest::LinearInterpolate(Tempest::MathPi/178.0f, Tempest::MathPi/4.0f, Tempest::FastFloatRand(seed));

        cos_angle = cosf(angle);

        float req_concentration = Tempest::BrentMethod(0.0001f, 10000.0f, 1e-6f, 
            [cos_angle, required_tolerance](float concentration) { return cos_angle - Tempest::VonMisesFisherToleranceCone(concentration, required_tolerance); });

        auto vmf_tolerance_req_angle = Tempest::StratifiedMonteCarloIntegratorSphericalCone(IntegratorSamples, cos_angle,
                                                                                           [req_concentration](const Tempest::Vector3& dir)
                                                                                           {
                                                                                               return Tempest::VonMisesFisherPDF(Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, req_concentration, dir);
                                                                                           });

        TGE_CHECK(Tempest::ApproxEqual(vmf_tolerance_integral, vmf_tolerance_req_angle, 1e-1f), "Invalid concentration estimation");
    }

    for(size_t i = 0; i < TestSamples; ++i)
    {
        auto uniform_sample = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        TGE_CHECK(Tempest::ApproxEqual(Tempest::Length(uniform_sample), 1.0f), "Bad uniform hemisphere sample generator");

        auto cos_sample = Tempest::CosineSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        TGE_CHECK(Tempest::ApproxEqual(Tempest::Length(cos_sample), 1.0f), "Bad cosine-weighted hemisphere sample generator");

        auto aniso_cos_sample = Tempest::AnisoPowerCosineSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), { 20.0f, 100.0f });
        TGE_CHECK(Tempest::ApproxEqual(Tempest::Length(aniso_cos_sample), 1.0f), "Bad anisotropic power cosine-weighted hemisphere sample generator");

        auto power_cos_sample = Tempest::PowerCosineSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), 100.0f);
        TGE_CHECK(Tempest::ApproxEqual(Tempest::Length(power_cos_sample), 1.0f), "Bad power cosine-weighted hemisphere sample generator");

        auto uniform_sphere_sample = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        TGE_CHECK(Tempest::ApproxEqual(Tempest::Length(uniform_sphere_sample), 1.0f), "Bad uniform sphere sample generator");
    }
}
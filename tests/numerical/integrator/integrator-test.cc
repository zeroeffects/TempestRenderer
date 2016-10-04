#include "tempest/utils/testing.hh"
#include "tempest/math/numerical-methods.hh"
#include <complex>

const uint32_t IntegratorSampleCount = 4096;

TGE_TEST("Testing whether numerical integrators work properly")
{
    auto sin_int = Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSampleCount, [](float angle) { return sinf(angle); });

    TGE_CHECK(Tempest::ApproxEqual(sin_int, 2.0f, 1e-4f), "Invalid sin integral");

    auto sin_int2 = Tempest::MonteCarloIntegrator(0.0f, Tempest::MathPi, IntegratorSampleCount, [](float angle) { return sinf(angle); });

    TGE_CHECK(Tempest::ApproxEqual(sin_int2, 2.0f, 1e-2f), "Invalid sin integral");

    auto sin_int3 = Tempest::StratifiedMonteCarloIntegrator(0.0f, Tempest::MathPi, IntegratorSampleCount, [](float angle) { return sinf(angle); });

    TGE_CHECK(Tempest::ApproxEqual(sin_int3, 2.0f, 1e-5f), "Invalid sin integral");

    auto sphere_integral = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSampleCount, [](const Tempest::Vector3& dir) { return 1.0f; });

    TGE_CHECK(Tempest::ApproxEqual(sphere_integral, 4.0f*Tempest::MathPi), "Invalid dot product integration");

    auto cos_int_norm = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSampleCount, [](float angle) { return Maxf(cosf(angle), 0.0f)*sinf(angle); });

    TGE_CHECK(Tempest::ApproxEqual(cos_int_norm, Tempest::MathPi, 1e-1f), "Invalid cos integrator");

    auto cos_norm = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSampleCount, [](const Tempest::Vector3& dir) { return Maxf(dir.z, 0.0f); });

    TGE_CHECK(Tempest::ApproxEqual(cos_norm, Tempest::MathPi, 1e-1f), "Invalid cos integrator");

    auto cos_norm2 = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSampleCount, [](const Tempest::Vector3& dir) { return Maxf(dir.z, 0.0f); });

    TGE_CHECK(Tempest::ApproxEqual(cos_norm2, Tempest::MathPi, 1e-1f), "Invalid dot integrator");

    auto indirect_sphere_int = Tempest::StratifiedMonteCarloIntegratorSphericalCone(IntegratorSampleCount, cosf(Tempest::MathPi), [](const Tempest::Vector3& dir) { return 1.0f; });

    TGE_CHECK(Tempest::ApproxEqual(indirect_sphere_int, 4.0f*Tempest::MathPi, 1e-1f), "Broken sphere integrator");

    auto indirect_hemisphere_int = Tempest::StratifiedMonteCarloIntegratorSphericalCone(IntegratorSampleCount, cosf(Tempest::MathPi*0.5f), [](const Tempest::Vector3& dir) { TGE_CHECK(dir.z >= 0.0f, "invalid direction"); return 1.0f; });

    TGE_CHECK(Tempest::ApproxEqual(indirect_hemisphere_int, 2.0f*Tempest::MathPi, 1e-1f), "Broken sphere integrator");

    float freq = IntegratorSampleCount/2.0f;

    auto integrate_to_zero = Tempest::SimpsonsCompositeRuleQuadratureIntegrator<std::complex<float>>(-1.0f, 1.0f, IntegratorSampleCount, [freq](float t)
                                {
                                    std::complex<float> arg{ 0.0f, -2.0f*Tempest::MathPi*freq*t };
                                    std::complex<float> value = std::exp(arg);
                                    return value;
                                });

    TGE_CHECK(Tempest::ApproxEqual(integrate_to_zero.real(), 0.0f, 1e-2f), "Bad integrator");
}
#define DEBUG_NUMERICAL

#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"
#include "tempest/math/numerical-methods.hh"

#include <algorithm>

inline float D(float rcp_sqrt2_stdev, float theta)
{
    return 0.5f - 0.5f * std::erff(cosf(theta) * rcp_sqrt2_stdev) / std::erff(rcp_sqrt2_stdev);
}

TGE_TEST("Testing Brent's method")
{
    unsigned seed = 1;
    float rcp_sqrt2_stdev = 1.0f / (Tempest::Sqrt2*0.3f);

    float D0 = D(rcp_sqrt2_stdev, 0.0f);
    float Dpi = D(rcp_sqrt2_stdev, Tempest::MathPi);

    TGE_CHECK(D0 == 0.0f && Dpi == 1.0f, "Bad function");

    size_t max_steps = 0;
    const size_t total_values = 10000;
    size_t total_steps = 0;

    for(size_t idx = 0; idx < total_values; ++idx)
    {
        float r1 = Tempest::FastFloatRand(seed);
        size_t cur_steps;
        float theta = Tempest::BrentMethod(0, Tempest::MathPi, 1e-6f,
                                           &cur_steps,
                                           [rcp_sqrt2_stdev, r1](float theta)
                                           {
                                               return D(rcp_sqrt2_stdev, theta) - r1;
                                           });
        max_steps = std::max(max_steps, cur_steps);
        total_steps += cur_steps;
    }

    Tempest::Log(Tempest::LogLevel::Info, "Maximum steps: ", max_steps, 
                                          "\nAverage steps: ", (float)total_steps/total_values);
}
#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"

TGE_TEST("Testing short-hand trigonometric functions out of paranoia")
{
    const size_t test_samples = 1024;

    for(size_t i = 0; i < test_samples; ++i)
    {
        float angle = 2.0f*Tempest::MathPi*i;
        float cos_angle = cosf(angle);
        TGE_CHECK(fabsf(cosf(2.0f*angle) - Tempest::Cos2x(cos_angle)) < 1e-6f, "Broken function");
        TGE_CHECK(fabsf(fabsf(cosf(0.5f*angle)) - Tempest::FastCos0_5x(cos_angle)) < 1e-6f, "Broken function");
    }
}
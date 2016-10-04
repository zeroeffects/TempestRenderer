#include "tempest/utils/testing.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/sampling3.hh"

const uint32_t AxisSamples = 16;

TGE_TEST("Testing math functions")
{
    for(uint32_t y = 0; y < AxisSamples; ++y)
    {
        for(uint32_t x = 0; x < AxisSamples; ++x)
        {
            Tempest::Vector2 orig{ (float)x/(AxisSamples - 1), (float)y/(AxisSamples - 1) };
            {
                auto cart = Tempest::ParabolicToCartesianCoordinates(orig);
                auto norm_cart = Tempest::Normalize(cart);

                auto recons = Tempest::CartesianToParabolicCoordinates(cart);
                TGE_CHECK(orig == recons, "Invalid conversion between parabolic and cartesian");
            }

            {
                Tempest::Vector3 cart = Tempest::UniformSampleHemisphere(orig.x, orig.y);

                auto sphere = Tempest::CartesianToLambertEqualAreaCoordinates(cart);
                auto recons_cart = Tempest::LambertEqualAreaToCartesianCoordinates(sphere);

                TGE_CHECK(recons_cart == cart, "Imvalid conversion");
            }
        }
    }
}
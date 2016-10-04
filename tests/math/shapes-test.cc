#include "tempest/utils/testing.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/math/shapes.hh"

const uint32_t ExperimentCount = 1024;

TGE_TEST("Testing shape related functions implementation")
{
    unsigned seed = 1;
    for(uint32_t i = 0; i < ExperimentCount; ++i)
    {
        Tempest::Vector2 v0{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) },
                         v1{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) },
                         v2{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };

        Tempest::AABB2 aabb;
        Tempest::TriangleBounds(v0, v1, v2, &aabb);

        auto min = Tempest::Vector2Min(v0, Tempest::Vector2Min(v1, v2));
        auto max = Tempest::Vector2Max(v0, Tempest::Vector2Max(v1, v2));

        TGE_CHECK(min == aabb.MinCorner && max == aabb.MaxCorner, "Invalid AABB generator");
    }
}
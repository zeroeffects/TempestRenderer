#include "tempest/utils/testing.hh"
#include "tempest/math/intersect.hh"
#include "tempest/math/vector3.hh"

TGE_TEST("Testing circle line intersection")
{
    uint32_t mirand = 1234;
    Tempest::Vector2 lineOrg = {0.0f, 0.0f};
    Tempest::Vector2 lineDir = {0.0f, 0.0f};

    Tempest::Vector2 circleOrg = {0.0f, 0.0f};
    float circleRadius = 0.5f;

    uint32_t testSamples_line = 10;
    uint32_t testSamples_circle = 1;

    float tmin, tmax;


    for (uint32_t sample_line = 0; sample_line < testSamples_line; sample_line++) {
        lineOrg = {Tempest::FastFloatRand(mirand),
                   Tempest::FastFloatRand(mirand)};

        lineDir = {Tempest::FastFloatRand(mirand),
                   Tempest::FastFloatRand(mirand)};

        for (uint32_t sample_circle = 0; sample_circle < testSamples_circle; sample_circle++) {
            circleOrg = {Tempest::FastFloatRand(mirand),
                         Tempest::FastFloatRand(mirand)};

            circleRadius = 0.5f * Tempest::FastFloatRand(mirand);

            if (Tempest::IntersectLineCircleInPlane(lineDir, lineOrg, circleOrg, circleRadius, &tmin, &tmax)) {
                TGE_CHECK(fabsf(tmax - tmin) <= 2.0f * circleRadius, "Intersection should be 2 x radius at most");
            }
        }
    }

    circleOrg = {0.0f, 0.0f};
    circleRadius = 0.25f;
    float containedLength;
    uint32_t intersectionCount = 0;

    for (uint32_t sample_line = 0; sample_line < testSamples_line; sample_line++) {
        lineOrg = {Tempest::FastFloatRand(mirand),
                   Tempest::FastFloatRand(mirand)};

        lineDir = lineOrg - circleOrg;
        Tempest::NormalizeSelf(&lineDir);

        if (Tempest::IntersectLineCircleInPlane(lineDir, lineOrg, circleOrg, circleRadius, &tmin, &tmax)) {
            intersectionCount += 1;

            if (sample_line > 0) {
                TGE_CHECK((containedLength - fabsf(tmax - tmin)) < 1e-6f, "Intersections through midpoint not equal");
            }
            containedLength = fabsf(tmax - tmin);
        }
    }
    TGE_CHECK(intersectionCount == testSamples_line, "Recorded to few intersections");
}

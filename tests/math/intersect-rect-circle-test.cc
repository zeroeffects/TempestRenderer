#include "tempest/utils/testing.hh"
#include "tempest/math/intersect.hh"
#include "tempest/math/vector3.hh"

TGE_TEST("Testing circle rectangle intersection area")
{
    Tempest::Vector2 circleOrg = {0.0f, 0.0f};
    float circleRadius = 0.5f;
    float intersectionArea;

    Tempest::Vector2 rectOrg = {0.0f, 1.0f};
    Tempest::Vector2 rectSize = {2.0f, 4.0f};
    Tempest::Vector2 rectDir = {0.0f, 1.0f};

    IntersectRectCircleInPlane(rectOrg, rectSize, rectDir,
                               circleOrg, circleRadius, &intersectionArea);

    printf("Intersection area is %f", intersectionArea);

}

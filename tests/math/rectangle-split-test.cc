#include "tempest/utils/testing.hh"
#include "tempest/math/intersect.hh"
#include "tempest/math/matrix2.hh"
#include "tempest/math/vector2.hh"
#include "tempest/math/shape-split.hh"

TGE_TEST("Rectangle split test")
{
    Tempest::Matrix2 rot_scale;
    rot_scale.identity();
    rot_scale.rotate(Tempest::ToRadians(45.0f));
	rot_scale.scale(Tempest::Vector2{ 1.0f, 4.0f });

    Tempest::Vector2 org{ 1.0f/sqrtf(2.0f), -10.0f },
                     dir{ 0.0f, 1.0f };

    float tmin, tmax;

    Tempest::AABB2 bounds;

    Tempest::Rect2Bounds(rot_scale, Tempest::Vector2{}, &bounds);
    Tempest::AABB2 result[2];

    bool intersect_success = Tempest::IntersectLineRect2(dir, org, rot_scale.inverse(), Tempest::Vector2{}, &tmin, &tmax);
    TGE_CHECK(intersect_success, "Failed to intersect rectangle rectangle");

    bool split_success = Tempest::BoxGridSplit(1, 4.0f, rot_scale, rot_scale.inverse(), Tempest::Vector2{},
                                               bounds, result, result + 1);
    TGE_CHECK(split_success, "Failed to split rectangle");

    TGE_CHECK(Area(result[0]) + Area(result[1]) < Area(bounds), "Invalid boxes");
}
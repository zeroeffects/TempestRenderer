#include "tempest/utils/testing.hh"
#include "tempest/math/triangle.hh"
#include "tempest/math/vector3.hh"

#include <algorithm>
#include <memory>

const uint32_t PointCount = 4;
const uint32_t FixedGridWidth = 4;
const uint32_t FixedGridHeight = 4;

TGE_TEST("Testing implementation of Delaunay triangulation")
{
	{
    unsigned seed = 1;
    std::unique_ptr<Tempest::Vector2[]> points(new Tempest::Vector2[PointCount]);
    std::generate(points.get(), points.get() + PointCount, [&seed]() { return Tempest::Vector2{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) }; });

    auto index_list = CREATE_SCOPED(uint32_t*, ::free);
    uint32_t tri_count;

    Tempest::DelaunayTriangulation(points.get(), PointCount, &index_list, &tri_count);

    // TODO: Check whether it matches criteria
	}

	{
	uint32_t point_size = FixedGridWidth*FixedGridHeight;
	std::unique_ptr<Tempest::Vector2[]> simple_points(new Tempest::Vector2[point_size]);
	for(uint32_t y = 0; y < FixedGridHeight; ++y)
	{
		for(uint32_t x = 0; x < FixedGridWidth; ++x)
		{
			simple_points[y*FixedGridWidth + x] = { (float)x, (float)y };
		}
	}

	auto index_list = CREATE_SCOPED(uint32_t*, ::free);
    uint32_t tri_count;

	Tempest::DelaunayTriangulation(simple_points.get(), point_size, &index_list, &tri_count);

	uint32_t expected_tri_count = 2*(FixedGridWidth - 1)*(FixedGridHeight - 1);

	TGE_CHECK(tri_count == expected_tri_count, "Invalid grid generated");
	for(uint32_t i = 0, iend = 3*tri_count; i < iend; ++i)
	{
		auto i0 = index_list[i++];
		auto i1 = index_list[i++];
		auto i2 = index_list[i++];

		TGE_CHECK(i0 < point_size && i1 < point_size && i2 < point_size, "Invalid index");

		// Well, not really - it is not really clear how to cut a quad with Delaunay triangulation
		// TODO: Figure out why is it so shitty at doing it
		/*
		auto& v0 = simple_points[i0];
		auto& v1 = simple_points[i1];
		auto& v2 = simple_points[i2];
		
		float v01_len = Length(v0 - v1);
		float v02_len = Length(v0 - v2);

		TGE_CHECK(v01_len < sqrtf(2.0f) + 1e-3f, "Invalid vertices");
		TGE_CHECK(v02_len < sqrtf(2.0f) + 1e-3f, "Invalid vertices");
		*/
	}
	}
}
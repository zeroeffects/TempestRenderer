#include "tempest/utils/testing.hh"
#include "tempest/mesh/octree.hh"
#include "tempest/image/btf.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/timer.hh"

const uint32_t RandomSampleCount = 1024;

TGE_TEST("Testing octree builders and intersection")
{
    {
    Tempest::Vector3 points[] =
    {
        { -1.0f, -1.0f, -1.0f },
        { -1.0f, -1.0f, +1.0f },
        { -1.0f, +1.0f, -1.0f },
        { -1.0f, +1.0f, +1.0f },
        { +1.0f, -1.0f, -1.0f },
        { +1.0f, -1.0f, +1.0f },
        { +1.0f, +1.0f, -1.0f },
        { +1.0f, +1.0f, +1.0f },
    };

    Tempest::Octree octree = Tempest::BuildOctreeMorton(points, TGE_FIXED_ARRAY_SIZE(points));
    
    bool check_status = Tempest::CheckOctree(octree, points, TGE_FIXED_ARRAY_SIZE(points));
    TGE_CHECK(check_status, "Invalid octree");
    }

    {
    Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(ROOT_SOURCE_DIR "/tests/image/btf/fabric09_resampled_W400xH400_L151xV151.btf")));
	TGE_CHECK(btf, "Failed to load BTF");

    auto light_count = btf->LightCount;
    std::unique_ptr<Tempest::Vector3[]> lights(new Tempest::Vector3[light_count]);
    for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
    {
        lights[light_idx] = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
    }

    Tempest::Octree octree = Tempest::BuildOctreeMorton(lights.get(), light_count);

    bool check_status = Tempest::CheckOctree(octree, lights.get(), light_count);
    TGE_CHECK(check_status, "Invalid octree");

    unsigned seed = 1;

    std::vector<uint32_t> octree_points, brute_force_points;

    Tempest::TimeQuery timer;

    for(uint32_t sample_idx = 0; sample_idx < RandomSampleCount; ++sample_idx)
    {
        Tempest::AABBUnaligned box;
        box.MinCorner = { Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };
        box.MaxCorner = Tempest::Vector3Clamp(box.MinCorner + Tempest::Vector3{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) }, 0.0f, 1.0f);
        
        octree_points.clear();
        brute_force_points.clear();

        auto octree_start = timer.time();

        Tempest::OctreeIntersect(octree, lights.get(), light_count, box,
                                 [&octree_points](int32_t point_idx)
                                 {
                                     octree_points.push_back(point_idx);
                                 });

        auto octree_elapsed = timer.time() - octree_start;

        auto brute_force_start = timer.time();

        for(uint32_t point_idx = 0; point_idx < light_count; ++point_idx)
        {
            auto& point = lights[point_idx];
            if(box.MinCorner <= point && point <= box.MaxCorner)
                brute_force_points.push_back(point_idx);
        }

        auto brute_force_elapsed = timer.time() - brute_force_start;

        if(octree_elapsed < brute_force_elapsed && octree_points.size())
        {
            Tempest::Log(Tempest::LogLevel::Info, "improved performance when query: ", octree_points.size(), "; ", octree_elapsed, "us; ", brute_force_elapsed, "us");
        }

        std::sort(std::begin(octree_points), std::end(octree_points));
        std::sort(std::begin(brute_force_points), std::end(brute_force_points));

        TGE_CHECK(std::equal(std::begin(octree_points), std::end(octree_points), std::begin(brute_force_points)), "Broken octree intersection");
    }

    }
}
#include "tempest/utils/testing.hh"
#include "tempest/math/triangle.hh"

TGE_TEST("Misc triangle-related function test")
{
    Tempest::Vector3 verts[] =
    {
        { 0.0f,   0.0f, 0.0f },
        { 15.0f,  0.0f, 0.0f },
        { 0.0f,  15.0f, 0.0f },
        { 15.0f, 15.0f, 0.0f }
    };

    uint32_t indices[] =
    {
        0, 2, 1,
        1, 2, 3
    };

    Tempest::Vector3 prev_tangent, prev_binormal,
                     cur_tangent, cur_binormal;

    for(uint32_t idx = 0; idx < TGE_FIXED_ARRAY_SIZE(indices); idx += 3)
    {
        for(uint32_t permutation_idx = 0; permutation_idx < 3; ++permutation_idx)
        {
            uint32_t i0 = indices[idx + (permutation_idx % 3)],
                     i1 = indices[idx + ((permutation_idx + 1) % 3)],
                     i2 = indices[idx + ((permutation_idx + 2) % 3)];

            auto& v0 = verts[i0];
            auto& v1 = verts[i1];
            auto& v2 = verts[i2];

            Tempest::Vector2 tc0{v0.x, v0.y},
                             tc1{v1.x, v1.y},
                             tc2{v2.x, v2.y};

            Tempest::GenerateTangentSpace(v0, v1, v2, tc0, tc1, tc2, &cur_tangent, &cur_binormal);

            Tempest::Vector2 dtc1 = tc1 - tc0;
            Tempest::Vector2 dtc2 = tc2 - tc0; 

            auto approx_v1 = cur_tangent*tc1.x + cur_binormal*tc1.y,
                 approx_v2 = cur_tangent*tc2.x + cur_binormal*tc2.y;
            TGE_CHECK(approx_v1 == v1, "Invalid tangent");
            TGE_CHECK(approx_v2 == v2, "Invalid binormal");

            if(idx > 3)
            {
                TGE_CHECK(prev_tangent == cur_tangent, "Invalid tangent");
                TGE_CHECK(prev_binormal == cur_binormal, "Invalid binormal");
            }

            prev_tangent = cur_tangent;
            prev_binormal = cur_binormal;
        }
    }
}
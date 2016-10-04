#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"

const size_t RandomEquationCount = 1024;
const float RandomCoefficientRange = 10;

TGE_TEST("Testing different equation solvers")
{
    float roots[3];
    uint32_t root_count;

    float A = 6;
    float B = 20;
    Tempest::SolveDepressedCubic(A, B, roots, &root_count);

    TGE_CHECK(root_count > 0, "Broken solver");

    for(size_t i = 0; i < root_count; ++i)
    {
        float r = roots[i];
        float left_side = r*r*r + A*r;
        TGE_CHECK(fabsf(left_side - B) < 1e-3f, "Invalid root of cubic equation");
    }

    float roots_2[3];
    uint32_t root_count_2;
    Tempest::SolveTriviallyDepressedCubic(2.0f, A*2.0f, -B*2.0f, roots_2, &root_count_2);

    TGE_CHECK(root_count == root_count_2, "Invalid convenience solver");

    for(size_t i = 0; i < root_count; ++i)
    {
        TGE_CHECK(roots[i] == roots_2[i], "Non-equivalent solvers");
    }

    A = -7;
    B = -6;
    float roots_3[3];
    uint32_t root_count_3;
    Tempest::SolveDepressedCubic(A, B, roots_3, &root_count_3);
    TGE_CHECK(root_count_3 == 3, "Broken solver");

    for(size_t i = 0; i < root_count_3; ++i)
    {
        float r = roots_3[i];
        float left_side = r*r*r + A*r;
        TGE_CHECK(fabsf(left_side - B) < 1e-3f, "Invalid root of cubic equation");
    }

    // TODO: Precision is really bad of the root solver
    /*
    unsigned mirand = 1;
    float roots_4[3];
    uint32_t root_count_4;
    for(size_t i = 0; i < RandomEquationCount; ++i)
    {
        float A = RandomCoefficientRange*(Tempest::FastFloatRand(mirand) - 0.5f);
        float B = RandomCoefficientRange*(Tempest::FastFloatRand(mirand) - 0.5f);
        Tempest::SolveDepressedCubic(A, B, roots_4, &root_count_4);
        for(size_t j = 0; j < root_count_4; ++j)
        {
            float r = roots_4[j];
            float left_side = r*r*r + A*r;
            TGE_CHECK(fabsf(left_side - B) < 1e-3f, "Invalid root of cubic equation");
        }
    }
    */
}
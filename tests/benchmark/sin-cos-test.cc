#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/timer.hh"
#include "tempest/utils/logging.hh"

const uint32_t Rerun = 100000000;

TGE_TEST("Benchmarking sin cos")
{
    Tempest::TimeQuery timer;

    {
    float sum_sin = 0.0f, sum_cos = 0.0f;
    unsigned seed = 1;
       
    auto start = timer.time();
    for(size_t i = 0; i < Rerun; ++i)
    {
        float s, c;
        float omega = 2.0f*Tempest::MathPi*Tempest::FloatIntelFastRand(seed);
        Tempest::SinCos_Naive(omega, &s, &c);
        sum_sin += s;
        sum_cos += c;
    }
    auto end = timer.time();
    Tempest::Log(Tempest::LogLevel::Info, "Computation time Sin Cos naive implementation: ", (end - start), "us");
    Tempest::Log(Tempest::LogLevel::Info, "Sums: ", sum_sin, ", ", sum_cos);
    }

    {
    float sum_sin = 0.0f, sum_cos = 0.0f;
    unsigned seed = 1;

    auto start = timer.time();
    for(size_t i = 0; i < Rerun; ++i)
    {
        float s, c;
        float omega = 2.0f*Tempest::MathPi*Tempest::FloatIntelFastRand(seed);
        Tempest::SinCos_Tan(omega, &s, &c);
        sum_sin += s;
        sum_cos += c;

    }
    auto end = timer.time();
    Tempest::Log(Tempest::LogLevel::Info, "Computation time Sin Cos through tangent implementation: ", (end - start), "us");
    Tempest::Log(Tempest::LogLevel::Info, "Sums: ", sum_sin, ", ", sum_cos);
    }

    {
    float sum_sin = 0.0f, sum_cos = 0.0f;
    unsigned seed = 1;

    auto start = timer.time();
    for(size_t i = 0; i < Rerun; ++i)
    {
        float s, c;
        float omega = 2.0f*Tempest::MathPi*Tempest::FloatIntelFastRand(seed);
        Tempest::SinCos_SqrtCorrect(omega, &s, &c);
        sum_sin += s;
        sum_cos += c;
    }
    auto end = timer.time();
    Tempest::Log(Tempest::LogLevel::Info, "Computation time Sin Cos sqrt correct implementation: ", (end - start), "us");
    Tempest::Log(Tempest::LogLevel::Info, "Sums: ", sum_sin, ", ", sum_cos);
    }
}
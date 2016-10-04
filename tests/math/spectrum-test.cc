#include "tempest/utils/testing.hh"
#include "tempest/math/spectrum.hh"

const size_t RandomTestSampleCount = 1024;

typedef Tempest::Vector3 (*ColorConversionFunction)(const Tempest::Vector3& color);

TGE_TEST("Testing spectrum conversion")
{
    Tempest::Vector3 orig_color{0.5f, 0.0f, 0.0f};

    Tempest::InitSpectrum();

    auto conv_color = Tempest::SRGBToXYZ(orig_color);
    auto recons_color = Tempest::XYZToSRGB(conv_color);

    TGE_CHECK(Length(orig_color - recons_color) < 1e-3f, "Bad color transformations between SRGB and XYZ");

    auto conv_spec = Tempest::SRGBToSpectrum(orig_color);
    auto xyz_back = Tempest::SpectrumToXYZ(conv_spec);
    auto recons_color2 = Tempest::XYZToSRGB(xyz_back);

    TGE_CHECK(Length(orig_color - recons_color2) < 1e-3f, "Bad color transformations between SRGB and XYZ");

    Tempest::Matrix3 conv(Tempest::Vector3{ 0.183f,  0.614f,  0.062f},
                          Tempest::Vector3{-0.101f, -0.339f,  0.439f},
                          Tempest::Vector3{ 0.439f, -0.399f, -0.040f});

    auto ycbcr_inv = conv.inverse();
    Tempest::Matrix3 exp_conv(Tempest::Vector3{1.164f,    0.0f,  1.793f},
                              Tempest::Vector3{1.164f, -0.213f, -0.533f},
                              Tempest::Vector3{1.164f,  2.112f,    0.0f});

    TGE_CHECK(Tempest::ApproxEqual(ycbcr_inv, exp_conv, 1e-2f), "Invalid YCbCr conversion matrix");

    auto conv_ycbcr = Tempest::RGBToYCbCr(orig_color);
    auto recons_color3 = Tempest::YCbCrToRGB(conv_ycbcr);

    TGE_CHECK(Tempest::ApproxEqual(orig_color, recons_color3, 1e-3f), "Bad color transformation between RGB and YCbCr");

    struct TestFunction
    {
        ColorConversionFunction ForwardConversion,
                                InverseConversion;
    } test_functions[] =
    {
        { Tempest::_impl_RGBToHSL, Tempest::_impl_HSLToRGB }
    };

    struct TestCase
    {
        Tempest::Vector3        InitialValue;
        Tempest::Vector3        ExpectedValue[TGE_FIXED_ARRAY_SIZE(test_functions)];
    } test_cases[] =
    {
        // RGB                  | HSL
        { { 0.0f, 0.0f, 0.0f }, { { 0.0f, 0.0f, 0.0f } } },
        { { 1.0f, 0.0f, 0.0f }, { { 0.0f, 1.0f, 0.5f } } },
        { { 0.0f, 1.0f, 0.0f }, { { 2.0f*Tempest::MathPi/3.0f, 1.0f, 0.5f } } },
        { { 0.0f, 0.0f, 1.0f }, { { 4.0f*Tempest::MathPi/3.0f, 1.0f, 0.5f } } }
    };
    
    for(size_t test_idx = 0; test_idx < TGE_FIXED_ARRAY_SIZE(test_cases); ++test_idx)
    {
        auto& test_case = test_cases[test_idx];
        for(size_t func_idx = 0; func_idx < TGE_FIXED_ARRAY_SIZE(test_functions); ++func_idx)
        {
            auto& test_func_set = test_functions[func_idx];
            auto& exp_value = test_case.ExpectedValue[func_idx];
            auto comp_value = test_func_set.ForwardConversion(test_case.InitialValue);
            TGE_CHECK(exp_value == comp_value, "Bad quality conversion");
            auto comp_init_value = test_func_set.InverseConversion(comp_value);
            TGE_CHECK(comp_init_value == test_case.InitialValue, "Irreversible conversion");
        }
    }

    unsigned seed = 1;

    for(size_t test_idx = 0; test_idx < RandomTestSampleCount; ++test_idx)
    {
        for(size_t func_idx = 0; func_idx < TGE_FIXED_ARRAY_SIZE(test_functions); ++func_idx)
        {
            auto& test_func_set = test_functions[func_idx];
            Tempest::Vector3 init_value{ Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed) };
            auto comp_value = test_func_set.ForwardConversion(init_value);
            auto comp_init_value = test_func_set.InverseConversion(comp_value);
            TGE_CHECK(comp_init_value == init_value, "Irreversible conversion");
        }
    }
}
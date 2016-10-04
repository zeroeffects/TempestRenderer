#include "tempest/utils/testing.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"
#include "tempest/math/spectrum.hh"

const uint32_t SamplesLong = 100;
const uint32_t SamplesAzimuth = 100;

namespace Tempest
{
namespace Cpp
{
Spectrum MarschnerBCSDF(const SampleData& sample_data);
}
}

TGE_TEST("Testing Marschner lobe diagram")
{
    Tempest::Vector3 color{ 0.2f, 0.2f, 0.2f };

    Tempest::RTMarschnerHairMaterial material;
    material.Absorption = Tempest::RGBToSpectrum(color);
    material.Eccentricity = 1.0f;
    material.LongitudalLobeShift = Tempest::ToRadians(-7.5f); // -5 to -10
    material.LongitudalLobeWidth = Tempest::ToRadians(7.5f); // 5 to 10
    material.CausticWidth = Tempest::ToRadians(16.5f); // 10 to 25
    material.CausticFadeRange = 0.3f; // 0.2 to 0.4
    material.CausticIntesityLimit = 0.5f;
    material.GlintScale = 2.75f; // 0.5 to 5.0

    material.Model = Tempest::IlluminationModel::Marschner;
	material.Diffuse = Tempest::RGBToSpectrum(color);
    material.Specular = Tempest::RGBToSpectrum(color);
    material.SpecularPower = Tempest::Vector2{ 100.0f, 100.0f };
	material.Fresnel.x = Tempest::ComputeReflectionCoefficient(1.56f);
    material.setup();

    Tempest::SampleData sample_data;
    sample_data.Material = &material;
    sample_data.TexCoord = {};
    sample_data.Tangent = Tempest::Vector3{0.0f, 0.0f, 1.0f};
	sample_data.Binormal = Tempest::Vector3{0.0f, -1.0f, 0.0f};
	sample_data.Normal = Cross(sample_data.Tangent, sample_data.Binormal);
    sample_data.TotalDensity = 0.0f;
    sample_data.DirectionalDensity = 0.0f;
    sample_data.PDF = 1.0f;

    uint32_t result[SamplesLong*SamplesAzimuth];

    for(uint32_t samp = 0, samp_end = SamplesLong*SamplesAzimuth; samp < samp_end; ++samp)
    {
        uint32_t samp_azimuth = samp % SamplesAzimuth;
        uint32_t samp_long = samp / SamplesAzimuth;

    //    Tempest::Log(Tempest::LogLevel::Debug, rot_angle_azimuth, "(", samp_azimuth, "), ", rot_angle_long, "(", samp_long, ")");
        float rot_angle_azimuth = Tempest::MathPi*(2.0f*samp_azimuth/(SamplesAzimuth - 1) - 1.0f);
        float rot_angle_long = 0.5f*Tempest::MathPi*(2.0f*samp_long/(SamplesLong - 1) - 1.0f);
        Tempest::Matrix4 rot;
        rot.identity();
        rot.rotateX(rot_angle_long);
        rot.rotateZ(rot_angle_azimuth);

        Tempest::Vector3 light = Tempest::Vector3{0.0f, -1.0f, 0.0f};
        Tempest::Vector3 view = rot.transformRotate(light);

        sample_data.OutgoingLight = view;
        sample_data.IncidentLight = light;

        result[samp] = Tempest::ToColor(Tempest::XYZToSRGB(Tempest::SpectrumToXYZ(Tempest::Cpp::MarschnerBCSDF(sample_data))));
    }

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = SamplesAzimuth;
    tex_desc.Height = SamplesLong;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    Tempest::SaveImage(tex_desc, result, Tempest::Path("marschner-lobe-diagram.tga"));
}
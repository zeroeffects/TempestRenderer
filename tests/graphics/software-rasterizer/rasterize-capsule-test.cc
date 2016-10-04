#include "tempest/utils/testing.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/graphics/software-rasterizer.hh"

const uint16_t TextureWidth = 512;
const uint16_t TextureHeight = 512;

TGE_TEST("Testing capsule rasterization capabilities")
{
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = 512;
    tex_desc.Height = 512;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    auto* tex_data = new uint8_t[tex_desc.Width*tex_desc.Height*Tempest::DataFormatElementSize(tex_desc.Format)];
    auto* tex_data_u32 = reinterpret_cast<uint32_t*>(tex_data);

    auto color = Tempest::ToColor(Tempest::Vector3{ 1.0f, 0.0f, 0.0f });
    std::fill(tex_data_u32, tex_data_u32 + tex_desc.Width*tex_desc.Height, color);

    Tempest::Texture tex(tex_desc, tex_data);

    struct RasterData
    {
        uint32_t* Data;
        float            Radius;
        float            Length;
        Tempest::Vector2 Direction;
    } rdata;

    Tempest::Vector2 p0{ 100, 250 },
                     p1{ 400, 450 };

    //rdata.Tangent = p0 - p1;
    //NormalizeSelf(&rdata.Tangent);
    //rdata.Binormal = { rdata.Tangent.y, -rdata.Tangent.x };

    rdata.Data = tex_data_u32;
    rdata.Radius = 50;
    rdata.Length = Length(p0 - p1);
    rdata.Direction = p1 - p0;
    rdata.Length = Tempest::Length(rdata.Direction);
    rdata.Direction /= rdata.Length;

    Tempest::Rasterizer::RasterizeCapsule2(Tempest::Capsule2{ { p0, p1 }, rdata.Radius }, tex_desc.Width, tex_desc.Height,
                                          [&rdata](uint32_t x, uint32_t y, uint32_t width, uint32_t height, const Tempest::Vector2& dist_vector, float tangent_ratio)
                                          {
                                              float binorm_ratio = fabsf(Tempest::WedgeZ(dist_vector, rdata.Direction));

                                              float circular_shift = sqrtf(rdata.Radius*rdata.Radius - binorm_ratio*binorm_ratio)/rdata.Length;

                                              float tr = circular_shift ? 1.0f - Tempest::Clamp(-(tangent_ratio - circular_shift)/(2.0f*circular_shift), 0.0f, 1.0f) : 1.0f;

                                              tr *= 1.0f - fabsf(binorm_ratio)/rdata.Radius;

                                              auto& pixel = rdata.Data[y*width + x];
                                              auto prev = Tempest::ToVector3(pixel);
                                              auto cur = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
                                              pixel = Tempest::ToColor(prev*(1 - tr) + cur*tr);
                                          });

    Tempest::DisplayImage(tex_desc, tex_data);
}
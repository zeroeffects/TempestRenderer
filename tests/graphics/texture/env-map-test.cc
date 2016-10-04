#include "tempest/utils/testing.hh"
#include "tempest/graphics/equirectangular-map.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/image/image.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/graphics/cube-map.hh"

const size_t TestSampleCount = 1024;
const uint32_t ImageWidth = 800,
               ImageHeight = 600,
               CubeWidth = 1024,
               CubeHeight = 1024;

TGE_TEST("Testing environment map rendering capabilities")
{
    unsigned seed = 1;

    for(size_t idx = 0; idx < TestSampleCount; ++idx)
    {
        auto dir = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        auto tc = Tempest::CartesianToEquirectangularCoordinates(dir);
        auto recons_dir = Tempest::EquirectangularToCartesianCoordinates(tc);
        TGE_CHECK(Tempest::ApproxEqual(dir, recons_dir, 1e-4f), "Invalid conversion between different spaces");
    }

    std::unique_ptr<Tempest::Texture> image(Tempest::LoadImage(Tempest::Path(TEST_ASSETS_DIR "/light-probes/grace-new.exr")));
    TGE_CHECK(image, "Missing texture");
    Tempest::EquirectangularMap eqrect_map(image.get());

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = ImageWidth;
    tex_desc.Height = ImageHeight;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    std::unique_ptr<uint32_t[]> data(new uint32_t[tex_desc.Width*tex_desc.Height]);

    Tempest::Vector3 target{0.0f, 1.0f, 0.0f},
                     origin{0.0f, 1.0f, 5.5f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAtUpTarget(origin, target, up);

    Tempest::Matrix4 proj = Tempest::PerspectiveMatrix(50.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);
    auto view_proj = proj*view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    for(size_t y = 0; y < tex_desc.Height; ++y)
        for(size_t x = 0; x < tex_desc.Width; ++x)
        {
            Tempest::Vector4 screen_tc{2.0f*x/(tex_desc.Width - 1) - 1.0f, 2.0f*y/(tex_desc.Height - 1) - 1.0f, 1.0f, 1.0};

			auto pos_end = Tempest::ToVector3(view_proj_inv*screen_tc);
            
            auto dir = Normalize(pos_end - origin);

            data[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::ConvertLinearToSRGB(eqrect_map.sampleRGB(dir)));
        }

    Tempest::SaveImage(tex_desc, data.get(), Tempest::Path("env.png"));

    for(size_t sample_idx = 0; sample_idx < TestSampleCount; ++sample_idx)
    {
        Tempest::Vector3 dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        size_t face_id;
        auto tc = Tempest::CartesianToCubeMapCoordinates(dir, &face_id);

        auto recons_dir = Tempest::CubeMapToCartesianCoordinates(tc, face_id);
        TGE_CHECK(Tempest::ApproxEqual(recons_dir, dir, 1e-3f), "Invalid reconstruction");
    }

    Tempest::TextureDescription cube_tex_desc;
    cube_tex_desc.Width = CubeWidth;
    cube_tex_desc.Height = CubeHeight;
    cube_tex_desc.Format = Tempest::DataFormat::RGBA32F;

    size_t plane_size = (size_t)tex_desc.Width*tex_desc.Height;
    std::unique_ptr<uint32_t[]> cube_data_proj(new uint32_t[plane_size]);
    std::unique_ptr<Tempest::CubeMap> env_cube_map(Tempest::ConvertEquirectangularMapToCubeMap(cube_tex_desc, eqrect_map));

    auto* cube_data = reinterpret_cast<const Tempest::Vector4*>(env_cube_map->getData());
    for(size_t face = 0; face < 6; ++face)
    {
        auto* data_plane = cube_data + face*plane_size;
        for(size_t y = 0, height = cube_tex_desc.Height; y < height; ++y)
            for(size_t x = 0, width = cube_tex_desc.Width; x < width; ++x)
            {
                Tempest::Vector2 tc{ (x + 0.5f)/width, (y + 0.5f)/height };
                auto dir = Tempest::CubeMapToCartesianCoordinates(tc, face);
                auto eqrect_sample = eqrect_map.sampleSpectrum(dir);
                auto cube_sample = env_cube_map->sampleSpectrum(dir);
                TGE_CHECK(Tempest::ApproxEqual(eqrect_sample, cube_sample, (MaxValue(eqrect_sample) + 1e-3f)*1e-3f), "invalid sampling when constructing cube map");
            }
    }


    for(size_t y = 0; y < tex_desc.Height; ++y)
        for(size_t x = 0; x < tex_desc.Width; ++x)
        {
            Tempest::Vector4 screen_tc{2.0f*x/(tex_desc.Width - 1) - 1.0f, 2.0f*y/(tex_desc.Height - 1) - 1.0f, 1.0f, 1.0};

			auto pos_end = Tempest::ToVector3(view_proj_inv*screen_tc);
            
            auto dir = Normalize(pos_end - origin);

            cube_data_proj[y*tex_desc.Width + x] = Tempest::ToColor(Tempest::ConvertLinearToSRGB(env_cube_map->sampleRGB(dir)));
        }

    Tempest::SaveImage(tex_desc, cube_data_proj.get(), Tempest::Path("env-cube.png"));

    for(size_t idx = 0, idx_end = tex_desc.Width*tex_desc.Height; idx < idx_end; ++idx)
    {
        auto color0 = Tempest::ToVector3(data[idx]);
        auto color1 = Tempest::ToVector3(cube_data_proj[idx]);
        TGE_CHECK(Tempest::ApproxEqual(color0, color1, 0.5f), "Broken cube map conversion"); // Somewhat horribly distorted
    }
}
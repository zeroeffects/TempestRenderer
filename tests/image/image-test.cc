#include "tempest/utils/testing.hh"
#include "tempest/image/exr-image.hh"
#include "tempest/image/tga-image.hh"
#include "tempest/image/png-image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/threads.hh"
#include "tempest/math/hdr.hh"
#include "tempest/image/eps-draw.hh"
#include "tempest/image/image-process.hh"
#include <cstring>

const float ExposureModifier = 0.18f;
const uint32_t ChunkSize = 64;

TGE_TEST("Testing the image loading capabilities")
{
    std::unique_ptr<Tempest::Texture> tex(Tempest::LoadEXRImage(Tempest::Path(ROOT_SOURCE_DIR "/external/tinyexr/asakusa.exr")));

    Tempest::ThreadPool pool;

    auto id = pool.allocateThreadNumber();

    auto& hdr = tex->getHeader();
    auto width = hdr.Width,
         height = hdr.Height;
    uint32_t area = width*height;

    auto* tex_ptr = tex.get();
    auto& tex_hdr = tex_ptr->getHeader();

    auto channel_count = Tempest::DataFormatChannels(tex_hdr.Format);
    TGE_CHECK(channel_count == 3 || channel_count == 4, "Unsupported texture");
    
    std::unique_ptr<Tempest::Texture> tga_tex(Tempest::ParallelConvertHDRToSRGB(id, pool, tex.get(), ChunkSize, ExposureModifier));

    bool status = Tempest::SaveTGAImage(tga_tex->getHeader(), tga_tex->getData(), Tempest::Path("test.tga"));
    TGE_CHECK(status, "Failed to save to TGA file");

    {
        std::unique_ptr<Tempest::Texture> after_save_tga(Tempest::LoadTGAImage(Tempest::Path("test.tga")));
        TGE_CHECK(after_save_tga, "Failed to load TGA file");

        auto* data0 = reinterpret_cast<uint32_t*>(tga_tex->getData());
        auto* data1 = reinterpret_cast<uint32_t*>(after_save_tga->getData());
        for(uint32_t pixel_idx = 0; pixel_idx < area; ++pixel_idx)
        {
            TGE_CHECK(data0[pixel_idx] == data1[pixel_idx], "Invalid saved data");
        }
    }

    status = Tempest::SaveEXRImage(tex->getHeader(), tex->getData(), Tempest::Path("test.exr"));
    TGE_CHECK(status, "Failed to save to EXR file");

    {
        std::unique_ptr<Tempest::Texture> after_save_exr(Tempest::LoadEXRImage(Tempest::Path("test.exr")));
        TGE_CHECK(after_save_exr, "Failed to load EXT file");

        auto* data0 = reinterpret_cast<Tempest::Vector4*>(tex->getData());
        auto* data1 = reinterpret_cast<Tempest::Vector4*>(after_save_exr->getData());
        for(uint32_t pixel_idx = 0; pixel_idx < area; ++pixel_idx)
        {
            TGE_CHECK(data0[pixel_idx] == data1[pixel_idx], "Invalid saved data");
        }
    }

    {
    Tempest::TextureDescription test_tex_desc;
    test_tex_desc.Width = 200;
    test_tex_desc.Height = 200;
    test_tex_desc.Format = Tempest::DataFormat::RGB32F;

    std::unique_ptr<Tempest::Vector3[]> tex_data(new Tempest::Vector3[test_tex_desc.Width*test_tex_desc.Height]);

    for(size_t y = 0, yend = test_tex_desc.Height; y < yend; ++y)
        for(size_t x = 0, xend = test_tex_desc.Width; x < xend; ++x)
        {
            Tempest::Vector3 color;
            if(x < test_tex_desc.Width/2)
            {
                color = y < test_tex_desc.Height/2 ?
                            Tempest::Vector3{ 1.0f, 0.0f, 0.0f } :
                            Tempest::Vector3{ 1.0f, 0.0f, 1.0f };
            }
            else
            {
                color = y < test_tex_desc.Height/2 ?
                            Tempest::Vector3{ 0.0f, 1.0f, 0.0f } :
                            Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
            }

            tex_data[y*test_tex_desc.Width + x] = color;
        }

    auto status = Tempest::SaveEXRImage(test_tex_desc, tex_data.get(), Tempest::Path("cross-fade-orig.exr"));
    TGE_CHECK(status, "Failed to save original cross fade texture");

    auto lerp3 = [](const Tempest::Vector3& lhs, const Tempest::Vector3& rhs, float t)
                 {
                     return (1.0f - t)*lhs + t*rhs;
                 };

    status = Tempest::CrossFadeSaveImageTyped<Tempest::Vector3>(test_tex_desc, tex_data.get(), 25, lerp3, Tempest::Path("cross-fade.exr"));
    TGE_CHECK(status, "Failed to save cross fade image");
    }


    status = Tempest::SavePNGImage(tga_tex->getHeader(), tga_tex->getData(), Tempest::Path("test.png"));
    TGE_CHECK(status, "Failed to save to PNG file");

    {
        std::unique_ptr<Tempest::Texture> after_save_png(Tempest::LoadPNGImage(Tempest::Path("test.png")));
        TGE_CHECK(after_save_png, "Failed to load PNG file");

        auto* data0 = reinterpret_cast<uint32_t*>(tga_tex->getData());
        auto* data1 = reinterpret_cast<uint32_t*>(after_save_png->getData());
        for(uint32_t pixel_idx = 0; pixel_idx < area; ++pixel_idx)
        {
            TGE_CHECK(data0[pixel_idx] == data1[pixel_idx], "Invalid saved data");
        }
    }

    auto& tga_tex_hdr = tga_tex->getHeader();
    
    Tempest::EPSImageInfo eps_info;
    eps_info.Width = tga_tex_hdr.Width;
    eps_info.Height = tga_tex_hdr.Height;
    
    Tempest::EPSDraw eps_draw(eps_info);
    eps_draw.drawImage(*tga_tex);

    auto str = eps_draw.get();
    size_t line_chars = 0;
    for(auto& c : str)
    {
        if(c == '\n')
        {
            line_chars = 0;
            continue;
        }
        ++line_chars;
        TGE_CHECK(line_chars < 255, "Too many characters per line");
    }

    status = eps_draw.saveImage("test.eps");
    TGE_CHECK(status, "Failed to save EPS file");
}

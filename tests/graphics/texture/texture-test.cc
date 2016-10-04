#include "tempest/utils/testing.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/image/image.hh"

TGE_TEST("Testing texture capabilities")
{
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = 256;
    tex_desc.Height = 256;
    tex_desc.Format = Tempest::DataFormat::R8UNorm;
    Tempest::Texture tex(tex_desc, new uint8_t[tex_desc.Width*tex_desc.Height]);

    for(uint32_t y = 0; y < tex_desc.Height; ++y)
    {
        for(uint32_t x = 0; x < tex_desc.Width; ++x)
        {
            uint8_t data = static_cast<uint8_t>(255*y/tex_desc.Height);
            tex.writeValue(data, x, y);
            uint8_t fetch_back = static_cast<uint8_t>(tex.fetchRed(x, y)*255.0f + 0.5f);
            TGE_CHECK(data == fetch_back, "Different stored and fetched value");
        }
    }

	std::unique_ptr<Tempest::Texture> orig_image(Tempest::LoadImage(Tempest::Path(TEST_ASSETS_DIR "/hand/rotate.png")));
    auto& orig_hdr = orig_image->getHeader();
	Tempest::TextureDescription upscaled_desc = orig_hdr;
	upscaled_desc.Width *= 8;
	upscaled_desc.Height *= 8;

	Tempest::Texture upscaled_image(upscaled_desc, new uint8_t[upscaled_desc.Width*upscaled_desc.Height*Tempest::DataFormatElementSize(upscaled_desc.Format)]);

	TGE_CHECK(Tempest::DataFormatElementSize(upscaled_desc.Format) == sizeof(uint32_t), "Invalid format");

	for(uint32_t y = 0; y < upscaled_desc.Height; ++y)
		for(uint32_t x = 0; x < upscaled_desc.Width; ++x)
		{
			Tempest::Vector2 tc{ (float)x/(upscaled_desc.Width - 1), (float)y/(upscaled_desc.Height - 1) };
			auto rgba = orig_image->sampleRGBA(tc);
			auto color = Tempest::ToColor(rgba);

			upscaled_image.writeValue(color, x*sizeof(uint32_t), y);
		}

	Tempest::SaveImage(upscaled_desc, upscaled_image.getData(), Tempest::Path("upscale_test.png"));

    Tempest::Texture flipped_tex(*orig_image);
    flipped_tex.flipY();
    flipped_tex.flipY();

    TGE_CHECK(!memcmp(orig_image->getData(), flipped_tex.getData(), orig_hdr.Width*orig_hdr.Height*Tempest::DataFormatElementSize(orig_hdr.Format)),
                    "broken flip operation");
}
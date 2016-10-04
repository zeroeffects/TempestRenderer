#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/display-image.hh"

const uint32_t RandomNumberToGenerate = 100000000;

inline unsigned intel_fastrand(unsigned& mirand)
{
    mirand = (214013*mirand+2531011);
	return (mirand>>16)&0x7FFF;
}

inline float float_intelfastrand(unsigned& mirand)
{
    return (float)intel_fastrand(mirand)/0x7FFF;
}

TGE_TEST("Testing random number generation")
{
    uint32_t image_width = 1000,
                    image_height = image_width;

    std::unique_ptr<uint32_t[]> uniform_random_image(new uint32_t[image_width*image_height]);
    memset(uniform_random_image.get(), 0, image_width*image_height*sizeof(uint32_t));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, image_width - 1);

    unsigned mirand = 1;
    const size_t div = (0x7FFF / (image_width - 1) + 1);
    for(size_t i = 0; i < RandomNumberToGenerate; ++i)
    {
        /*
        size_t x = (size_t)((image_width - 1)*Tempest::sfrand(mirand) + 0.5f);
        size_t y = (size_t)((image_height - 1)*Tempest::sfrand(mirand) + 0.5f);
        //*/

        /*
        size_t x = dis(gen);
        size_t y = dis(gen);
        //*/

        /*
        size_t x = intel_fastrand(mirand) / div;
        size_t y = intel_fastrand(mirand) / div;
        //*/

        //*
        size_t x = (size_t)((image_width - 1)*float_intelfastrand(mirand) + 0.5f);
        size_t y = (size_t)((image_height - 1)*float_intelfastrand(mirand) + 0.5f);
        //*/

        uniform_random_image[y*image_width + x] += 1;
    }

    uint32_t max_value = 0;
    for(size_t y = 0; y < image_height; ++y)
    {
        for(size_t x = 0; x < image_width; ++x)
        {
            uint32_t cur_value = uniform_random_image[y*image_width + x];
            if(max_value < cur_value)
                max_value = cur_value;
        }
    }

    for(size_t y = 0; y < image_height; ++y)
    {
        for(size_t x = 0; x < image_width; ++x)
        {
            uint32_t cur_value = uniform_random_image[y*image_width + x];
            uniform_random_image[y*image_width + x] = Tempest::ToColor(Tempest::ToVector3((float)cur_value/max_value));
        }
    }


    Tempest::TextureDescription tex_desc;
    tex_desc.Width = image_width;
    tex_desc.Height = image_height;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    Tempest::SaveImage(tex_desc, uniform_random_image.get(), Tempest::Path("random.tga"));

    Tempest::DisplayImage(tex_desc, uniform_random_image.get());
}
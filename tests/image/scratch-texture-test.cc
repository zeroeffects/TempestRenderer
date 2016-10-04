#include "tempest/utils/testing.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/logging.hh"
#include "tempest/graphics/scratch-texture.hh"
#include <cstring>

TGE_TEST("Testing scratch-texture data structure")
{
    Tempest::ScratchTextureDescription tex_desc;
    tex_desc.Width = 32;
    tex_desc.Height = 32;
    tex_desc.Layers = 128;
    tex_desc.Scale = 10000.0f;
    Tempest::ScratchTexture scratchTex(tex_desc);

    uint32_t seed = 1234;
    uint32_t texelSamples = 10;

    uint32_t multi_x = Tempest::FastUintRand(0, tex_desc.Width, seed);
    uint32_t multi_y = Tempest::FastUintRand(0, tex_desc.Height, seed);
    uint32_t multi_samples = 12;

    for (uint32_t i = 0; i < texelSamples; i++)
    {
        uint32_t rand_x = Tempest::FastUintRand(0, tex_desc.Width, seed);
        uint32_t rand_y = Tempest::FastUintRand(0, tex_desc.Height, seed);

        if ((rand_y == multi_y) && (rand_x == multi_x)) {
            i -=1;
            continue;
        }

        Tempest::Scratch scr;
        scratchTex.addScratch(scr, rand_x, rand_y);
        Tempest::Log(Tempest::LogLevel::Info, "Created scratch at ", rand_x, ", ", rand_y, " ", rand_y * tex_desc.Width + rand_x);
    }
    Tempest::Log(Tempest::LogLevel::Info, "Creating multiple scratches at ", multi_x, ", ", multi_y);
    for (uint32_t j = 0; j < multi_samples; j++)
    {
        Tempest::Scratch scr;
        scr.Depth = (float) j;
        scr.Length = (float) j + 1.0f;
        scr.Width = (float) j;
        scr.Origin = {j * multi_x, j * multi_y};
        scr.Tangent = {j * multi_x, j * multi_y};
        scratchTex.addScratch(scr, multi_x, multi_y);
        Tempest::Log(Tempest::LogLevel::Info, "Created scratch at ", multi_x, ", ", multi_y);
    }

    int scr_count = 0;

    for (uint32_t y = 0; y < tex_desc.Height; y++)
    {
        for (uint32_t x = 0; x < tex_desc.Width; x++)
        {
            const Tempest::Scratch* curTex = scratchTex.fetchTexel(x, y);
            for (int scr = 0; scr < tex_desc.Layers; scr++)
            {
                if (!isScratch(curTex[scr])) continue;
                Tempest::Log(Tempest::LogLevel::Info, "Found scratch at ", x, ", ", y);
                scr_count +=1;
                if ((y == multi_y) && (x == multi_x))
                {
                    TGE_CHECK(curTex[scr].Depth == (float)scr, "Data not conserved");
                    TGE_CHECK(curTex[scr].Width == (float)scr, "Data not conserved");
                    TGE_CHECK(curTex[scr].Length == (float)scr +1.0f, "Data not conserved");
                    TGE_CHECK(curTex[scr].Origin.x == (float)scr * multi_x, "Data not conserved");
                    TGE_CHECK(curTex[scr].Origin.y == (float)scr * multi_y, "Data not conserved");
                    TGE_CHECK(curTex[scr].Tangent.x == (float)scr * multi_x, "Data not conserved");
                    TGE_CHECK(curTex[scr].Tangent.y == (float)scr * multi_y, "Data not conserved");
                }

            }
        }
    }
    TGE_CHECK(scr_count == (texelSamples + multi_samples), "Lost some scratches on the way");
    TGE_CHECK(scr_count == scratchTex.getScratchCount(), "getScratchCount not working properly");

    scratchTex.clear();

    TGE_ASSERT(scratchTex.getScratchCount() == 0, "Clear did not remove all scratches");

    Tempest::Log(Tempest::LogLevel::Info, "Testing line creation capabilities");
    Tempest::Scratch scr;
    scr.Depth = 2000.0f;
    scr.Length = Tempest::FastFloatRand(seed) * tex_desc.Width * tex_desc.Scale;
    scr.Width = 3000.0f;
    scr.Origin = {0.f, 0.f};
    scr.Tangent = Tempest::Normalize(Tempest::Vector2{0.7, 0.7});
    scratchTex.addLine(scr, 0, 1);
    TGE_ASSERT(scratchTex.maxCountPerTexel() == 1, "Too many scratches per texel");

    scratchTex.clear();
    scr.Tangent = {1.0f, 0.0f};
    scr.Length = tex_desc.Scale * tex_desc.Width;
    scratchTex.addLine(scr, 0, 0);

    Tempest::Log(Tempest::LogLevel::Info, "Line generated with ",scratchTex.getScratchCount()," scratches");
    for (uint32_t y = 0; y < tex_desc.Height; y++)
    {
        for (uint32_t x = 0; x < tex_desc.Width; x++)
        {
            const Tempest::Scratch* curTex = scratchTex.fetchTexel(x, y);
            for (int scr = 0; scr < tex_desc.Layers; scr++)
            {
                if (!isScratch(curTex[scr])) continue;
                Tempest::Log(Tempest::LogLevel::Info, "Found scratch at ", x, ", ", y , " with length ", curTex[scr].Length);
            }
        }
    }
    
    Tempest::Log(Tempest::LogLevel::Info, "Testing circle creation capabilities");
    scratchTex.clear();
    scr.Depth = 2000.f;
    scr.Length = 10000.f;
    scr.Width = 3000.f;
    scr.Origin = {0.f, 0.f};
    scr.Tangent = {0.f, 0.f};


    scratchTex.addCircle(scr, tex_desc.Width/2, tex_desc.Height/2,
                         Tempest::FastFloatRand(seed) * tex_desc.Width/2.0f * tex_desc.Scale);

    for (uint32_t y = 0; y < tex_desc.Height; y++)
    {
        for (uint32_t x = 0; x < tex_desc.Width; x++)
        {
            const Tempest::Scratch* curTex = scratchTex.fetchTexel(x, y);
            for (int scr = 0; scr < tex_desc.Layers; scr++)
            {
                if (!isScratch(curTex[scr])) continue;
                Tempest::Log(Tempest::LogLevel::Info, "Found scratch at ", x, ", ", y , " with length ", curTex[scr].Length);
            }
        }
    }
    TGE_CHECK(scratchTex.maxCountPerTexel() == 1, "Too many scratches per texel");
}

#include "tempest/utils/testing.hh"
#include "tempest/utils/threads.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/math/hdr.hh"
#include "tempest/image/image.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/timer.hh"
#include "tempest/image/btf.hh"

#define PARALLEL_DECODE

const uint32_t BTFTestSamples = 1024;

TGE_TEST("Testing BTF loading")
{
    Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(CURRENT_SOURCE_DIR "/fabric09_resampled_W400xH400_L151xV151.btf")));
	TGE_CHECK(btf, "Failed to load BTF");

	auto btf_width = btf->Width;
    auto btf_height = btf->Height;
    auto btf_view_count = btf->LightCount;
    auto btf_light_count = btf->LightCount;

    {
    Tempest::BTFPtr cut_btf(Tempest::CutBTF(btf.get(), 0, 0, btf->Width, btf->Height));
    TGE_ASSERT(cut_btf->LeftSingularUSize == btf->LeftSingularUSize, "Invalid U matrix");
    TGE_ASSERT(cut_btf->RightSingularSxVSize == btf->RightSingularSxVSize, "Invalid SxV matrix");
    TGE_ASSERT(!memcmp(btf->LeftSingularU, cut_btf->LeftSingularU, btf->LeftSingularUSize), "Invalid copy");
    TGE_ASSERT(!memcmp(btf->RightSingularSxV, cut_btf->RightSingularSxV, btf->RightSingularSxVSize), "Invalid copy");
    }

    {
        uint32_t start_x = 2, start_y = 3,
                 width = 5, height = 7;
    
        Tempest::BTFPtr cut_btf(Tempest::CutBTF(btf.get(), start_x, start_y, width, height));
        for(uint32_t y = 0, yend = height; y < yend; ++y)
        {
            for(uint32_t x = 0, xend = width; x < xend; ++x)
            {
                for(uint32_t v = 0; v < btf_view_count; ++v)
                {
                    for(uint32_t l = 0; l < btf_light_count; ++l)
                    {
                        auto orig_color = Tempest::BTFFetchSpectrum(btf.get(), l, v, start_x + x, start_y + y);
                        auto cut_color = Tempest::BTFFetchSpectrum(cut_btf.get(), l, v, x, y);
                        TGE_ASSERT(orig_color == cut_color, "Invalid cut procedure");
                    }
                }
            }
        }
    }

	/*
	std::unique_ptr<uint32_t[]> dbg_in_tri(new uint32_t[btf_light_count]);
    memset(dbg_in_tri.get(), 0, btf_light_count*sizeof(float));

	for(uint32_t light_idx = 0, light_idx_end = btf->LightTriangleCount; light_idx < light_idx_end;)
    {
        auto i0 = btf->LightIndices[light_idx++];
        auto i1 = btf->LightIndices[light_idx++];
        auto i2 = btf->LightIndices[light_idx++];
		
        dbg_in_tri[i0]++;
        dbg_in_tri[i1]++;
        dbg_in_tri[i2]++;
    }   

	for(uint32_t idx = 0; idx < btf_light_count; ++idx)
        TGE_CHECK(dbg_in_tri[idx], "Some light directions are unused in BTF!!!");
	*/

	uint32_t upper_bound_tree_size = (2*btf->LightTriangleCount - 1);
	for(uint32_t node_idx = 0; node_idx < upper_bound_tree_size; ++node_idx)
	{
		auto& node = btf->LightBVH[node_idx];
		if(node.Patch & LBVH_LEAF_DECORATION)
		{
			auto prim_id = node.Patch & ~LBVH_LEAF_DECORATION;
			TGE_CHECK(prim_id < btf->LightTriangleCount, "Invalid index");
		}
		else
		{
			TGE_CHECK(node.Child2 < upper_bound_tree_size && node_idx + 1 != upper_bound_tree_size, "Out of bounds tree nodes");
		}
	}

	std::unique_ptr<float[]> weights(ComputeLightViewSampleWeights(btf.get()));

	float total_probability = 1e-3f;
	for(uint32_t idx = 0, idx_end = btf_light_count*btf_view_count; idx < idx_end; ++idx)
	{
        total_probability += weights[idx];
	}

	TGE_CHECK(Tempest::ApproxEqual(total_probability, 1.0f, 1e-3f), "Invalid probabilities");
	
    /*
    // TODO: Incomplete triangulation
    {
    auto chan_count = btf->ChannelCount;
    auto* result_single = TGE_TYPED_ALLOCA(float, chan_count);
    auto* result_simd = TGE_TYPED_ALLOCA(float, chan_count);


    unsigned seed = 1;
    for(uint32_t sample = 0; sample < BTFTestSamples; ++sample)
    {
        uint32_t x = Tempest::FastUintRand(0, btf_width, seed);
        uint32_t y = Tempest::FastUintRand(0, btf_height, seed);
        uint32_t l = Tempest::FastUintRand(0, btf_light_count, seed);
        uint32_t v = Tempest::FastUintRand(0, btf_view_count, seed);

        TGE_CHECK(x < btf_width && y < btf_height && l < btf_light_count && v < btf_view_count, "Invalid indices");

		auto light = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[l]),
			 view = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[v]);

		uint32_t light_prim, view_prim;
		Tempest::Vector3 light_barycentric, view_barycentric;
		auto intersect = BTFFetchLightViewDirection(btf.get(), light, view, &light_prim, &light_barycentric, &view_prim, &view_barycentric);
		TGE_CHECK(intersect, "Bad intersection");
		bool found_idx = false;
		for(uint32_t i = 0; i < 3; ++i)
		{
			auto idx = btf->LightIndices[light_prim*3 + i];
			if(idx == l)
			{
				found_idx = true;
				break;
			}
		}
		TGE_CHECK(found_idx, "Broken intersection");

		found_idx = false;
		for(uint32_t i = 0; i < 3; ++i)
		{
			auto idx = btf->LightIndices[light_prim*3 + i];
			if(idx == v)
			{
				found_idx = true;
				break;
			}
		}
		TGE_CHECK(found_idx, "Broken intersection");

        uint32_t lv_idx = v*btf_light_count + l;
        uint32_t xy_idx = y*btf_width + x;
        BTFFetchChannelsSingleFloat(btf.get(), lv_idx, xy_idx, &result_single);
    #ifndef LINUX
        BTFFetchChannelsSIMD(btf.get(), lv_idx, xy_idx, &result_simd);
        for(uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
            TGE_CHECK(Tempest::ApproxEqual(result_single[chan_idx], result_simd[chan_idx], 1e-5f), "Broken implementation");
    #endif
    }
    }
    */

    const float angle = Tempest::ToRadians(45.0f);
    float sin_theta, cos_theta;
    Tempest::FastSinCos(angle, &sin_theta, &cos_theta);

    Tempest::Vector3 view{ 0.0f, -sin_theta, cos_theta },
                     light{ 0.0f, sin_theta, cos_theta };

    auto spec_direct = BTFFetchSpectrum(btf.get(), 48, 56, 0, btf->Height - 1);
    auto rgb = Tempest::SpectrumToRGB(spec_direct);
    TGE_CHECK(Tempest::ApproxEqual(rgb, Tempest::Vector3{ 0.0225f, 0.0301, 0.0264f }, 1e-3f), "Invalid spectrum");

    auto spec_interpolate = BTFFetchPixelSampleLightViewSpectrum(btf.get(), light, view, 0, 0);

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = btf_width;
    tex_desc.Height = btf_height;
    tex_desc.Format = Tempest::DataFormat::RGB32F;

	Tempest::TimeQuery timer;
	auto start_time = timer.time();

    auto area = tex_desc.Width*tex_desc.Height;
    std::unique_ptr<Tempest::Vector3[]> data(new Tempest::Vector3[area]);
	auto data_ptr = data.get();
	auto width = tex_desc.Width;


    Tempest::Vector3 light_barycentric, view_barycentric;
    uint32_t light_prim_id, view_prim_id;
    auto intersect = BTFFetchLightViewDirection(btf.get(), light, view, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
    TGE_CHECK(intersect, "Failed to get direction information");

	auto btf_ptr = btf.get();
    
#ifdef PARALLEL_DECODE
	auto parallel_decode = Tempest::CreateParallelForLoop2D(tex_desc.Width, tex_desc.Height, 64,
															[data_ptr, width, btf_ptr, &light, &view,
                                                             &light_barycentric, &view_barycentric,
                                                             light_prim_id, view_prim_id](uint32_t worker_id, uint32_t x, uint32_t y)
#else
	for(uint32_t y = 0; y < tex_desc.Height; ++y)
		for(uint32_t x = 0; x < tex_desc.Width; ++x)
#endif
															{
																uint32_t idx = y*width + x;
																data_ptr[idx] = Tempest::SpectrumToRGB(BTFFetchPixelSampleLightViewSpectrum(btf_ptr, light_prim_id, light_barycentric, view_prim_id, view_barycentric, x, y));
														    }
#ifdef PARALLEL_DECODE
														    );
	
	pool.enqueueTask(&parallel_decode);
#endif

    Tempest::Texture hdr_texture(tex_desc, reinterpret_cast<uint8_t*>(data.release()));

#ifdef PARALLEL_DECODE
	pool.waitAndHelp(id, &parallel_decode);
#endif

	auto elapsed_time = timer.time() - start_time;

	Tempest::Log(Tempest::LogLevel::Info, "BTF decode time: ", elapsed_time, "us");

    std::unique_ptr<Tempest::Texture> tga_tex(Tempest::ParallelConvertHDRToSRGB(id, pool, &hdr_texture, 64));

    Tempest::SaveImage(tga_tex->getHeader(), tga_tex->getData(), Tempest::Path("btf-slice.tga"));
}

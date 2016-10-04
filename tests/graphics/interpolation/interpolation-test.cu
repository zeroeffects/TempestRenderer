#include "tempest/utils/testing.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/compute/compute-texture.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/image/image.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/graphics/custom-samplers.hh"

const uint32_t UpsamplingCoefficient = 8,
               RepeatCoeffient = 2,
               DataWidth = 4,
               DataHeight = 4,
               QuadtreeTextureWidth = 16,
               QuadtreeTextureHeight = 16,
               FocusSize = 64;

#include <cuda_runtime_api.h>
#include <texture_types.h>

Tempest::Quaternion SampleQuaternionSlerpFetch(void* tex, uint32_t tex_width, uint32_t tex_height, const Tempest::Vector2& tc)
{
    Tempest::Vector2 tc_unorm{ tex_width*tc.x - 0.5f, tex_height*tc.y - 0.5f };

    int x_trunc = Tempest::FastFloorToInt(tc_unorm.x),
        y_trunc = Tempest::FastFloorToInt(tc_unorm.y);

    uint32_t x0 = static_cast<uint32_t>(Tempest::Modulo(x_trunc, (int)tex_width));
    uint32_t y0 = static_cast<uint32_t>(Tempest::Modulo(y_trunc, (int)tex_height));
    uint32_t x1 = (x0 + 1) % tex_width;
    uint32_t y1 = (y0 + 1) % tex_height;

    float tx = tc_unorm.x - (float)x_trunc,
          ty = tc_unorm.y - (float)y_trunc;

    Tempest::Vector4 q00 = Tempest::FetchRGBA(tex, x0, y0),
                     q01 = Tempest::FetchRGBA(tex, x1, y0),
                     q10 = Tempest::FetchRGBA(tex, x0, y1),
                     q11 = Tempest::FetchRGBA(tex, x1, y1);

    return Tempest::Normalize(Slerp(Slerp(reinterpret_cast<Tempest::Quaternion&>(q00), reinterpret_cast<Tempest::Quaternion&>(q01), tx),
                                    Slerp(reinterpret_cast<Tempest::Quaternion&>(q10), reinterpret_cast<Tempest::Quaternion&>(q11), tx), ty));;
}

__global__ void UpsampleTexture(cudaTextureObject_t tex, uint32_t width, uint32_t height, Tempest::Vector4* out_texture)
{
     auto x = blockIdx.x * blockDim.x + threadIdx.x;
     auto y = blockIdx.y * blockDim.y + threadIdx.y;

     if(x >= width || y >= height)
         return;

     Tempest::Vector2 tc{ (float)x/(width - 1), (float)y/(height - 1) };

     auto value = Tempest::SampleRGBA(reinterpret_cast<void*>(tex), tc);

     out_texture[y*width + x] = value;
}

__device__ __host__ Tempest::Vector4 ManualBilinearSampling(void* tex, uint32_t tex_width, uint32_t tex_height, const Tempest::Vector2& tc)
{
    float tex_width_f = static_cast<float>(tex_width);
    float tex_height_f = static_cast<float>(tex_height);

    Tempest::Vector2 tc_unorm{ tex_width_f*tc.x - 0.5f, tex_height_f*tc.y - 0.5f };

    Tempest::Vector2 tc_unorm_trunc{ fmodf(floorf(tc_unorm.x), tex_width_f), fmodf(floorf(tc_unorm.y), tex_height_f) };

    float fx1 = tc_unorm.x - tc_unorm_trunc.x,
         fx0 = 1.0f - fx1,
         fy1 = tc_unorm.y - tc_unorm_trunc.y,
         fy0 = 1.0f - fy1;

    Tempest::Vector2 tc_norm_trunc_shift{ (tc_unorm_trunc.x + 1.0f)/tex_width_f, (tc_unorm_trunc.y + 1.0f)/tex_height_f };

    Tempest::Vector4 out_texture;

    {
    auto comp0 = Gather2D(tex, tc_norm_trunc_shift, 0);
     
    out_texture.x = (fx0 * comp0.w + fx1 * comp0.z) * fy0 +
                    (fx0 * comp0.x + fx1 * comp0.y) * fy1;
    }

    {
    auto comp1 = Gather2D(tex, tc_norm_trunc_shift, 1);
     
    out_texture.y = (fx0 * comp1.w + fx1 * comp1.z) * fy0 +
                    (fx0 * comp1.x + fx1 * comp1.y) * fy1;
    }

    {
    auto comp2 = Gather2D(tex, tc_norm_trunc_shift, 2);
     
    out_texture.z = (fx0 * comp2.w + fx1 * comp2.z) * fy0 +
                    (fx0 * comp2.x + fx1 * comp2.y) * fy1;
    }

    {
    auto comp3 = Gather2D(tex, tc_norm_trunc_shift, 3);

    out_texture.w = (fx0 * comp3.w + fx1 * comp3.z) * fy0 +
                    (fx0 * comp3.x + fx1 * comp3.y) * fy1;
    }

    return out_texture;
}

__global__ void UpsampleGatherTexture(cudaTextureObject_t tex, uint32_t tex_width, uint32_t tex_height, uint32_t out_width, uint32_t out_height, Tempest::Vector4* out_texture)
{
     auto x = blockIdx.x * blockDim.x + threadIdx.x;
     auto y = blockIdx.y * blockDim.y + threadIdx.y;

     if(x >= out_width || y >= out_height)
         return;

     Tempest::Vector2 tc{ (float)x/(out_width - 1), (float)y/(out_height - 1) };

     size_t out_idx = y*out_width + x;
     out_texture[out_idx] = ManualBilinearSampling(reinterpret_cast<void*>(tex), tex_width, tex_height, tc);
}

__device__ __host__ Tempest::Vector4 Gather0(void* tex, uint32_t tex_width, uint32_t tex_height, const Tempest::Vector2& tc)
{
    float tex_width_f = static_cast<float>(tex_width);
    float tex_height_f = static_cast<float>(tex_height);

    Tempest::Vector2 tc_unorm{ tex_width_f*tc.x - 0.5f, tex_height_f*tc.y - 0.5f };
    Tempest::Vector2 tc_unorm_trunc{ fmodf(floorf(tc_unorm.x), tex_width_f), fmodf(floorf(tc_unorm.y), tex_height_f) };

    Tempest::Vector2 tc_norm_trunc_shift{ (tc_unorm_trunc.x + 1.0f)/tex_width_f, (tc_unorm_trunc.y + 1.0f)/tex_height_f };

    return Gather2D(tex, tc_norm_trunc_shift, 0);
}

__global__ void GatherSingle(cudaTextureObject_t tex, uint32_t tex_width, uint32_t tex_height, uint32_t out_width, uint32_t out_height, Tempest::Vector4* out_texture)
{
     auto x = blockIdx.x * blockDim.x + threadIdx.x;
     auto y = blockIdx.y * blockDim.y + threadIdx.y;

     if(x >= out_width || y >= out_height)
         return;

     Tempest::Vector2 tc{ (float)x/(out_width - 1), (float)y/(out_height - 1) };

     size_t out_idx = y*out_width + x;
     out_texture[out_idx] = Gather0(reinterpret_cast<void*>(tex), tex_width, tex_height, tc);
}


TGE_TEST("Testing different intepolation techniques")
{
    Tempest::TextureDescription tex_desc;
    tex_desc.Width = DataWidth;
    tex_desc.Height = DataHeight;
    tex_desc.Format = Tempest::DataFormat::RGBA32F;
    uint32_t tex_area = tex_desc.Width*tex_desc.Height;

    Tempest::Vector4* tex_data = new Tempest::Vector4[tex_area];
    Tempest::Texture tex(tex_desc, reinterpret_cast<uint8_t*>(tex_data));

    for(uint32_t y = 0; y < DataHeight; ++y)
        for(uint32_t x = 0; x < DataWidth; ++x)
        {
            tex_data[y*DataWidth + x] = { (float)x/(DataWidth - 1), (float)y/(DataHeight - 1), 0.0f, 1.0f };
        }

    uint32_t upsampled_height = UpsamplingCoefficient*tex_desc.Height,
             upsampled_width = UpsamplingCoefficient*tex_desc.Width;
    auto upsampled_area = upsampled_width*upsampled_height;
    auto cuda_tex = CREATE_SCOPED(cudaTextureObject_t, Tempest::CudaTextureDeleter);
    cuda_tex = Tempest::CreateCudaTexture(&tex, Tempest::TEMPEST_CUDA_TEXTURE_GATHER);

    auto cuda_upsampled_data = CREATE_SCOPED(Tempest::Vector4*, ::cudaFree);
    auto status = cudaMalloc(reinterpret_cast<void**>(&cuda_upsampled_data), upsampled_area*sizeof(cuda_upsampled_data[0]));
    TGE_CHECK(status == cudaSuccess, "Failed to allocate CUDA backbuffer data");


    std::unique_ptr<Tempest::Vector4[]> upsampled_data(new Tempest::Vector4[upsampled_area]);
    for(uint32_t y = 0; y < upsampled_height; ++y)
    {
        for(uint32_t x = 0; x < upsampled_width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(upsampled_width - 1), (float)y/(upsampled_height - 1) };
            upsampled_data[y*upsampled_width + x] = Tempest::SampleRGBA(&tex, tc);
        }
    }

    std::unique_ptr<Tempest::Vector4[]> cpu_gather_data(new Tempest::Vector4[upsampled_area]);
    for(uint32_t y = 0; y < upsampled_height; ++y)
    {
        for(uint32_t x = 0; x < upsampled_width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(upsampled_width - 1), (float)y/(upsampled_height - 1) };

            size_t out_idx = y*upsampled_width + x;
            cpu_gather_data[out_idx] = Gather0(reinterpret_cast<void*>(&tex), tex_desc.Width, tex_desc.Height, tc);
        }
    }

    dim3 up_group_size(8, 8, 1);
    dim3 up_thread_groups((upsampled_width + up_group_size.x - 1)/up_group_size.x, (upsampled_height + up_group_size.y - 1)/up_group_size.y, 1);

    GatherSingle<<<up_thread_groups, up_group_size>>>(cuda_tex.get(), tex_desc.Width, tex_desc.Height, upsampled_width, upsampled_height, cuda_upsampled_data.get());
    cudaThreadSynchronize();
    status = cudaGetLastError();
    TGE_CHECK(status == cudaSuccess, "Failed to launch kernel");

    std::unique_ptr<Tempest::Vector4[]> copy_upsampled_data(new Tempest::Vector4[upsampled_area]);
    status = cudaMemcpy(copy_upsampled_data.get(), cuda_upsampled_data.get(), upsampled_area*sizeof(cuda_upsampled_data[0]), cudaMemcpyDeviceToHost);
    TGE_CHECK(status == cudaSuccess, "Failed to copy upsampled data");

    for(uint32_t idx = 0; idx < upsampled_area; ++idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(cpu_gather_data[idx], copy_upsampled_data[idx], 1e-2f), "Invalid interpolation");
    }

    std::unique_ptr<Tempest::Vector4[]> gather_upsampled_data(new Tempest::Vector4[upsampled_area]);
    for(uint32_t y = 0; y < upsampled_height; ++y)
    {
        for(uint32_t x = 0; x < upsampled_width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(upsampled_width - 1), (float)y/(upsampled_height - 1) };

            size_t out_idx = y*upsampled_width + x;
            gather_upsampled_data[out_idx] = ManualBilinearSampling(reinterpret_cast<void*>(&tex), tex_desc.Width, tex_desc.Height, tc);
        }
    }

    for(uint32_t idx = 0; idx < upsampled_area; ++idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(upsampled_data[idx], gather_upsampled_data[idx], 1e-2f), "Invalid interpolation");
    }

    UpsampleTexture<<<up_thread_groups, up_group_size>>>(cuda_tex.get(), upsampled_width, upsampled_height, cuda_upsampled_data.get());
    cudaThreadSynchronize();
    status = cudaGetLastError();
    TGE_CHECK(status == cudaSuccess, "Failed to launch kernel");

    status = cudaMemcpy(copy_upsampled_data.get(), cuda_upsampled_data.get(), upsampled_area*sizeof(cuda_upsampled_data[0]), cudaMemcpyDeviceToHost);
    TGE_CHECK(status == cudaSuccess, "Failed to copy upsampled data");

    for(uint32_t idx = 0; idx < upsampled_area; ++idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(upsampled_data[idx], copy_upsampled_data[idx], 1e-2f), "Invalid interpolation");
    }

    UpsampleGatherTexture<<<up_thread_groups, up_group_size>>>(cuda_tex.get(), tex_desc.Width, tex_desc.Height, upsampled_width, upsampled_height, cuda_upsampled_data.get());
    cudaThreadSynchronize();
    status = cudaGetLastError();
    TGE_CHECK(status == cudaSuccess, "Failed to launch kernel");

    status = cudaMemcpy(copy_upsampled_data.get(), cuda_upsampled_data.get(), upsampled_area*sizeof(cuda_upsampled_data[0]), cudaMemcpyDeviceToHost);
    TGE_CHECK(status == cudaSuccess, "Failed to copy upsampled data");

    for(uint32_t idx = 0; idx < upsampled_area; ++idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(upsampled_data[idx], copy_upsampled_data[idx], 1e-2f), "Invalid interpolation");
    }

    Tempest::Vector3 interpolation_cells[] =
    {
        { 0.0f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
        { 0.1f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
        { -0.1f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
        { 0.1f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
    };

    const uint32_t interp_cells_count = TGE_FIXED_ARRAY_SIZE(interpolation_cells);

    Tempest::Quaternion quat_interpolation_cells[interp_cells_count];

    for(uint32_t idx = 0; idx < interp_cells_count; ++idx)
    {
        auto& interp = interpolation_cells[idx];
        quat_interpolation_cells[idx] = Tempest::ToQuaternionNormal(Tempest::Normalize(interp));
    }

    size_t rows = interp_cells_count/2;

    Tempest::TextureDescription interp_tex_desc;
    interp_tex_desc.Width = (uint16_t)(2*UpsamplingCoefficient);
    interp_tex_desc.Height = (uint16_t)(rows*UpsamplingCoefficient);
    interp_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    uint32_t* quat_interp_tex_data = new uint32_t[interp_tex_desc.Width*interp_tex_desc.Height];

    Tempest::Texture quat_interp_tex(interp_tex_desc, reinterpret_cast<uint8_t*>(quat_interp_tex_data));

    for(size_t y = 0; y < interp_tex_desc.Height; ++y)
    {
        for(size_t x = 0; x < interp_tex_desc.Width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(interp_tex_desc.Width - 1), (float)y/(interp_tex_desc.Height - 1) };

            size_t width = 2;
            size_t height = rows;

			Tempest::Vector2 tc_unorm{ (width - 1)*tc.x - 0.5f, (height - 1)*tc.y };

            size_t x0 = static_cast<size_t>(Tempest::FastFloorToInt64(tc_unorm.x));
            size_t y0 = static_cast<size_t>(Tempest::FastFloorToInt64(tc_unorm.y));
            size_t x1 = Mini((x0 + 1), width - 1);
            size_t y1 = Mini((y0 + 1), height - 1);

            auto& c00 = quat_interpolation_cells[y0*width + x0];
            auto& c01 = quat_interpolation_cells[y0*width + x1];
            auto& c10 = quat_interpolation_cells[y1*width + x0];
            auto& c11 = quat_interpolation_cells[y1*width + x1];

            float fx1 = tc_unorm.x - FastFloor(tc_unorm.x),
                  fx0 = 1.0f - fx1,
                  fy1 = tc_unorm.y - FastFloor(tc_unorm.y),
                  fy0 = 1.0f - fy1;

            auto interp_quat = Tempest::Normalize((fx0 * c00 + fx1 * c01) * fy0 +
                                                  (fx0 * c10 + fx1 * c11) * fy1);

            quat_interp_tex_data[y*interp_tex_desc.Width + x] = Tempest::ToColor(Tempest::ToNormal(interp_quat)*0.5f + 0.5f);
        }
    }

    Tempest::SaveImage(interp_tex_desc, reinterpret_cast<uint8_t*>(quat_interp_tex_data), Tempest::Path("test-interpolation-lerp.png"));

    for(size_t y = 0; y < interp_tex_desc.Height; ++y)
    {
        for(size_t x = 0; x < interp_tex_desc.Width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(interp_tex_desc.Width - 1), (float)y/(interp_tex_desc.Height - 1) };

            size_t width = 2;
            size_t height = rows;

			Tempest::Vector2 tc_unorm{ (width - 1)*tc.x - 0.5f, (height - 1)*tc.y };

            size_t x0 = static_cast<size_t>(Tempest::FastFloorToInt64(tc_unorm.x));
            size_t y0 = static_cast<size_t>(Tempest::FastFloorToInt64(tc_unorm.y));
            size_t x1 = Mini((x0 + 1), width - 1);
            size_t y1 = Mini((y0 + 1), height - 1);

            auto& c00 = quat_interpolation_cells[y0*width + x0];
            auto& c01 = quat_interpolation_cells[y0*width + x1];
            auto& c10 = quat_interpolation_cells[y1*width + x0];
            auto& c11 = quat_interpolation_cells[y1*width + x1];

            float tx = tc_unorm.x - FastFloor(tc_unorm.x),
                  ty = tc_unorm.y - FastFloor(tc_unorm.y);            

            auto interp_quat = Tempest::Normalize(Tempest::Slerp(Tempest::Slerp(c00, c01, tx),
                                                                 Tempest::Slerp(c10, c11, tx), ty));

            quat_interp_tex_data[y*interp_tex_desc.Width + x] = Tempest::ToColor(Tempest::ToNormal(interp_quat)*0.5f + 0.5f);
        }
    }

    Tempest::SaveImage(interp_tex_desc, reinterpret_cast<uint8_t*>(quat_interp_tex_data), Tempest::Path("test-interpolation-slerp.png"));

    Tempest::TextureDescription quadtree_tex_desc;
    quadtree_tex_desc.Width = QuadtreeTextureWidth;
    quadtree_tex_desc.Height = QuadtreeTextureHeight;
    quadtree_tex_desc.Format = Tempest::DataFormat::RGBA32F;
    auto* quadtree_tex_data = new Tempest::Quaternion[quadtree_tex_desc.Width*quadtree_tex_desc.Height];
    Tempest::Texture quadtree_tex(quadtree_tex_desc, reinterpret_cast<uint8_t*>(quadtree_tex_data));

    unsigned seed = 1;
    for(uint32_t y = 0; y < quadtree_tex_desc.Height; ++y)
        for(uint32_t x = 0; x < quadtree_tex_desc.Width; ++x)
        {
            auto dir = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
            quadtree_tex_data[y*quadtree_tex_desc.Width + x] = Tempest::ToQuaternionNormal(dir);
        }

    Tempest::TextureDescription up_quadtree_tex_desc;
    up_quadtree_tex_desc.Width = UpsamplingCoefficient*quadtree_tex_desc.Width;
    up_quadtree_tex_desc.Height = UpsamplingCoefficient*quadtree_tex_desc.Height;
    up_quadtree_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
    auto* up_quadtree_tex_data = new uint32_t[up_quadtree_tex_desc.Width*up_quadtree_tex_desc.Height];
    Tempest::Texture up_quadtree_tex(up_quadtree_tex_desc, reinterpret_cast<uint8_t*>(up_quadtree_tex_data));

    for(uint32_t y = 0; y < up_quadtree_tex_desc.Height; ++y)
        for(uint32_t x = 0; x < up_quadtree_tex_desc.Width; ++x)
        {
            Tempest::Vector2 tc{ (float)x/(up_quadtree_tex_desc.Width - 1), (float)y/(up_quadtree_tex_desc.Height - 1) };

            auto dbg_quat = SampleQuaternionSlerpFetch(&quadtree_tex, quadtree_tex_desc.Width, quadtree_tex_desc.Height, tc);

            auto quat = Tempest::SampleQuaternionSlerp(&quadtree_tex, quadtree_tex_desc.Width, quadtree_tex_desc.Height, tc);

            TGE_CHECK(dbg_quat == quat, "Invalid quaternion sampling");

            up_quadtree_tex_data[y*up_quadtree_tex_desc.Width + x] = Tempest::ToColor(Tempest::ToNormal(quat));
        }

    Tempest::SaveImage(up_quadtree_tex_desc, reinterpret_cast<uint8_t*>(up_quadtree_tex_data), Tempest::Path("test-quaternion-interpolation.png"));

    std::unique_ptr<Tempest::Texture> basis_tex(Tempest::LoadImage(Tempest::Path(ROOT_SOURCE_DIR "/tests/graphics/interpolation/alcantara_center_right_sggx_basis.exr")));
    auto& basis_hdr = basis_tex->getHeader();
    //basis_tex->setSamplingMode(Tempest::TextureSampling::Nearest);

    Tempest::TextureDescription up_basis_tex_desc;
    up_basis_tex_desc.Width = RepeatCoeffient*UpsamplingCoefficient*basis_hdr.Width;
    up_basis_tex_desc.Height = RepeatCoeffient*UpsamplingCoefficient*basis_hdr.Height;
    up_basis_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    uint32_t* up_basis_data = new uint32_t[up_basis_tex_desc.Width*up_basis_tex_desc.Height];
    Tempest::Texture up_basis_tex(up_basis_tex_desc, reinterpret_cast<uint8_t*>(up_basis_data));

    const uint32_t FocusX = 2030;
    const uint32_t FocusY = up_basis_tex_desc.Height - 2300;

    Tempest::TextureDescription focus_patch_tex_desc;
    focus_patch_tex_desc.Width = FocusSize;
    focus_patch_tex_desc.Height = FocusSize;
    focus_patch_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    uint32_t* focus_basis_data = new uint32_t[focus_patch_tex_desc.Width*focus_patch_tex_desc.Height];
    Tempest::Texture focus_basis_tex(focus_patch_tex_desc, reinterpret_cast<uint8_t*>(focus_basis_data));

    for(uint32_t y = 0; y < up_basis_tex_desc.Height; ++y)
    {
        for(uint32_t x = 0; x < up_basis_tex_desc.Width; ++x)
        {
            Tempest::Vector2 tc{ (float)RepeatCoeffient*x/(up_basis_tex_desc.Width - 1), (float)RepeatCoeffient*y/(up_basis_tex_desc.Height - 1) };

            #if 0
                Tempest::Quaternion quat;
                quat.V4 = Tempest::SampleRGBA(basis_tex.get(), tc);
            #else
                Tempest::Quaternion quat = Tempest::SampleQuaternionSlerp(basis_tex.get(), basis_hdr.Width, basis_hdr.Height, tc);

                auto dbg_quat = SampleQuaternionSlerpFetch(basis_tex.get(), basis_hdr.Width, basis_hdr.Height, tc);

                TGE_CHECK(dbg_quat == quat, "Invalid quaternion sampling");
            #endif

            quat = Tempest::Normalize(quat);

            auto color = up_basis_data[y*up_basis_tex_desc.Width + x] = Tempest::ToColor(Tempest::ToTangent(quat)*0.5f + Tempest::Vector3{ 0.5f, 0.5f, 0.5f });

          
            if(FocusX <= x && x < FocusX + FocusSize &&
               FocusY <= y && y < FocusY + FocusSize)
            {
                auto focus_x = x - FocusX,
                     focus_y = y - FocusY;
                  
                focus_basis_data[focus_y*FocusSize + focus_x] = color;
            }
        }
    }

    Tempest::SaveImage(up_basis_tex_desc, up_basis_data, Tempest::Path("alcantara_interpolation_tangent.tga"));
    
    Tempest::SaveImage(focus_patch_tex_desc, focus_basis_data, Tempest::Path("alcantara_interpolation_tangent_focus.tga"));
}

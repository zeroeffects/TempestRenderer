#include "tempest/utils/testing.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/graphics/equirectangular-map.hh"
#include "tempest/graphics/cube-map.hh"
#include "tempest/compute/compute-texture.hh"
#include "tempest/math/sampling3.hh"

#include <cuda_runtime.h>
#include <texture_indirect_functions.h>

const size_t ImageWidth = 512,
             ImageHeight = 512;

__global__ void SampleCubeMapTest(cudaTextureObject_t cube_map, const Tempest::Vector3* sample_locations, unsigned sample_count, Tempest::Spectrum* out_tex_samples)
{
    auto idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= sample_count)
        return;

    auto& dir = sample_locations[idx];

    Tempest::Vector4 result;
    texCubemap(reinterpret_cast<float4*>(&result), cube_map, dir.x, dir.y, dir.z);

    out_tex_samples[idx] = reinterpret_cast<Tempest::Vector3&>(result);
}


TGE_TEST("Testing cube map sampling capabilities")
{
    Tempest::Texture* cube_tex[] =
    {
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posx-256.png")),
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negx-256.png")),
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posy-256.png")),
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negy-256.png")),
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posz-256.png")),
        LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negz-256.png"))
    };
    auto at_exit = Tempest::CreateAtScopeExit([&cube_tex]()
    {
        for(size_t i = 0; i < 6; ++i)
        {
            delete cube_tex[i];
        }
    });

    Tempest::CubeMap cube_map(cube_tex);

    auto cuda_tex = CREATE_SCOPED(cudaTextureObject_t, Tempest::CudaTextureDeleter);
    cuda_tex = Tempest::CreateCudaTexture(&cube_map);

    Tempest::Vector3 target{0.0f, 1.0f, 0.0f},
                     origin{0.0f, 1.0f, 5.5f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAtUpTarget(origin, target, up);

    Tempest::Matrix4 proj = Tempest::PerspectiveMatrix(50.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);
    auto view_proj = proj*view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    size_t sample_count = ImageWidth*ImageHeight;

    std::unique_ptr<Tempest::Vector3[]> tex_samples_dirs(new Tempest::Vector3[sample_count]);
    std::unique_ptr<Tempest::Spectrum[]> tex_samples(new Tempest::Spectrum[sample_count]);

    for(size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx)
    {
        size_t x = sample_idx % ImageWidth,
               y = sample_idx / ImageWidth;

        Tempest::Vector4 screen_tc{2.0f*x/(ImageWidth - 1) - 1.0f, 2.0f*y/(ImageHeight - 1) - 1.0f, 1.0f, 1.0};

		auto pos_end = Tempest::ToVector3(view_proj_inv*screen_tc);

        auto dir = Tempest::Normalize(pos_end - origin);
        tex_samples_dirs[sample_idx] = dir;
        tex_samples[sample_idx] = cube_map.sampleSpectrum(dir);
    }

    auto cuda_tex_sample_dirs = CREATE_SCOPED(Tempest::Vector3*, ::cudaFree);
    auto status = cudaMalloc(&cuda_tex_sample_dirs, sample_count*sizeof(Tempest::Vector3));
    TGE_CHECK(status == cudaSuccess, "Failed to allocate memory for sampled directions");

    status = cudaMemcpy(cuda_tex_sample_dirs.get(), tex_samples_dirs.get(), sample_count*sizeof(Tempest::Vector3), cudaMemcpyHostToDevice);
    TGE_CHECK(status == cudaSuccess, "Failed to copy samples to CUDA");

    auto cuda_tex_samples = CREATE_SCOPED(Tempest::Vector3*, ::cudaFree);
    status = cudaMalloc(&cuda_tex_samples, sample_count*sizeof(Tempest::Spectrum));
    TGE_CHECK(status == cudaSuccess, "Failed to allocate memory for cuda texture samples");

    unsigned group_size_x = 64;
    dim3 thread_groups(((unsigned)sample_count + group_size_x - 1)/group_size_x);
    dim3 group_size(group_size_x);

    SampleCubeMapTest<<<thread_groups, group_size>>>(cuda_tex.get(), cuda_tex_sample_dirs.get(), (unsigned)sample_count, cuda_tex_samples.get());

    std::unique_ptr<Tempest::Spectrum[]> result_cuda_tex_samples(new Tempest::Spectrum[sample_count]);
    status = cudaMemcpy(result_cuda_tex_samples.get(), cuda_tex_samples.get(), sample_count*sizeof(Tempest::Spectrum), cudaMemcpyDeviceToHost);

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = ImageWidth;
    tex_desc.Height = ImageHeight;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    auto* cpu_tex_data = new uint32_t[tex_desc.Width*tex_desc.Height];
    Tempest::Texture cpu_texture(tex_desc, reinterpret_cast<uint8_t*>(cpu_tex_data));

    auto gpu_tex_data = new uint32_t[tex_desc.Width*tex_desc.Height];
    Tempest::Texture gpu_texture(tex_desc, reinterpret_cast<uint8_t*>(gpu_tex_data));

    for(size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx)
    {
        cpu_tex_data[sample_idx] = Tempest::ToColor(Tempest::ConvertLinearToSRGB(Tempest::SpectrumToRGB(tex_samples[sample_idx])));
        gpu_tex_data[sample_idx] = Tempest::ToColor(Tempest::ConvertLinearToSRGB(Tempest::SpectrumToRGB(result_cuda_tex_samples[sample_idx])));
    }

    Tempest::SaveImage(tex_desc, cpu_tex_data, Tempest::Path("cpu-cube-sampling.png"));
    Tempest::SaveImage(tex_desc, gpu_tex_data, Tempest::Path("gpu-cube-sampling.png"));

    for(size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx)
    {
        auto& cpu_sample = tex_samples[sample_idx];
        auto& gpu_sample = result_cuda_tex_samples[sample_idx];
        TGE_CHECK(Tempest::ApproxEqual(cpu_sample, gpu_sample, 1e-2f), "Invalid Cubemap sampling");
    }
}
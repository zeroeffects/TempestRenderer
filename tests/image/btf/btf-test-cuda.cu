#include "tempest/utils/testing.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/image/btf.hh"
#include "tempest/utils/threads.hh"
#include "tempest/math/sampling3.hh"

#include <cuda_runtime_api.h>

#define GROUP_SIZE_SIDE 8

__global__ void SampleBTFCheapSampling(const Tempest::BTF* btf, Tempest::Vector3 light_barycentric, uint32_t light_prim_id, Tempest::Vector3 view_barycentric, uint32_t view_prim_id, Tempest::Vector3* image)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= btf->Width ||
       y >= btf->Height)
        return;

	auto color = Tempest::SpectrumToRGB(BTFFetchPixelSampleLightViewSpectrum(btf, light_prim_id, light_barycentric, view_prim_id, view_barycentric, x, y));
    image[y*btf->Width + x] = color;
}

__global__ void SampleBTFLightViewSampling(const Tempest::BTF* btf, Tempest::Vector3 light, Tempest::Vector3 view, Tempest::Vector3* image)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= btf->Width ||
       y >= btf->Height)
        return;

	auto color = Tempest::SpectrumToRGB(BTFFetchPixelSampleLightViewSpectrum(btf, light, view, x, y));
    image[y*btf->Width + x] = color;
}

__global__ void SampleBTFHemisphereSampling(const Tempest::BTF* btf, Tempest::Vector3* image)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= btf->Width ||
       y >= btf->Height)
        return;
    auto light = Tempest::UniformSampleSphere((float)x/btf->Width, (float)y/btf->Height); // texture reuse, i.e. same width-height
    auto view = Tempest::Reflect(light, Tempest::Vector3{ 0.0f, 0.0f, 1.0f });

    auto color = Tempest::SpectrumToRGB(BTFSampleSpectrum(btf, light, view, { 0.1f, 0.1f }));
    image[y*btf->Width + x] = color;
}

TGE_TEST("Testing BTF support on CUDA")
{
    Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(ROOT_SOURCE_DIR "/tests/image/btf/fabric09_resampled_W400xH400_L151xV151.btf")));
	TGE_CHECK(btf, "Failed to load BTF");
    auto btf_ptr = btf.get();

    Tempest::BTFGPUPtr btf_gpu(Tempest::CreateGPUBTF(btf.get()));

    uint32_t btf_width = btf->Width,
             btf_height = btf->Height,
             tex_area = btf->Width*btf->Height;

    const float angle = Tempest::ToRadians(45.0f);
    float sin_theta, cos_theta;
    Tempest::FastSinCos(angle, &sin_theta, &cos_theta);

    Tempest::Vector3 view{ 0.0f, -sin_theta, cos_theta },
                     light{ 0.0f, sin_theta, cos_theta };

    Tempest::Vector3 light_barycentric, view_barycentric;
    uint32_t light_prim_id, view_prim_id;
    auto intersect = BTFFetchLightViewDirection(btf.get(), light, view, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
    TGE_CHECK(intersect, "Failed to get direction information");

 	auto cuda_btf_slice = CREATE_SCOPED(Tempest::Vector3*, ::cudaFree);
	size_t slice_size = tex_area*sizeof(cuda_btf_slice[0]);
	auto status = cudaMalloc(reinterpret_cast<void**>(&cuda_btf_slice), slice_size);
	TGE_CHECK(status == cudaSuccess, "Failed to allocate GPU BTF slice memory");
	
	std::unique_ptr<Tempest::Vector3[]> cpu_btf_slice(new Tempest::Vector3[tex_area]);
    auto cpu_btf_slice_ptr = cpu_btf_slice.get();

    dim3 thread_groups((btf->Width + GROUP_SIZE_SIDE - 1)/GROUP_SIZE_SIDE, (btf->Height + GROUP_SIZE_SIDE - 1)/GROUP_SIZE_SIDE, 1);
    dim3 group_size(GROUP_SIZE_SIDE, GROUP_SIZE_SIDE, 1);
    SampleBTFCheapSampling<<<thread_groups, group_size>>>(btf_gpu.get(), light_barycentric, light_prim_id, view_barycentric, view_prim_id, cuda_btf_slice.get());
	cudaThreadSynchronize();
	status = cudaGetLastError();
	TGE_CHECK(status == cudaSuccess, "Failed to perform fixed Light-View BTF sampling");

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();


    auto parallel_decode = Tempest::CreateParallelForLoop2D(btf_width, btf_height, 64,
															[cpu_btf_slice_ptr, btf_width, btf_ptr, &light, &view,
                                                             &light_barycentric, &view_barycentric,
                                                             light_prim_id, view_prim_id](uint32_t worker_id, uint32_t x, uint32_t y)
															{
																uint32_t idx = y*btf_width + x;
																cpu_btf_slice_ptr[idx] = Tempest::SpectrumToRGB(BTFFetchPixelSampleLightViewSpectrum(btf_ptr, light_prim_id, light_barycentric, view_prim_id, view_barycentric, x, y));
														    });
	
	pool.enqueueTask(&parallel_decode);

	pool.waitAndHelp(id, &parallel_decode);

	std::unique_ptr<Tempest::Vector3[]> gpu_btf_slice(new Tempest::Vector3[tex_area]);
	status = cudaMemcpy(gpu_btf_slice.get(), cuda_btf_slice, slice_size, cudaMemcpyDeviceToHost);
	TGE_CHECK(status == cudaSuccess, "Failed to BTF slice from GPU");

    for(uint32_t texel_idx = 0; texel_idx < tex_area; ++texel_idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(gpu_btf_slice[texel_idx], cpu_btf_slice[texel_idx], 1e-4f), "Broken GPU btf sampler");
    }

	SampleBTFLightViewSampling<<<thread_groups, group_size>>>(btf_gpu.get(), light, view, cuda_btf_slice.get());
	cudaThreadSynchronize();
	status = cudaGetLastError();
	TGE_CHECK(status == cudaSuccess, "Failed to perform full BTF sampling");

	status = cudaMemcpy(gpu_btf_slice.get(), cuda_btf_slice, slice_size, cudaMemcpyDeviceToHost);
	TGE_CHECK(status == cudaSuccess, "Failed to BTF slice from GPU");

	for(uint32_t texel_idx = 0; texel_idx < tex_area; ++texel_idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(gpu_btf_slice[texel_idx], cpu_btf_slice[texel_idx], 1e-4f), "Broken GPU btf sampler");
    }

    SampleBTFHemisphereSampling<<<thread_groups, group_size>>>(btf_gpu.get(), cuda_btf_slice.get());
    cudaThreadSynchronize();
	status = cudaGetLastError();
	TGE_CHECK(status == cudaSuccess, "Failed to perform full BTF sampling");
}
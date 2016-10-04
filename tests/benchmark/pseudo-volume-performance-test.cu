#include "tempest/utils/testing.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/quaternion.hh"

#define ILLUMINATION_MODEL_IMPLEMENTATION
#define ILLUMINATION_MODEL_STATIC_IMPLEMENTATION
#include "tempest/graphics/ray-tracing/illumination-models.hh"

#include <cuda_runtime_api.h>
#include <memory>

const uint32_t ImageWidth = 1920;
const uint32_t ImageHeight = 1080;

__global__ void ComputeImage(Tempest::RTSGGXSurface sggx_render_material, uint32_t width, uint32_t height, uint32_t* backbuffer)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y >= height)
		return;

	float angle = Tempest::MathPi*0.25f, s, c;
	Tempest::FastSinCos(angle, &s, &c);

    Tempest::SampleData sample_data{};
	sample_data.Material = &sggx_render_material;
	sample_data.IncidentLight = Tempest::Vector3{ 0.0f, s, c };
	sample_data.OutgoingLight = Tempest::Vector3{ 0.0f, -s, c };
    sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
	 
	unsigned seed = (y << 16) + x;
    Tempest::Cuda::SGGXSurfaceCache(sample_data, seed);

	auto spec = Tempest::Cuda::SGGXMicroFlakePseudoVolumeBRDF(sample_data);

	backbuffer[y*width + x] = Tempest::ToColor(Tempest::SpectrumToRGB(spec));
}

TGE_TEST("Testing pseudo volume performance in the most optimistic case")
{
	auto backbuffer = CREATE_SCOPED(uint32_t*, ::cudaFree);
	uint32_t backbuffer_area = ImageWidth*ImageHeight;
	uint32_t backbuffer_size = backbuffer_area*sizeof(backbuffer[0]);
	auto status = cudaMalloc(reinterpret_cast<void**>(&backbuffer), backbuffer_size);
	TGE_CHECK(status == cudaSuccess, "Failed to allocate backbuffer");

	Tempest::TextureDescription stddev_tex_desc;
	
	Tempest::TextureDescription sggx_tex_desc;
	sggx_tex_desc.Width = 1;
	sggx_tex_desc.Height = 1;
	sggx_tex_desc.Format = Tempest::DataFormat::RGBA32F;
	Tempest::Texture sggx_stddev_tex(sggx_tex_desc, reinterpret_cast<uint8_t*>(new Tempest::Vector4{ 0.5f, 0.5f, 0.5f, 0.0f }));
	Tempest::Texture sggx_basis_tex(sggx_tex_desc, reinterpret_cast<uint8_t*>(new Tempest::Quaternion{ 0.0f, 0.0f, 0.0f, 1.0f }));

	Tempest::RayTracingCudaSystem rt_sys(ImageWidth, ImageHeight, Tempest::Matrix4::identityMatrix());

	auto rt_scene = rt_sys.getRayTracer();

    Tempest::RTSGGXSurface sggx_render_material{};
	sggx_render_material.Depth = 1; 
	sggx_render_material.SampleCount = 256;
	sggx_render_material.Model = Tempest::IlluminationModel::SGGXPseudoVolume;
    sggx_render_material.Diffuse = {};
    sggx_render_material.Specular = Tempest::ToSpectrum(0.75f);
    sggx_render_material.BasisMapWidth = sggx_tex_desc.Width;
    sggx_render_material.BasisMapWidth = sggx_tex_desc.Height;
    sggx_render_material.BasisMap = rt_scene->bindTexture(&sggx_basis_tex);
    sggx_render_material.StandardDeviationMap = rt_scene->bindTexture(&sggx_stddev_tex);
    sggx_render_material.setup();
	
	Tempest::TimeQuery timer;

	auto start_time = timer.time();

	dim3 comp_group_size(8, 8, 1);
    dim3 comp_thread_groups((ImageWidth + comp_group_size.x - 1) / comp_group_size.x, (ImageHeight + comp_group_size.y - 1) / comp_group_size.y, 1);
	ComputeImage<<<comp_thread_groups, comp_group_size>>>(sggx_render_material, ImageWidth, ImageHeight, backbuffer);

	cudaThreadSynchronize();
	status = cudaGetLastError();
	TGE_CHECK(status == cudaSuccess, "Failed to launch kernel");
	
	auto end_time = timer.time();
	auto elapsed_time = end_time - start_time;

	Tempest::Log(Tempest::LogLevel::Info, "Compute a Full HD image in ", elapsed_time*1e-6f, "s");

	std::unique_ptr<uint32_t[]> backbuffer_copy(new uint32_t[backbuffer_size]);

	status = cudaMemcpy(backbuffer_copy.get(), backbuffer, backbuffer_size, cudaMemcpyDeviceToHost);
	TGE_CHECK(status == cudaSuccess, "Failed to copy backbuffer data");
}
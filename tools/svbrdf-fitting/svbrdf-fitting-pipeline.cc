/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2016 Zdravko Velinov
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *   THE SOFTWARE.
 */

#include "tempest/math/fitting.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/utils/timer.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/math/vector3.hh"
#include "tempest/utils/threads.hh"

#include "svbrdf-fitting.hh"

#ifndef DISABLE_CUDA
#include <cuda_runtime_api.h>

void CudaResourceDeleter::operator()(void* data)
{
    ::cudaFree(data);
}
#endif

SVBRDFFitPipeline::SVBRDFFitPipeline(const Tempest::BTF* btf_cpu
#ifndef DISABLE_CUDA
    , const Tempest::BTF* btf_gpu
#endif
    )
    :   m_BTFCPU(btf_cpu)
#ifndef DISABLE_CUDA
    ,   m_BTFGPU(btf_gpu)
#endif
{
    auto light_count = btf_cpu->LightCount,
         view_count = btf_cpu->LightCount;

	uint32_t lv_size = light_count*view_count;
    m_LuminanceSlice = std::unique_ptr<float[]>(new float[lv_size]);

    Tempest::TextureDescription ndf_tex_desc;
    auto ndf_width = ndf_tex_desc.Width = NDFTextureResolution;
    auto ndf_height = ndf_tex_desc.Height = NDFTextureResolution;
    ndf_tex_desc.Format = Tempest::DataFormat::R32F;
	float* ndf_texture_data = new float[ndf_tex_desc.Width*ndf_tex_desc.Height];
	size_t ndf_tex_size = ndf_tex_desc.Width*ndf_tex_desc.Height*sizeof(ndf_texture_data[0]);
    m_NDFTexture = std::unique_ptr<Tempest::Texture>(new Tempest::Texture(ndf_tex_desc, reinterpret_cast<uint8_t*>(ndf_texture_data)));

#ifndef DISABLE_CUDA
    if(btf_gpu)
    {
	    float* gpu_ndf_texture_data;
	    auto status = cudaMalloc(reinterpret_cast<void**>(&gpu_ndf_texture_data), ndf_tex_size);
        m_GPUNDFData = std::unique_ptr<float, CudaResourceDeleter>(gpu_ndf_texture_data, CudaResourceDeleter());

	    TGE_ASSERT(status == cudaSuccess, "Failed to alloc NDF texture");
	    status = cudaMemset(gpu_ndf_texture_data, 0, ndf_tex_size);
	    TGE_ASSERT(status == cudaSuccess, "Failed to memset NDF texture");

	    float* gpu_lv_lum_slice;
	    status = cudaMalloc(reinterpret_cast<void**>(&gpu_lv_lum_slice), lv_size*sizeof(gpu_lv_lum_slice[0])); 
	    TGE_ASSERT(status == cudaSuccess, "Failed to alloc luminance slice buffer");
        m_GPULuminanceSlice = std::unique_ptr<float, CudaResourceDeleter>(gpu_lv_lum_slice, CudaResourceDeleter());
    }
#endif

	memset(ndf_texture_data, 0, ndf_tex_size);

    m_Lights = std::unique_ptr<Tempest::Vector3[]>(new Tempest::Vector3[light_count]);
    for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
    {
        m_Lights[light_idx] = Tempest::ParabolicToCartesianCoordinates(btf_cpu->LightsParabolic[light_idx]);
    }
        
    m_AveragedProbabilities = std::unique_ptr<float[]>(new float[light_count]);
}

struct ReduceNDFIntegrand
{
	Tempest::Vector3       Normal;
	const Tempest::BTF*    BTFData;
	float*          	   LuminanceSlice;

	inline EXPORT_CUDA float operator()(const Tempest::Vector3& inc_light)
	{
		auto out_light = Tempest::Reflect(inc_light, Normal);
        return BTFSampleLuminanceSlice(BTFData, inc_light, out_light, LuminanceSlice);
    }
};

struct NDFRecord
{
	const Tempest::BTF* BTFData;
	float*		        NDFTextureData;
	uint32_t	        NDFWidth,
				        NDFHeight;
	float*		        LuminanceSlice;

	inline EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t ndf_x, uint32_t ndf_y)
	{
		uint32_t idx = ndf_y*NDFWidth + ndf_x;
		
		auto norm = Tempest::ParabolicMapToCartesianCoordinates(Tempest::Vector2{ (float)ndf_x/(NDFWidth - 1), (float)ndf_y/(NDFHeight - 1) });
		if(norm.z <= 0.0f)
		{
			NDFTextureData[idx] = 0.0f;
			return;
		}

		ReduceNDFIntegrand ndf_integrand{ norm, BTFData, LuminanceSlice };

		NDFTextureData[idx] = Tempest::StratifiedMonteCarloIntegratorHemisphere(NDRReduceSphereSamples, ndf_integrand);
	}
};

struct NDFRecordTop
{
	const Tempest::BTF* BTFData;
	float*		        NDFTextureData;
	uint32_t	        NDFWidth,
				        NDFHeight;
	float*		        LuminanceSlice;

	inline EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t ndf_x, uint32_t ndf_y)
	{
		uint32_t idx = ndf_y*NDFWidth + ndf_x;
		
		auto norm = Tempest::ParabolicMapToCartesianCoordinates(Tempest::Vector2{ (float)ndf_x/(NDFWidth - 1), (float)ndf_y/(NDFHeight - 1) });
		if(norm.z <= 0.0f)
		{
			NDFTextureData[idx] = 0.0f;
			return;
		}

        Tempest::Vector3 out_light{ 0.0f, 0.0f, 1.0f };
        auto inc_light = Tempest::Reflect(out_light, norm);

		NDFTextureData[idx] = BTFSampleLuminanceSlice(BTFData, inc_light, out_light, LuminanceSlice);
	}
};

struct PCAIntegrand
{
	void* NDFTexture;

	EXPORT_CUDA float operator()(const Tempest::Vector3& norm)
    {
        if(norm.z < 0.0f)
            return 0.0f;

        auto parabolic_coord = Tempest::CartesianToParabolicMapCoordinates(norm);
        return Tempest::SampleRed(NDFTexture, parabolic_coord);
    }
};

struct SGGXProjectIntegrand
{
	Tempest::Matrix3  Basis;
	void* NDFTexture;

    EXPORT_CUDA Tempest::Vector3 operator()(const Tempest::Vector3& norm)
	{
        if(norm.z <= 0.0f)
            return Tempest::Vector3{};

        auto parabolic_coord = Tempest::CartesianToParabolicMapCoordinates(norm);

		return Tempest::Vector3Abs(Basis.transformRotationInverse(norm))*Tempest::SampleRed(NDFTexture, parabolic_coord);
	}
};

struct NDFNormalizationIntegrand
{
	void*			NDFTexture;
	EXPORT_CUDA float operator()(const Tempest::Vector3& norm)
    {
        auto tc = Tempest::CartesianToParabolicMapCoordinates(norm);
        TGE_ASSERT(0.0f <= tc && tc <= 1.0f, "Invalid coordinate");
        return Tempest::SampleRed(NDFTexture, tc);
    }
};

#if 0
template<class TLoopBody>
__global__ void ParallelForLoop2DImpl(uint32_t width, uint32_t height, TLoopBody body)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= width || y >= height)
		return;

	body(blockDim.x*threadIdx.y + threadIdx.x, x, y);
}

template<class TPool, class TLoopBody>
void GPUExecParallelForLoop2D(uint32_t, TPool&, uint32_t width, uint32_t height, TLoopBody& body)
{
	dim3 group_size(8, 8, 1);
    dim3 thread_groups((width + group_size.x - 1)/group_size.x,
                       (height + group_size.y - 1)/group_size.y, 1);

	ParallelForLoop2DImpl<<<thread_groups, group_size>>>(width, height, body);

#ifndef NDEBUG
	cudaThreadSynchronize();
	auto status = cudaGetLastError();
	TGE_ASSERT(status == cudaSuccess, "Failed to execute");
#endif
}
#endif

template<class TPool, class TLoopBody>
void CPUExecParallelForLoop2D(uint32_t id, TPool& pool, uint32_t width, uint32_t height, TLoopBody& body)
{
	auto func = Tempest::CreateParallelForLoop2D(width, height, 16, body);
    pool.enqueueTask(&func);
	pool.waitAndHelp(id, &func);
}

void SVBRDFFitPipeline::cache(uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, FitPipelineOptions& pipeline_opts, Tempest::TimeQuery* timer)
{
    auto btf_cpu = m_BTFCPU;

    auto light_count = btf_cpu->LightCount,
         view_count = btf_cpu->LightCount,
         lv_size = light_count*view_count;

    uint64_t start_luminance;

    auto* lv_lum_slice = m_LuminanceSlice.get();

    if(timer)
        start_luminance = timer->time();

    switch(pipeline_opts.Filter)
    {
    case FilteringTechnique::Gaussian:
    {
        auto thread_count = pool.getThreadCount();
        auto* avg_spec_storage = TGE_TYPED_ALLOCA(Tempest::Spectrum, thread_count);
        memset(avg_spec_storage, 0, thread_count*sizeof(Tempest::Spectrum));

        // TODO: Spatial caching
        auto kernel_size = pipeline_opts.KernelRadius;
        auto slice_light = Tempest::CreateParallelForLoop2D(light_count, view_count, 64,
                                                            [avg_spec_storage, btf_cpu, btf_x, btf_y, kernel_size, lv_lum_slice](uint32_t worker_id, uint32_t light_idx, uint32_t view_idx) 
            {
                float total_contrib = 0.0f;
                uint32_t lv_idx = view_idx*btf_cpu->LightCount + light_idx;
                auto btf_width = btf_cpu->Width;

                float sigma = 2.0f*kernel_size/3.0f; // three sigma rule/empirical rule
                float gaussian_spread = 0.5f/(sigma*sigma);

                Tempest::Spectrum spec = {};

                for(uint32_t sample_y = Maxi(btf_y - kernel_size, 0u),
                                sample_y_end = Mini(btf_y + kernel_size + 1, btf_cpu->Width - 1);
                    sample_y < sample_y_end; ++sample_y)
                    for(uint32_t sample_x = Maxi(btf_x - kernel_size, 0u),
                                    sample_x_end = Mini(btf_x + kernel_size + 1, btf_cpu->Width - 1);
                        sample_x < sample_x_end; ++sample_x)
                    {
                        auto btf_xy_idx = sample_y*btf_width + sample_x;

                        float x_coef = static_cast<float>((int32_t)sample_x - (int32_t)btf_x);
                        float y_coef = static_cast<float>((int32_t)sample_y - (int32_t)btf_y);

                        float weight = expf(-(x_coef*x_coef + y_coef*y_coef)*gaussian_spread);
                        total_contrib += weight;

                        spec += weight*BTFFetchSpectrum(btf_cpu, lv_idx, btf_xy_idx);
                    }

                spec /= total_contrib;

                float luminance = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec)); // TODO: Luminance out of spectrum
                spec = luminance > 1e-6f ? spec / luminance : Tempest::ToSpectrum(1.0f);

                avg_spec_storage[worker_id] += spec;
                lv_lum_slice[lv_idx] = luminance;
		    });
        pool.enqueueTask(&slice_light);

        pool.waitAndHelp(id, &slice_light);

        auto avg_spec = avg_spec_storage[0];
        for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
        {
            avg_spec += avg_spec_storage[thread_idx];
        }
        m_AverageSpectrum = avg_spec/static_cast<float>(lv_size);
    } break;
    default:
    {
	    Tempest::BTFParallelExtractLuminanceSlice(btf_cpu, id, pool, btf_x, btf_y, &lv_lum_slice, &m_AverageSpectrum);
    }
    }

    if(timer)
    {
        auto elapsed_luminance = timer->time() - start_luminance;
	    Tempest::Log(Tempest::LogLevel::Info, "Finished generating luminance slice: ", elapsed_luminance, "us");
    }

}

void SVBRDFFitPipeline::fit(uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y,
                          LeastSquaresFitOptions* fit_opts, FitPipelineOptions& pipeline_opts, Tempest::TimeQuery* timer, SGGXParameters* out_parameters, bool reoptimize)
{
    OptimizationParameters parameters;

    auto btf_cpu = m_BTFCPU;

    auto light_count = btf_cpu->LightCount,
         view_count = btf_cpu->LightCount,
         lv_size = light_count*view_count;
    
    auto* ndf_texture_data = reinterpret_cast<float*>(m_NDFTexture->getData());
    auto* lv_lum_slice = m_LuminanceSlice.get();
    
    PCAIntegrand pca_integrand{ m_NDFTexture.get() };
	SGGXProjectIntegrand sggx_integrand;
	sggx_integrand.NDFTexture = m_NDFTexture.get();

	uint64_t start_ndf_convert, start_pca, start_sggx_project, start_fit;

    Tempest::Spectrum avg_spec = m_AverageSpectrum;

    if(!reoptimize)
    {    
        if(timer)
        {
            start_ndf_convert = timer->time();
        }

        auto& ndf_hdr = m_NDFTexture->getHeader();

        if(pipeline_opts.Flags & PIPELINE_OPTION_NDF_TOP)
        {
            NDFRecordTop ndf_record{ m_BTFCPU, reinterpret_cast<float*>(m_NDFTexture->getData()), ndf_hdr.Width, ndf_hdr.Height, lv_lum_slice };
            CPUExecParallelForLoop2D(id, pool, ndf_hdr.Width, ndf_hdr.Height, ndf_record);
        }
        else
        {
        #if 0
            if(m_BTFGPU)
            {
	            NDFRecord ndf_record{ m_BTFGPU, m_GPUNDFData.get(), ndf_hdr.Width, ndf_hdr.Height, m_GPULuminanceSlice.get() };

		        auto status = cudaMemcpy(m_GPULuminanceSlice.get(), lv_lum_slice, lv_size*sizeof(lv_lum_slice[0]), cudaMemcpyHostToDevice);
		        TGE_ASSERT(status == cudaSuccess, "Failed to transfer luminance slice");
            
                GPUExecParallelForLoop2D(id, pool, ndf_hdr.Width, ndf_hdr.Height, ndf_record);

                status = cudaMemcpy(ndf_texture_data, m_GPUNDFData.get(), ndf_hdr.Width*ndf_hdr.Height*sizeof(m_GPUNDFData.get()[0]), cudaMemcpyDeviceToHost);
		        TGE_ASSERT(status == cudaSuccess, "Failed to copy memory");

		        #if 0
		        std::unique_ptr<float[]> cpu_ndf_texture_data(new float[ndf_width*ndf_height]);
		        memset(cpu_ndf_texture_data.get(), 0, ndf_tex_size);
		        NDFRecord cpu_ndf_record{ btf_ptr, cpu_ndf_texture_data.get(), ndf_width, ndf_height, lv_lum_slice };
		        CPUExecParallelForLoop2D(id, pool, ndf_width, ndf_height, cpu_ndf_record);

		        for(uint32_t pixel_idx = 0, pixel_idx_end = ndf_width*ndf_height; pixel_idx < pixel_idx_end; ++pixel_idx)
			        TGE_ASSERT(Tempest::ApproxEqual(cpu_ndf_texture_data[pixel_idx], ndf_texture_data[pixel_idx], (cpu_ndf_texture_data[pixel_idx] + 1e-1f)*1e-1f), "Invalid CUDA computation");
		        #endif
            }
            else
        #endif
            {
                NDFRecord ndf_record{ m_BTFCPU, reinterpret_cast<float*>(m_NDFTexture->getData()), ndf_hdr.Width, ndf_hdr.Height, lv_lum_slice };
		        CPUExecParallelForLoop2D(id, pool, ndf_hdr.Width, ndf_hdr.Height, ndf_record);
            }
        }

        if(timer)
        {
            auto elapsed_ndf_convert = timer->time() - start_ndf_convert;
            Tempest::Log(Tempest::LogLevel::Info, "Completed conversion of BTF sample(", btf_x, ", ", btf_y, ") to NDF: ", elapsed_ndf_convert, "us");
        }

	    NDFNormalizationIntegrand ndf_norm_integrand{ m_NDFTexture.get() };

        auto ndf_denom = Tempest::StratifiedMonteCarloIntegratorHemisphere(256*256, ndf_norm_integrand);
        auto ndf_norm = 1/ndf_denom;

    // Energy conservation check -- most probably fails for real world materials because it works only for non-absorbing materials.
    #if 0
        auto reflected_radiance = Tempest::StratifiedMonteCarloIntegratorHemisphere(256*256,
            [lv_lum_slice, btf_ptr](const Tempest::Vector3& out_light)
            {
                return BTFSampleLuminanceSlice(btf_ptr, Tempest::Vector3{ 0.0f, 0.0f, 1.0f }, out_light, lum_slice.get());
            });
        TGE_ASSERT(reflected_radiance <= 1.0f, "BTF is energy generating");
        TGE_ASSERT(Tempest::ApproxEqual(reflected_radiance, ndf_denom, 1e-2f), "Wildly different NDF and averaged reflected radiance");
    #endif

        if(timer)
        {
            start_pca = timer->time(); 
        }

        {
        Tempest::Vector3 most_prominent_norm;

        switch(pipeline_opts.BasisExtract)
        {
        case BasisExtractStrategy::PCAHemisphere:
        {
            float sqrt_samples = sqrtf((float)EigenVectorSphereSamples);
            uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
            TGE_ASSERT((EigenVectorSphereSamples % sqrt_samples_i) == 0, "Invalid sample count");

            struct PartialMostDominant
            {
                float                NDFValue;
                Tempest::Vector3     Axis;
            };

            auto thread_count = pool.getThreadCount();
            auto* partial_most_prominent = TGE_TYPED_ALLOCA(PartialMostDominant, thread_count);
    
            PartialMostDominant fill_value{ 0.0f, Tempest::Vector3{ 0.0f, 0.0f, 1.0f } };
            fill_value.NDFValue = pca_integrand(fill_value.Axis);

            std::fill(partial_most_prominent, partial_most_prominent + thread_count, fill_value);

    
            unsigned seed = 1;
            auto parallel_search = Tempest::CreateParallelForLoop2D(sqrt_samples_i, sqrt_samples_i, 8,
                [&pca_integrand, sqrt_samples, &seed, partial_most_prominent](uint32_t worker_id, uint32_t x, uint32_t y)
                {
                    float x_rand = (x + Tempest::FastFloatRand(seed))/sqrt_samples;
                    float y_rand = (y + Tempest::FastFloatRand(seed))/sqrt_samples;

                    auto norm = Tempest::UniformSampleHemisphere(x_rand, y_rand);
                    auto cur_ndf_value = pca_integrand(norm);

                    auto& partial_info = partial_most_prominent[worker_id];
                    auto& ndf_value = partial_info.NDFValue;
                    auto& most_prominent_norm = partial_info.Axis;

                    if(cur_ndf_value > ndf_value)
                    {
                        most_prominent_norm = norm;
                        ndf_value = cur_ndf_value;
                    }
                });

            pool.enqueueTask(&parallel_search);
            pool.waitAndHelp(id, &parallel_search);

            auto& first_most_prominent = partial_most_prominent[0];
            float ndf_value = first_most_prominent.NDFValue;
            most_prominent_norm = first_most_prominent.Axis;
            for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
            {
                auto& partial_info = partial_most_prominent[thread_idx];
                auto cur_ndf_value = partial_info.NDFValue;

                if(cur_ndf_value > ndf_value)
                {
                    most_prominent_norm = partial_info.Axis;
                    ndf_value = cur_ndf_value;
                }
            }
        } break;
        case BasisExtractStrategy::PhotometricNormals:
        {
            auto normal_ptr = most_prominent_norm.Components;
            for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
		    {
			    m_AveragedProbabilities[light_idx] = lv_lum_slice[light_idx];
		    }


		    for(uint32_t view_idx = 1; view_idx < view_count; ++view_idx)
			    for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
			    {
				    m_AveragedProbabilities[light_idx] += lv_lum_slice[view_idx*light_count + light_idx];
			    }

            float scale_coef = 1.0f/view_count;
		    for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
		    {
			    m_AveragedProbabilities[light_idx] *= scale_coef;
		    }

            Tempest::MatrixTransposeLinearSolve(reinterpret_cast<float*>(m_Lights.get()), TGE_FIXED_ARRAY_SIZE(Tempest::Vector3().Components), light_count, 
                                                m_AveragedProbabilities.get(), 1, light_count, 
                                                &normal_ptr);

            Tempest::NormalizeSelf(&most_prominent_norm);
        } break;
        default:
        {
            Tempest::Log(Tempest::LogLevel::Error, "Unsupported basis extract strategy");
            return;
        }
        }

        if(pipeline_opts.Flags & PIPELINE_OPTION_MAXIMIZE_NORMAL_PROJECTION)
        {
            const uint32_t ndf_sample_count = 256;
            const uint32_t max_steps = 10;

            if(most_prominent_norm.z < 0.0f)
                most_prominent_norm = -most_prominent_norm;

            Tempest::Vector2 parameters = Tempest::UniformSampleHemisphereSeed(most_prominent_norm);

            auto* ndf_texture = m_NDFTexture.get();

            // Newton method
            auto conv_dir = Tempest::UniformSampleHemisphere(parameters.Components[0], parameters.Components[1]);
            TGE_ASSERT(Tempest::ApproxEqual(conv_dir, most_prominent_norm), "Invalid conversion");
            Tempest::Vector3 normal_axis = most_prominent_norm;

            auto comp_projection = [ndf_texture, &normal_axis](const Tempest::Vector3& norm)
                                   {
                                       Tempest::Vector2 parabolic_coord = Tempest::CartesianToParabolicMapCoordinates(norm);
                                       return fabsf(Tempest::Dot(normal_axis, norm))*Tempest::SampleRed(ndf_texture, parabolic_coord);
                                   };

            auto cur_proj = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere(id, pool, ndf_sample_count, 1, comp_projection);

            float epsilon = 1e-5f;

            bool continue_fit = true;

            // Maximize area
            for(uint32_t step = 0; continue_fit && step < max_steps; ++step)
            {
                float next_theta_seed = parameters.Components[0] + epsilon;

                if(next_theta_seed > 1.0f)
                    next_theta_seed = 2.0f - next_theta_seed;

                normal_axis = Tempest::UniformSampleHemisphere(next_theta_seed, parameters.Components[1]);

                Tempest::Vector2 gradient;

                gradient.x = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere(id, pool, ndf_sample_count, 1, comp_projection) - cur_proj;

                normal_axis = Tempest::UniformSampleHemisphere(parameters.Components[0], parameters.Components[1] + epsilon);
                gradient.y = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere(id, pool, ndf_sample_count, 1, comp_projection) - cur_proj;

                float step_length = 1.0f;
                const float step_reduce_multiplier = 0.5f;
                const float min_step = 1e-5f;
                auto prev_proj = cur_proj;
                for(;;)
                {
                    Tempest::Vector2 new_parameters = parameters + step_length*gradient;
                    if(new_parameters.x > 1.0f)
                        new_parameters.x = 2.0f - new_parameters.x;

                    normal_axis = Tempest::UniformSampleHemisphere(new_parameters.Components[0], new_parameters.Components[1]);

                    cur_proj = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere(id, pool, ndf_sample_count, 1, comp_projection);
                    if(prev_proj < cur_proj)
                    {
                        parameters = new_parameters;
                        break;
                    }

                    step_length *= step_reduce_multiplier;
                    if(step_length < min_step)
                    {
                        continue_fit = false;
                        break;
                    }
                }
            }

            most_prominent_norm = Tempest::UniformSampleHemisphere(parameters.Components[0], parameters.Components[1]);

            Tempest::Log(Tempest::LogLevel::Info, "Completed PCA normal optimization");
        }
    

        Tempest::Matrix3 basis;
        basis.makeBasis(most_prominent_norm);

        auto rot = Tempest::Matrix2::rotation(2.0f*Tempest::MathPi/(EigenVectorPlaneSamples - 1));

        float ndf_value = 0.0f;
        Tempest::Vector3 most_prominent_tan{ 1.0f, 0.0f, 0.0f };

        for(size_t theta = 0; theta < EigenVectorPlaneSamples; ++theta)
        {
            auto& tan = basis.tangent();    

            auto cur_ndf_value = pca_integrand(tan);
            if(cur_ndf_value > ndf_value)
            {
                most_prominent_tan = tan;
                ndf_value = cur_ndf_value;
            }

            basis.rotateTangentPlane(rot);
        }

        auto tan = Cross(most_prominent_tan, most_prominent_norm);
        sggx_integrand.Basis = Tempest::Matrix3(tan, most_prominent_tan, most_prominent_norm);
        TGE_ASSERT(Tempest::Dot(Tempest::Cross(tan, most_prominent_tan), most_prominent_norm) > 0.0f, "awful basis");

        }
    
        if(timer)
        {
            auto elapsed_pca = timer->time() - start_pca;
            Tempest::Log(Tempest::LogLevel::Info, "Completed PCA on BTF sample(", btf_x, ", ", btf_y, "): ", elapsed_pca, "us");
            start_sggx_project = timer->time();
        }

        Tempest::Vector3 sggx_stddev = ndf_norm*Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere<Tempest::Vector3>(id, pool, ConvertSamples, 8, sggx_integrand);

    #if 0 
        auto ndf_check = Tempest::StratifiedMonteCarloIntegratorSphere(256*256,
            [&sggx_stddev, &sggx_basis](const Tempest::Vector3& norm)
            {
                return Tempest::SGGXMicroFlakeNDF(sggx_stddev, sggx_basis, norm);
            });
        TGE_ASSERT(Tempest::ApproxEqual(ndf_check, 1.0f, 1e-1f), "Invalid SGGX NDF");
    #endif

        if(timer)
        {
            auto elapsed_sggx_convert = timer->time() - start_sggx_project;
            Tempest::Log(Tempest::LogLevel::Info, "Completed SGGX integrate on BTF sample(", btf_x, ", ", btf_y, "): ", elapsed_sggx_convert, "us");
        }

        Tempest::Vector3 scaling, euler;
	    sggx_integrand.Basis.decompose(&scaling, &euler);
        Tempest::Vector2 sggx_stddev_v2 = { sggx_stddev.x/sggx_stddev.z, sggx_stddev.y/sggx_stddev.z };
        parameters.Parameters = { sggx_stddev_v2, ndf_denom, 0.0f, euler };
    }
    else
    {
        auto mat = Tempest::ToMatrix3(out_parameters->Orientation);
        parameters.Parameters.Diffuse = Tempest::RGBToLuminance(out_parameters->Diffuse);
        parameters.Parameters.Specular = Tempest::RGBToLuminance(out_parameters->Specular);
        parameters.Parameters.StandardDeviation = out_parameters->StandardDeviation;
        Tempest::Vector3 scaling;
	    sggx_integrand.Basis.decompose(&scaling, &parameters.Parameters.Euler);
        avg_spec = out_parameters->Specular/parameters.Parameters.Specular;
    }

    if(fit_opts)
    {
        if(timer)
        {
            start_fit = timer->time();
        }

	    OptimizationParameters opt_parameters;

        static_assert(sizeof(OptimizationParameters::ParametersStruct) == sizeof(OptimizationParameters().ParametersArray), "Invalid parameters array");

        float rmse;
        LeastSquaresFitSGGX(id, pool, btf_cpu, lv_lum_slice, *fit_opts, parameters, &opt_parameters, &rmse);

        if(timer)
        {
            auto elapsed_fit = timer->time() - start_fit;
            Tempest::Log(Tempest::LogLevel::Info, "Completed SGGX fitting on BTF sample(", btf_x, ", ", btf_y, "): ", elapsed_fit, "us");
        }

        auto& opt_sggx_stddev = out_parameters->StandardDeviation = opt_parameters.Parameters.StandardDeviation;
        auto& opt_quaternion = out_parameters->Orientation = Tempest::ToQuaternion(opt_parameters.Parameters.Euler);
        auto& opt_albedo = out_parameters->Diffuse = Tempest::SpectrumToRGB(avg_spec*opt_parameters.Parameters.Diffuse);
        auto& opt_specular = out_parameters->Specular = Tempest::SpectrumToRGB(avg_spec*opt_parameters.Parameters.Specular);
        out_parameters->RMSE = rmse;

        if(pipeline_opts.Flags & PIPELINE_OPTION_PRINT_PER_PIXEL)
        {
            auto& sggx_stddev = parameters.Parameters.StandardDeviation;
            Tempest::Log(Tempest::LogLevel::Info, "Initial SGGX PCA fitted distribution of BTF sample(", btf_x, ",", btf_y, "): ", sggx_stddev.x, ", ", sggx_stddev.y, ", ", 1.0f);
     
            Tempest::Log(Tempest::LogLevel::Info, "Optimal SGGX fitted distribution of BTF sample(", btf_x, ",", btf_y,"): ", opt_sggx_stddev.x, ", ", opt_sggx_stddev.y, ", ", 1.0f);
            Tempest::Log(Tempest::LogLevel::Info, "Optimal specular term of BTF sample(", btf_x, ",", btf_y,"): ", opt_specular.x, ", ", opt_specular.y, ", ", opt_specular.z);
            Tempest::Log(Tempest::LogLevel::Info, "Optimal diffuse term of BTF sample(", btf_x, ",", btf_y,"): ", opt_albedo.x, ", ", opt_albedo.y, ", ", opt_albedo.z);

		    Tempest::Matrix3 opt_basis = Tempest::ToMatrix3(opt_quaternion);
		    auto& opt_tan = opt_basis.tangent();
		    auto& opt_binorm = opt_basis.binormal();
		    auto& opt_norm = opt_basis.normal();

            Tempest::Log(Tempest::LogLevel::Info, "Optimal tangent of BTF sample(", btf_x, ",", btf_y,"): ", opt_tan);
		    Tempest::Log(Tempest::LogLevel::Info, "Optimal binormal of BTF sample(", btf_x, ",", btf_y,"): ", opt_binorm);
		    Tempest::Log(Tempest::LogLevel::Info, "Optimal normal of BTF sample(", btf_x, ",", btf_y,"): ", opt_norm);
        }
    }
    else
    {
        out_parameters->StandardDeviation = parameters.Parameters.StandardDeviation;
        auto& quaternion = out_parameters->Orientation = Tempest::ToQuaternion(sggx_integrand.Basis);
        out_parameters->Diffuse = {};
        auto& specular = out_parameters->Specular = Tempest::SpectrumToRGB(avg_spec*parameters.Parameters.Specular);
        out_parameters->RMSE = 0.0f;

        if(pipeline_opts.Flags & PIPELINE_OPTION_PRINT_PER_PIXEL)
        {
            auto& sggx_stddev = parameters.Parameters.StandardDeviation;
            Tempest::Log(Tempest::LogLevel::Info, "SGGX PCA fitted distribution of BTF sample(", btf_x, ",", btf_y,"): ", sggx_stddev.x, ", ", sggx_stddev.y, ", ", 1.0f);
            Tempest::Log(Tempest::LogLevel::Info, "Specular term of BTF sample(", btf_x, ",", btf_y,"): ", specular.x, ", ", specular.y, ", ", specular.z);
            Tempest::Log(Tempest::LogLevel::Info, "Quaternion of BTF sample(", btf_x, ",", btf_y,"): ", quaternion.x, ", ", quaternion.y, ", ", quaternion.z, ", ", quaternion.w);
        }
    }
}

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

#include "tempest/utils/config.hh"

#if !defined(DISABLE_CUDA) && !defined(LINUX)
#   define CUDA_ACCELERATED_FITTING
#endif

#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/image/btf.hh"
#include "tempest/math/fitting.hh"
#include "tempest/math/functions.hh"

#define EXPORT_EVALUATOR

#include "svbrdf-fitting.hh"
#include "svbrdf-lsq-fitting-internal.hh"

void DownsampleFilterLightView(const Tempest::Vector2* orig_light_dirs, uint32_t orig_light_count, float* orig_lv_lum_slice, Tempest::Vector2* downsample_light_dirs, uint32_t downsample_light_count, float* downsampled_lum_slice)
{
    Tempest::Log(Tempest::LogLevel::Warning, "Filtering currently does not produce good results");
    // TODO:
    // Fixing this issue requires doing Voronoi diagrams on a sphere to compute exact area for each point.
    // Some starting points are:
    // Fogel et al. "Exact Implementation of Arrangements of Geodesic Arcs on the Sphere with Applications"
    // Caroli et al. "Robust and Ecient Delaunay triangulations of points on or close to a sphere"
    // The main issue is that these proposals heavily rely on CGAL which is under GPL... However, there
    // are some libraries that compute 3D convex hulls under less strict license. The main caveat is that they
    // would probably result in points being dropped, which is not nice as we can definitely guarantee points
    // strictly lying on a sphere.

    // After computing the Voronoi cells we can use them to compute unbiased estimate of the probability of
    // each point in angular domain.

    // Overall applying this technique should result in better estimate than simply dropping random points.

    // Another option which would probably lead to overestimation is to construct the 3D convex hull of the points
    // weighted by probabilities and start from there.

    TGE_ASSERT(downsample_light_count < orig_light_count, "Invalid light view count");
    uint32_t last_light_dir = ~0u;
    float step_ratio = (float)(orig_light_count - 1)/(downsample_light_count - 1);

    std::unique_ptr<Tempest::Vector3[]> resample_dir(new Tempest::Vector3[downsample_light_count]);

    uint32_t sqrt_count = static_cast<uint32_t>(sqrtf(downsample_light_count));
    
    unsigned seed = 42;
    if(sqrt_count*sqrt_count == downsample_light_count)
    {
        for(uint32_t light_idx = 0; light_idx < downsample_light_count; ++light_idx)
        {
            resample_dir[light_idx] = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        }
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Warning, "Poor quality downsampled diretions(", downsample_light_count, "). Use power of two instead as it enables stratification.");
        for(uint32_t y = 0; y < sqrt_count; ++y)
            for(uint32_t x = 0; x < sqrt_count; ++x)
            {
                resample_dir[y*sqrt_count + x] = Tempest::UniformSampleHemisphere((y + Tempest::FastFloatRand(seed))/sqrt_count, (x + Tempest::FastFloatRand(seed))/sqrt_count);
            }
    }

    float cos_2x = Tempest::Cos2x(1.0f - 1.0f/downsample_light_count);
    float concentration = Tempest::BrentMethod(0.0001f, 10000.0f, 1e-6f, 
            [cos_2x](float concentration) { return cos_2x - Tempest::VonMisesFisherToleranceCone(concentration, 0.95f); });    
    
    float pdf = 1.0f/(2.0f*Tempest::MathPi);
    float rcp_pdf = 1.0f/(orig_light_count*pdf);

    for(uint32_t resampled_view_idx = 0; resampled_view_idx < downsample_light_count; ++resampled_view_idx)
        for(uint32_t resampled_light_idx = 0; resampled_light_idx < downsample_light_count; ++resampled_light_idx)
        {
            auto& sample = downsampled_lum_slice[resampled_view_idx*downsample_light_count + resampled_light_idx];
            sample = 0.0f;

            float view_prob = 0.0f, light_prob = 0.0f;

            float dbg_estimate = 0.0f;

            for(uint32_t orig_view_idx = 0; orig_view_idx < orig_light_count; ++orig_view_idx)
            {
                auto view_kernel = Tempest::VonMisesFisherPDF(Tempest::ParabolicToCartesianCoordinates(orig_light_dirs[orig_view_idx]), concentration,  resample_dir[resampled_view_idx])*rcp_pdf;
                for(uint32_t orig_light_idx = 0; orig_light_idx < orig_light_count; ++orig_light_idx)
                {
                    auto light_kernel = Tempest::VonMisesFisherPDF(Tempest::ParabolicToCartesianCoordinates(orig_light_dirs[orig_light_idx]), concentration, resample_dir[resampled_light_idx])*rcp_pdf;
                    float lum = orig_lv_lum_slice[orig_view_idx*orig_light_count + orig_light_idx];
                    sample += view_kernel*light_kernel*lum;
                    if(orig_view_idx == 0)
                        light_prob += light_kernel;
                }

                view_prob += view_kernel;

                dbg_estimate += Tempest::ParabolicToCartesianCoordinates(orig_light_dirs[orig_view_idx]).z;
            }

            dbg_estimate *= rcp_pdf/Tempest::MathPi;

            sample /= light_prob*view_prob;
        }
}

void DownsampleLightView(const Tempest::Vector2* orig_light_dirs, uint32_t orig_light_count, float* orig_lv_lum_slice, Tempest::Vector2* downsample_light_dirs, uint32_t downsample_light_count, float* downsampled_lum_slice)
{
    TGE_ASSERT(downsample_light_count < orig_light_count, "Invalid light view count");
    uint32_t last_light_dir = ~0u;
    float step_ratio = (float)(orig_light_count - 1)/(downsample_light_count - 1);

    std::unique_ptr<uint32_t[]> resample_idx(new uint32_t[downsample_light_count]);

    uint32_t min_value = 0;
    unsigned seed = 42;
    for(uint32_t light_idx = 0; light_idx < downsample_light_count; ++light_idx)
    {
        uint32_t max_value = Mini((uint32_t)Tempest::FastCeil(step_ratio*(light_idx + 1)), orig_light_count);
        TGE_ASSERT(min_value != max_value, "Broken index compute procedure");

        uint32_t sample_idx;

        do
        {
            sample_idx = Tempest::FastUintRand(min_value, max_value, seed);
        } while(sample_idx == last_light_dir); // we don't want overlapping, so rejection sample
        last_light_dir = sample_idx;

        downsample_light_dirs[light_idx] = orig_light_dirs[sample_idx];
        resample_idx[light_idx] = sample_idx;

        min_value = max_value;
    }

    for(uint32_t view_idx = 0; view_idx < downsample_light_count; ++view_idx)
        for(uint32_t light_idx = 0; light_idx < downsample_light_count; ++light_idx)
        {
            downsampled_lum_slice[view_idx*downsample_light_count + light_idx] = orig_lv_lum_slice[resample_idx[view_idx]*orig_light_count + resample_idx[light_idx]];
        }
}

void NullTestFillBTF(const LeastSquaresFitOptions& opts, const OptimizationParameters& sggx_parameters, Tempest::BTF* btf)
{
	BRDFEvaluator brdf_evaluator{ btf->LightsParabolic, btf->LightCount, opts.MultiScatteringBounceCount, opts.Fresnel };
	auto light_count = btf->LightCount;

	const size_t opt_params_count = TGE_FIXED_ARRAY_SIZE(sggx_parameters.ParametersArray);
	FittingType conv_params[opt_params_count];

	for(uint32_t i = 0; i < opt_params_count; ++i)
	{
		conv_params[i] = sggx_parameters.ParametersArray[i];
	}
	
	for(uint32_t idx = 0, idx_end = light_count*light_count; idx < idx_end; ++idx)
	{
		uint32_t view_idx = idx / btf->LightCount;
		uint32_t light_idx = idx % btf->LightCount;

        auto brdf_value = brdf_evaluator(0, idx, conv_params);
		reinterpret_cast<float*>(btf->LeftSingularU)[idx] = brdf_value;

        auto btf_value = Tempest::BTFFetchSpectrum(btf, light_idx, view_idx, 0, 0);
        TGE_ASSERT(brdf_value == Array(btf_value)[0], "Invalid generated BTF");
	}
}

bool SymmetryTest(const Tempest::BTF* btf, const OptimizationParameters& sggx_parameters)
{
    Tempest::RTSGGXSurface rt_material;
    rt_material.Specular = Tempest::ToSpectrum(sggx_parameters.Parameters.Specular);
    rt_material.Diffuse = Tempest::ToSpectrum(sggx_parameters.Parameters.Diffuse);
    Tempest::Vector2 sggx_stddev{ sggx_parameters.Parameters.StandardDeviation.x, sggx_parameters.Parameters.StandardDeviation.y };
    Tempest::Vector2 rev_sggx_stddev{ sggx_stddev.y, sggx_stddev.x };
    
    auto quat = Tempest::ToQuaternion(sggx_parameters.Parameters.Euler);
    rt_material.SGGXBasis = reinterpret_cast<Tempest::Vector4&>(quat);

    auto sggx_basis = Tempest::ToMatrix3(quat);

    Tempest::SampleData sample_data;
    sample_data.Material = &rt_material;
    sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

    bool is_symmetric = true;

    for(uint32_t view_idx = 0; view_idx < btf->LightCount; ++view_idx)
        for(uint32_t light_idx = 0; light_idx < btf->LightCount; ++light_idx)
        {
            sample_data.IncidentLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
            sample_data.OutgoingLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);

            rt_material.StandardDeviation = sggx_stddev;

            sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_stddev.x, sggx_stddev.y, 1.0f }, sggx_basis, sample_data.IncidentLight);
            Tempest::Spectrum r0 = Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);
        
            rt_material.StandardDeviation = rev_sggx_stddev;
            
            sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ rev_sggx_stddev.x, rev_sggx_stddev.y, 1.0f }, sggx_basis, sample_data.IncidentLight);
            Tempest::Spectrum r1 = Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);

            is_symmetric &= Tempest::ApproxEqual(r0, r1, 1e-2f);
        }

    return is_symmetric;
}

void LeastSquaresFitSGGX(uint32_t id, Tempest::ThreadPool& pool, const Tempest::BTF* btf, float* orig_lv_lum_slice, const LeastSquaresFitOptions& opts, const OptimizationParameters& parameters, OptimizationParameters* opt_parameters, float* out_rmse)
{
    static_assert(sizeof(Tempest::RTMicrofacetMaterial) == sizeof(RTMicrofacetMaterialProxy), "Invalid proxy class");
    static_assert(sizeof(Tempest::RTSGGXSurface) == sizeof(RTSGGXSurfaceProxy), "Invalid proxy class");

    const size_t opt_params_count = TGE_FIXED_ARRAY_SIZE(opt_parameters->ParametersArray);
    FittingType conv_params[opt_params_count];


    uint32_t fit_params_count = TGE_FIXED_ARRAY_SIZE(parameters.ParametersArray);

    if(opts.Flags & LSF_OPTON_DIFFUSE)
    {
        for(uint32_t i = 0; i < opt_params_count; ++i)
        {
            conv_params[i] = parameters.ParametersArray[i];
        }
    }
    else
    {
        uint32_t idx = 0;
        conv_params[idx++] = parameters.Parameters.StandardDeviation.x;
        conv_params[idx++] = parameters.Parameters.StandardDeviation.y;
        conv_params[idx++] = parameters.Parameters.Specular;
        conv_params[idx++] = parameters.Parameters.Euler.x;
        conv_params[idx++] = parameters.Parameters.Euler.y;
        conv_params[idx++] = parameters.Parameters.Euler.z;
        TGE_ASSERT(fit_params_count == idx + 1, "invalid parameter count");
        fit_params_count = idx;
    }

    if(opts.Flags & LSF_OPTION_DISABLE_DIRECTION_FITTING)
    {
        fit_params_count = fit_params_count - TGE_FIXED_ARRAY_SIZE(parameters.Parameters.Euler.Components);
    }

    Tempest::FitStatistics<FittingType> stats;

	Tempest::LevenbergMarquardtSettings<FittingType> settings;
	settings.BreakOnSmallStep = 1e-3f;

	auto params_ptr = conv_params,
		 opt_params_ptr = conv_params;

    uint32_t sampled_light_count;
    float* sampled_lv_lum_slice;
    Tempest::Vector2* sampled_lights;

	auto btf_light_count = btf->LightCount;
    //     view_count = btf->LightCount;

    std::unique_ptr<Tempest::Vector2[]> downsampled_light_dirs;
    std::unique_ptr<float[]> downsampled_lv_lum_slice;

    uint32_t lv_size;
    if(opts.Flags & LSF_OPTION_FIT_TOP)
    {
        uint32_t idx = 0; 
        float len = 1.0f;

        auto* light_parabolic = btf->LightsParabolic;
        for(uint32_t view_idx = 0; view_idx < btf_light_count; ++view_idx)
        {
            auto cur_len = Tempest::Length(light_parabolic[view_idx]);
            if(cur_len < len)
            {
                len = cur_len;
                idx = view_idx;
            }
        }

        downsampled_lv_lum_slice = std::unique_ptr<float[]>(new float[btf_light_count]);

        std::copy_n(orig_lv_lum_slice + idx*btf_light_count, btf_light_count, downsampled_lv_lum_slice.get());

        sampled_light_count = btf_light_count;
        sampled_lights = btf->LightsParabolic;
        sampled_lv_lum_slice = downsampled_lv_lum_slice.get();
        lv_size = sampled_light_count;
    }
    else if(btf_light_count > opts.DownSampleLightView)
    {
        Tempest::Log(Tempest::LogLevel::Info, "Downsampling selected for fitting: ", opts.DownSampleLightView);

        downsampled_light_dirs = std::unique_ptr<Tempest::Vector2[]>(new Tempest::Vector2[opts.DownSampleLightView]);
        downsampled_lv_lum_slice = std::unique_ptr<float[]>(new float[opts.DownSampleLightView*opts.DownSampleLightView]);

        if(opts.Flags & LSF_OPTION_FILTER_DOWNSAMPLING)
            DownsampleFilterLightView(btf->LightsParabolic, btf_light_count, orig_lv_lum_slice, downsampled_light_dirs.get(), opts.DownSampleLightView, downsampled_lv_lum_slice.get());
        else
            DownsampleLightView(btf->LightsParabolic, btf_light_count, orig_lv_lum_slice, downsampled_light_dirs.get(), opts.DownSampleLightView, downsampled_lv_lum_slice.get());

        sampled_light_count = opts.DownSampleLightView;
        sampled_lights = downsampled_light_dirs.get();
        sampled_lv_lum_slice = downsampled_lv_lum_slice.get();
        lv_size = sampled_light_count*sampled_light_count;
    }
    else
    {
        sampled_light_count = btf_light_count;
        sampled_lights = btf->LightsParabolic;
        sampled_lv_lum_slice = orig_lv_lum_slice;
        lv_size = sampled_light_count*sampled_light_count;
    }

	#ifdef CUDA_ACCELERATED_FITTING
        if(opts.Flags & LSF_OPTION_CUDA)
        {
            LeastSquaresFitSGGXCuda(sampled_lights, sampled_light_count, sampled_lv_lum_slice, opts,
                                    params_ptr, fit_params_count, &opt_params_ptr, &stats, &settings);
        }
        else
	#endif
        {
            

            if(opts.Flags & LSF_OPTON_DIFFUSE)
            {
                BRDFEvaluator evaluate_brdf{ sampled_lights, sampled_light_count, opts.MultiScatteringBounceCount, opts.Fresnel };
                auto approx_jacobian = Tempest::CreateParallelFDComputeResidualJacobian<float, FittingType>(sampled_lv_lum_slice, TGE_FIXED_ARRAY_SIZE(parameters.ParametersArray), 1e-4f, evaluate_brdf);

                auto evaluator = Tempest::CreateParallelResidualEvaluator<float, FittingType>(id, pool, sampled_lv_lum_slice, lv_size, evaluate_brdf, approx_jacobian);

                Tempest::LevenbergMarquardtCurveFit(evaluator,
                                                    1e-6, CurveFitSteps,
                                                    params_ptr,
                                                    fit_params_count,
                                                    AcceptStepConstraints(),
                                                    &opt_params_ptr,
                                                    &stats,
                                                    &settings);
            }
            else
            {
                BRDFEvaluatorSpecular evaluate_brdf{ sampled_lights, sampled_light_count, opts.MultiScatteringBounceCount, opts.Fresnel };
                auto approx_jacobian = Tempest::CreateParallelFDComputeResidualJacobian<float, FittingType>(sampled_lv_lum_slice, TGE_FIXED_ARRAY_SIZE(parameters.ParametersArray) - 1, 1e-4f, evaluate_brdf);

                auto evaluator = Tempest::CreateParallelResidualEvaluator<float, FittingType>(id, pool, sampled_lv_lum_slice, lv_size, evaluate_brdf, approx_jacobian);

                Tempest::LevenbergMarquardtCurveFit(evaluator,
                                                    1e-6, CurveFitSteps,
                                                    params_ptr,
                                                    fit_params_count,
                                                    AcceptStepSpecularConstraints(),
                                                    &opt_params_ptr,
                                                    &stats,
                                                    &settings);
            }

        }

    if(opts.Flags & LSF_OPTON_DIFFUSE)
    {
        for(uint32_t i = 0; i < opt_params_count; ++i)
        {
            opt_parameters->ParametersArray[i] = static_cast<float>(conv_params[i]);
        }
    }
    else
    {
        uint32_t idx = 0;
        opt_parameters->Parameters.StandardDeviation.x = static_cast<float>(conv_params[idx++]);
        opt_parameters->Parameters.StandardDeviation.y = static_cast<float>(conv_params[idx++]);
        opt_parameters->Parameters.Specular = static_cast<float>(conv_params[idx++]);
        opt_parameters->Parameters.Diffuse = 0.0f;
        opt_parameters->Parameters.Euler.x = static_cast<float>(conv_params[idx++]);
        opt_parameters->Parameters.Euler.y = static_cast<float>(conv_params[idx++]);
        opt_parameters->Parameters.Euler.z = static_cast<float>(conv_params[idx++]);
        TGE_ASSERT(fit_params_count == idx, "invalid parameter count");
    }
	

    FittingType rmse;
    if(btf_light_count > opts.DownSampleLightView || (opts.Flags & LSF_OPTION_FIT_TOP))
    {
        FittingType sq_sum = 0.0f;
        BRDFEvaluator brdf_evaluator;
        brdf_evaluator.LightDirections = btf->LightsParabolic;
        brdf_evaluator.LightCount = btf_light_count;
        brdf_evaluator.BounceCount = opts.MultiScatteringBounceCount;
        brdf_evaluator.Fresnel = 1.0f;
        for(uint32_t idx = 0, idx_end = btf_light_count*btf_light_count; idx < idx_end; ++idx)
        {
            auto value = brdf_evaluator(0, idx, conv_params);
            FittingType residual = value - orig_lv_lum_slice[idx];
            sq_sum += residual*residual;
        }

        rmse = std::sqrt(sq_sum / (btf_light_count*btf_light_count));
    }
    else
    {
        rmse = std::sqrt(stats.MeanSequaredError);
    }

    Tempest::Log(Tempest::LogLevel::Info, "Iteration required for fitting: ", stats.EffectiveIterationCount, "(", stats.IterationCount, "). RMSE: ", rmse, ".");
    *out_rmse = static_cast<float>(rmse);
}

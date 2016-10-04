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

#define EXPORT_EVALUATOR __device__
#define ILLUMINATION_MODEL_STATIC_IMPLEMENTATION
#define ILLUMINATION_MODEL_IMPLEMENTATION
#include "tempest/graphics/ray-tracing/illumination-models.hh"

#include "tempest/compute/fitting-cuda.hh"
#include "tempest/math/fitting.hh"

#include "svbrdf-fitting.hh"
#include "svbrdf-lsq-fitting-internal.hh"

void LeastSquaresFitSGGXCuda(const Tempest::Vector2* sampled_lights, uint32_t sampled_light_count, float* sampled_lv_lum_slice, const LeastSquaresFitOptions& opts,
                             const FittingType* parameters, uint32_t parameters_count, FittingType** opt_parameters, Tempest::FitStatistics<FittingType>* stats,
							 Tempest::LevenbergMarquardtSettings<FittingType>* settings)
{
     uint32_t lv_size = sampled_light_count*sampled_light_count;

    Tempest::Log(Tempest::LogLevel::Info, "Cuda accelerated fitting enabled");
    Tempest::Vector2* light_dirs;
	size_t light_dir_size = sampled_light_count*sizeof(light_dirs[0]);
	auto status = cudaMalloc(reinterpret_cast<void**>(&light_dirs), light_dir_size);
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate light dir matrix");
	status = cudaMemcpy(light_dirs, sampled_lights, light_dir_size, cudaMemcpyHostToDevice);
	TGE_ASSERT(status == cudaSuccess, "Failed to copy light dir matrix");
	
    if(opts.Flags & LSF_OPTON_DIFFUSE)
    {
        BRDFEvaluator evaluate_brdf{ light_dirs, sampled_light_count, opts.MultiScatteringBounceCount, opts.Fresnel };

        auto approx_jacobian = Tempest::CreateFDComputeResidualJacobianCuda<float, FittingType>(sampled_lv_lum_slice, parameters_count, 1e-4f, evaluate_brdf);

        auto evaluator = Tempest::CreateCUDAResidualEvaluator<float, FittingType>(sampled_lv_lum_slice, lv_size, parameters_count, evaluate_brdf, approx_jacobian);

        Tempest::LevenbergMarquardtCurveFit(evaluator,
                                            1e-6, CurveFitSteps,
                                            parameters,
                                            parameters_count,
                                            AcceptStepConstraints(),
                                            opt_parameters,
                                            stats,
                                            settings);
    }
    else
    {
        BRDFEvaluatorSpecular evaluate_brdf{ light_dirs, sampled_light_count, opts.MultiScatteringBounceCount, opts.Fresnel };

        auto approx_jacobian = Tempest::CreateFDComputeResidualJacobianCuda<float, FittingType>(sampled_lv_lum_slice, parameters_count, 1e-4f, evaluate_brdf);

        auto evaluator = Tempest::CreateCUDAResidualEvaluator<float, FittingType>(sampled_lv_lum_slice, lv_size, parameters_count, evaluate_brdf, approx_jacobian);

        Tempest::LevenbergMarquardtCurveFit(evaluator,
                                            1e-6, CurveFitSteps,
                                            parameters,
                                            parameters_count,
                                            AcceptStepSpecularConstraints(),
                                            opt_parameters,
                                            stats,
                                            settings);
    }
}
        
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

#ifndef _TEMPEST_FITTING_HH_
#define _TEMPEST_FITTING_HH_

#include "tempest/math/numerical-methods.hh"
#include "tempest/math/matrix-variadic.hh"
#include "tempest/utils/memory.hh"
#include "tempest/utils/threads.hh"
#include "tempest/math/functions.hh"

namespace Tempest
{
template<class TFloat>
struct FitStatistics
{
	uint32_t	IterationCount = 0;
    uint32_t    EffectiveIterationCount = 0;
	TFloat		MeanSequaredError = 0.0f;
};

template<class TFloat>
struct LevenbergMarquardtSettings
{
	TFloat		BreakOnSmallStep = 1e-6f;
	TFloat		InitialMinDamping = 1.0f;
	TFloat		InitialDampingMultiplier = 0.75f;
	TFloat		DampingFailureMultiplier = 2.0f;
};

template<class TFloat, class TFunction, class TAcceptStep>
void ParallelFDLevenbergMarquardtCurveFit(uint32_t id, ThreadPool& pool,
                                          TFloat mean_sq_error, uint32_t max_steps,
                                          TFloat* output_values,
                                          uint32_t value_pair_count,
                                          const TFloat* input_parameters,
                                          uint32_t parameter_count,
                                          TFunction func, TFloat finite_diff_step,
                                          TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
                                          TFloat** opt_parameters);

template<class TFloat, class TFunction, class TMaxStep>
void ParallelFDGaussNewtonCurveFit(uint32_t id, ThreadPool& pool,
                                   TFloat mean_sq_error, uint32_t max_steps,
                                   TFloat* output_values,
                                   uint32_t value_pair_count,
                                   const TFloat* input_parameters,
                                   uint32_t parameter_count,
                                   TFunction func, TFloat finite_diff_step,
                                   TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                                   TFloat** opt_parameters);

template<class TFloat, class TFunction, class TMaxStep>
void ParallelFDGradientDescentCurveFit(uint32_t id, ThreadPool& pool,
                                       TFloat mean_sq_error, uint32_t max_steps,
                                       TFloat* output_values,
                                       uint32_t value_pair_count,
                                       const TFloat* input_parameters,
                                       uint32_t parameter_count,
                                       TFunction func, TFloat finite_diff_step,
                                       TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                                       TFloat** opt_parameters);

template<class TDataFloat, class TReturnFloat, class TFunction, class TJacobian>
struct ParallelResidualEvaluator
{
    uint32_t    MainThreadId;
    ThreadPool& Pool;
    TDataFloat* OutputValues;
    uint32_t    SampleCount;
    TFunction   Function;
    TJacobian   Jacobian;

    ParallelResidualEvaluator(uint32_t id, ThreadPool& pool, TDataFloat* output_values, uint32_t sample_count, TFunction func, TJacobian jacob)
        :   MainThreadId(id),
            Pool(pool),
            OutputValues(output_values),
            SampleCount(sample_count),
            Function(func),
            Jacobian(jacob)
    {
    }
    
    void computeResidual(TReturnFloat* parameters, TReturnFloat* out_residual_sq_sum, TReturnFloat** out_residual_vec)
    {
        TReturnFloat* residual_vec = *out_residual_vec;

        uint32_t thread_count = Pool.getThreadCount();
        auto* partial_residual_sq_sum = TGE_TYPED_ALLOCA(TReturnFloat, thread_count);
        memset(partial_residual_sq_sum, 0, thread_count*sizeof(partial_residual_sq_sum[0]));

        auto* output_values = OutputValues;
        auto& func = Function;

        auto comp_residual = CreateParallelForLoop(SampleCount, 1, 
                                                   [&func, output_values, parameters, residual_vec, partial_residual_sq_sum](uint32_t worker_id, uint32_t start_idx, uint32_t chunk_size)
                                                   {
                                                       for(uint32_t sample_idx = start_idx, sample_idx_end = start_idx + chunk_size; sample_idx < sample_idx_end; ++sample_idx)
                                                       {
                                                           auto output_value = output_values[sample_idx];
                                                           auto func_value = func(worker_id, sample_idx, parameters);
                                                           auto residual = output_value - func_value;
                                                           partial_residual_sq_sum[worker_id] += residual*residual;
		                                                   residual_vec[sample_idx] = residual;
                                                       }
                                                   });

        Pool.enqueueTask(&comp_residual);

        Pool.waitAndHelp(MainThreadId, &comp_residual);

        auto residual_sq_sum = partial_residual_sq_sum[0];
        for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
        {
            residual_sq_sum += partial_residual_sq_sum[thread_idx];
        }

        *out_residual_sq_sum = residual_sq_sum;
    }

    void computeJacobianMatrix(TReturnFloat* parameters, uint32_t parameter_count, TReturnFloat* residual_vec, TReturnFloat** out_jacob_transpose_matrix)
    {
        auto* jacob_transpose_matrix = *out_jacob_transpose_matrix;
        auto& jacob = Jacobian;
        auto comp_jacob = CreateParallelForLoop(SampleCount, 1, 
                                                [&jacob, jacob_transpose_matrix, parameters, residual_vec, parameter_count](uint32_t worker_id, uint32_t start_idx, uint32_t chunk_size)
                                                {
                                                    for(uint32_t sample_idx = start_idx, sample_idx_end = start_idx + chunk_size; sample_idx < sample_idx_end; ++sample_idx)
                                                    {
                                                        auto* res_matrix = jacob_transpose_matrix + sample_idx*parameter_count;
                                                        jacob(worker_id, sample_idx, parameters, residual_vec[sample_idx], &res_matrix);
                                                    }
                                                });

        Pool.enqueueTask(&comp_jacob);

        Pool.waitAndHelp(MainThreadId, &comp_jacob);
    }

    void computeWeightedResidual(TReturnFloat* output_weights, TReturnFloat* parameters, TReturnFloat* out_residual_sq_sum, TReturnFloat** out_residual_vec, TReturnFloat** out_weighted_vec)
    {
        auto* residual_vec = *out_residual_vec;
	    auto* weighted_residual_vec = *out_weighted_vec;

        uint32_t thread_count = Pool.getThreadCount();
        auto* partial_residual_sq_sum = TGE_TYPED_ALLOCA(TReturnFloat, thread_count);
        memset(partial_residual_sq_sum, 0, thread_count*sizeof(partial_residual_sq_sum[0]));

        auto output_values = OutputValues;
        auto& func = Function;
        auto comp_residual = CreateParallelForLoop(SampleCount, 1,
                                                   [output_values, output_weights, &func, partial_residual_sq_sum, parameters, residual_vec, weighted_residual_vec](uint32_t worker_id, uint32_t start_idx, uint32_t chunk_size)
                                                   {
                                                       for(uint32_t sample_idx = start_idx, sample_idx_end = start_idx + chunk_size; sample_idx < sample_idx_end; ++sample_idx)
                                                       {
                                                           auto output_value = output_values[sample_idx];
                                                           auto func_value = func(worker_id, sample_idx, parameters);
													       auto weight = output_weights[sample_idx];
                                                           auto residual = (output_value - func_value);
                                                           partial_residual_sq_sum[worker_id] += weight*residual*residual;
		                                                   residual_vec[sample_idx] = residual;
													       weighted_residual_vec[sample_idx] = weight*residual;
                                                       }
                                                   });

        Pool.enqueueTask(&comp_residual);

        Pool.waitAndHelp(MainThreadId, &comp_residual);

        auto residual_sq_sum = partial_residual_sq_sum[0];
        for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
        {
            residual_sq_sum += partial_residual_sq_sum[thread_idx];
        }

        *out_residual_sq_sum = residual_sq_sum;
    }
};

template<class TDataFloat, class TReturnFloat = TDataFloat, class TFunction, class TJacobian>
ParallelResidualEvaluator<TDataFloat, TReturnFloat, TFunction, TJacobian> CreateParallelResidualEvaluator(uint32_t id, ThreadPool& pool, TDataFloat* output_values, uint32_t sample_count, TFunction func, TJacobian jacob)
{
    return ParallelResidualEvaluator<TDataFloat, TReturnFloat, TFunction, TJacobian>(id, pool, output_values, sample_count, func, jacob);
}

template<class TFloat, class TFunction, class TJacobian>
struct ResidualEvaluator
{
    TFloat*     OutputValues;
    uint32_t    SampleCount;
    TFunction   Function;
    TJacobian   Jacobian;

    ResidualEvaluator(TFloat* output_values, uint32_t sample_count, TFunction func, TJacobian jacob)
        :   OutputValues(output_values),
            SampleCount(sample_count),
            Function(func),
            Jacobian(jacob)
    {
    }

    void computeResidual(TFloat* parameters, TFloat* out_residual_sq_sum, TFloat** out_residual_vec)
    {
        TFloat residual_sq_sum = 0.0f;
        TFloat* residual_vec = *out_residual_vec;
        for(uint32_t sample_idx = 0; sample_idx < SampleCount; ++sample_idx)
        {
            TFloat output_value = OutputValues[sample_idx];
            TFloat func_value = Function(sample_idx, parameters);
            TFloat residual = output_value - func_value;
            residual_sq_sum += residual*residual;
		    residual_vec[sample_idx] = residual;
        }
        *out_residual_sq_sum = residual_sq_sum;
    }

    void computeJacobianMatrix(TFloat* parameters, uint32_t parameter_count, TFloat* residual_vec, TFloat** out_jacob_transpose_matrix)
    {
        for(uint32_t sample_idx = 0; sample_idx < SampleCount; ++sample_idx)
        {
            TFloat* res_matrix = *out_jacob_transpose_matrix + sample_idx*parameter_count;
            Jacobian(sample_idx, parameters, residual_vec[sample_idx], &res_matrix);
        }
    }
};

template<class TFloat, class TFunction, class TJacobian>
ResidualEvaluator<TFloat, TFunction, TJacobian> CreateResidualEvaluator(TFloat* output_values, uint32_t sample_count, TFunction func, TJacobian jacob)
{
    return ResidualEvaluator<TFloat, TFunction, TJacobian>(output_values, sample_count, func, jacob);
}

template<class TFunc>
float ReduceToNDF(uint32_t reduce_sphere_samples, const Tempest::Vector3& norm, TFunc func)
{
    return Tempest::StratifiedMonteCarloIntegratorSphere(reduce_sphere_samples,
        [&func, &norm](const Tempest::Vector3& inc_light)
        {
            auto out_light = Tempest::Reflect(inc_light, norm);
            return func(inc_light, out_light);
        })/(4.0f*Tempest::MathPi);
}

template<class TFunc>
Matrix3 PerformPCA(uint32_t id, ThreadPool& pool, bool hemisphere, uint32_t sphere_samples, uint32_t plane_samples, TFunc sample_ndf)
{
    float sqrt_samples = sqrtf((float)sphere_samples);
    uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
    TGE_ASSERT((sphere_samples % sqrt_samples_i) == 0, "Invalid sample count");

    struct PartialMostDominant
    {
        float       NDFValue;
        Vector3     Axis;
    };

    auto thread_count = pool.getThreadCount();
    auto* partial_most_prominent = TGE_TYPED_ALLOCA(PartialMostDominant, thread_count);
    
    PartialMostDominant fill_value {};
    fill_value.Axis.z = 1.0f;
    fill_value.NDFValue = sample_ndf(fill_value.Axis);

    std::fill(partial_most_prominent, partial_most_prominent + thread_count, fill_value);

    unsigned seed = 1;
    auto parallel_search = Tempest::CreateParallelForLoop2D(sqrt_samples_i, sqrt_samples_i, 8,
        [&sample_ndf, sqrt_samples, &seed, partial_most_prominent, hemisphere](uint32_t worker_id, uint32_t x, uint32_t y)
        {
            float x_rand = (x + Tempest::FastFloatRand(seed))/sqrt_samples;
            float y_rand = (y + Tempest::FastFloatRand(seed))/sqrt_samples;

            auto norm = hemisphere ? Tempest::UniformSampleHemisphere(x_rand, y_rand) : Tempest::UniformSampleSphere(x_rand, y_rand);
            auto cur_ndf_value = sample_ndf(norm);

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
    Vector3 most_prominent_norm = first_most_prominent.Axis;
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

    Matrix3 basis;
    basis.makeBasis(most_prominent_norm);

    auto rot = Matrix2::rotation(2.0f*MathPi/(plane_samples - 1));

    ndf_value = 0.0f;
    Vector3 most_prominent_tan{ 1.0f, 0.0f, 0.0f };

    for(size_t theta = 0; theta < plane_samples; ++theta)
    {
        auto& tan = basis.tangent();    

        auto cur_ndf_value = sample_ndf(tan);
        if(cur_ndf_value > ndf_value)
        {
            most_prominent_tan = tan;
            ndf_value = cur_ndf_value;
        }

        basis.rotateTangentPlane(rot);
    }

    auto binorm = Cross(most_prominent_norm, most_prominent_tan);
    return Matrix3(most_prominent_tan, binorm, most_prominent_norm);
}

// This is the version where you can specify the variance of measurements
// but i didn't test it. So yeah - use it at your own risk.
template<class TFloat, class TResidualEvaluator, class TMaxStep>
void GaussNewtonCurveFit(TResidualEvaluator& eval_residual,
                         TFloat mean_sq_error, uint32_t max_steps,
                         TFloat* output_weights,
                         const TFloat* input_parameters,
                         uint32_t parameter_count,
                         TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                         TFloat** opt_parameters)
{
//#define ARMIJO_GOLDSTEIN
    uint32_t value_pair_count = eval_residual.SampleCount;
                         
    const uint32_t jacobian_matrix_size = parameter_count*value_pair_count;
    const uint32_t jacobian_sq_matrix_size = parameter_count*parameter_count;
    const uint32_t inverse_matrix_size = jacobian_sq_matrix_size;

	uint32_t total_data_size = 2*value_pair_count + 2*parameter_count + jacobian_matrix_size + jacobian_sq_matrix_size + inverse_matrix_size + jacobian_matrix_size;

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
	auto* residual_vec = data_offset;
	data_offset += value_pair_count;
	auto* weighted_residual_vec = data_offset;
	data_offset += value_pair_count;
    auto* jacob_transpose_matrix = data_offset;
    data_offset += jacobian_matrix_size;
    auto* approx_hessian_matrix = data_offset;
    data_offset += jacobian_sq_matrix_size;
    auto* approx_hessian_inverse_matrix = data_offset;
    data_offset += inverse_matrix_size;
    auto* transform_param_matrix = data_offset;
    data_offset += jacobian_matrix_size;
    auto* param_step = data_offset;
    data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;

    TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

    eval_residual.computeWeightedResidual(output_weights, *opt_parameters, &last_residual_sq_sum, &residual_vec, &weighted_residual_vec);

    for(uint32_t iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
        eval_residual.computeJacobianMatrix(*opt_parameters, parameter_count, residual_vec, &jacob_transpose_matrix);

        // Gauss-Newton step
        Tempest::MatrixTransformCovarianceDiagonal(jacob_transpose_matrix, parameter_count, value_pair_count, 
												   output_weights, value_pair_count, &approx_hessian_matrix);
        Tempest::MatrixInverse(approx_hessian_matrix, parameter_count, &approx_hessian_inverse_matrix);
        Tempest::MatrixMultiply(approx_hessian_inverse_matrix, parameter_count, parameter_count,
                                jacob_transpose_matrix, parameter_count, value_pair_count, &transform_param_matrix);
        Tempest::MatrixTransformVector(transform_param_matrix, parameter_count, value_pair_count, weighted_residual_vec, value_pair_count, &param_step);

        const TFloat tau = 0.5f;
        
        TFloat step_size = max_step(parameter_count, *opt_parameters, param_step);
        TFloat break_criteria = 1e-6f;

        TFloat residual_sq_sum = 0.0f;
        do
        {
            // Stationary point reached - completely pointless to continue
            if(step_size < break_criteria)
            {
                return;
            }

            for(uint32_t param_idx = 0; param_idx < parameter_count; ++param_idx)
            {
                tmp_parameters[param_idx] = (*opt_parameters)[param_idx] - step_size*param_step[param_idx];
            }
                        
            eval_residual.computeWeightedResidual(output_weights, tmp_parameters, &residual_sq_sum, &residual_vec, &weighted_residual_vec);

            step_size *= tau;
        } while(last_residual_sq_sum <= residual_sq_sum);
        last_residual_sq_sum = residual_sq_sum;

        std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
    }
}

template<class TFloat>
struct ConstantStep
{
    TFloat MinimumStep;

    ConstantStep(TFloat max_step)
        :   MinimumStep(max_step) {}

    TFloat operator()(uint32_t parameter_count, const TFloat* input_parameters, const TFloat* step_size)
    {
        return MinimumStep;
    }
};

template<class TFloat, class TResidualEvaluator, class TAcceptStep>
void LevenbergMarquardtCurveFit(TResidualEvaluator& eval_residual,
                                TFloat mean_sq_error, uint32_t max_steps,
                                const TFloat* input_parameters,
                                uint32_t parameter_count,
                                TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
                                TFloat** opt_parameters,
								FitStatistics<TFloat>* stats = nullptr,
								LevenbergMarquardtSettings<TFloat>* in_settings = nullptr)
{
    auto value_pair_count = eval_residual.SampleCount;
    const uint32_t jacobian_matrix_size = parameter_count*value_pair_count;
    const uint32_t hessian_size = parameter_count*parameter_count;

//#define BFGS
//#define DFP

	uint32_t total_data_size = value_pair_count + jacobian_matrix_size +  4*parameter_count + 3*hessian_size;

#if defined(BFGS)
	total_data_size += parameter_count + hessian_size;
#elif defined(DFP)
	total_data_size += parameter_count + 2*hessian_size;
#endif

	LevenbergMarquardtSettings<TFloat> settings;
	if(in_settings)
	{
		settings = *in_settings;
	}

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
	auto* residual_vec = data_offset;
	data_offset += value_pair_count;
    auto* jacob_transpose_matrix = data_offset;
    data_offset += jacobian_matrix_size;
    auto* param_step = data_offset;
	data_offset += parameter_count;
    auto* gradient = data_offset;
    data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;
    auto* tmp_diff = data_offset;
    data_offset += parameter_count;
    auto* approx_hessian_matrix = data_offset;
    data_offset += hessian_size;
    auto* damp_approx_hessian_matrix = data_offset;
    data_offset += hessian_size;
    auto* approx_hessian_inverse_matrix = data_offset;
    data_offset += hessian_size;

#if defined(BFGS) || defined(DFP)
	auto* gradient_step = data_offset;
	data_offset += parameter_count;
	auto* tmp_diff2 = tmp_parameters;
	auto* hessian_step = data_offset;
	data_offset += hessian_size;
#ifdef DFP
	auto* coef_matrix = data_offset;
	data_offset += hessian_size;
#endif
#endif

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

    eval_residual.computeResidual(*opt_parameters, &last_residual_sq_sum, &residual_vec);
    eval_residual.computeJacobianMatrix(*opt_parameters, parameter_count, residual_vec, &jacob_transpose_matrix);

    Tempest::MatrixSquare(jacob_transpose_matrix, parameter_count, value_pair_count, &approx_hessian_matrix);
    Tempest::MatrixTransformVector(jacob_transpose_matrix, parameter_count, value_pair_count, residual_vec, value_pair_count, &gradient);
    
    TFloat damping_multiplier = settings.DampingFailureMultiplier;
    const TFloat tau = settings.InitialDampingMultiplier;
    const TFloat break_criteria = settings.BreakOnSmallStep;

    TFloat damping = settings.InitialMinDamping;

    for(uint32_t i = 0; i < parameter_count; ++i)
    {
        damping = std::fmax(damping, approx_hessian_matrix[i*parameter_count + i]);
    }

    damping *= tau;

	uint32_t iter_step, effective_iter = 0;

	auto at_exit = CreateAtScopeExit([&last_residual_sq_sum, &iter_step, &effective_iter, value_pair_count, stats]
	{
		if(stats)
		{
			stats->IterationCount = iter_step;
            stats->EffectiveIterationCount = effective_iter;
			stats->MeanSequaredError = last_residual_sq_sum/value_pair_count;
		}
	});

    for(iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
        memcpy(damp_approx_hessian_matrix, approx_hessian_matrix, parameter_count*parameter_count*sizeof(damp_approx_hessian_matrix[0]));
        for(uint32_t i = 0; i < parameter_count; ++i)
        {
            damp_approx_hessian_matrix[i*parameter_count + i] += damping;
        }

	//#define CHOLESKY_DECOMPOSE

	#ifdef CHOLESKY_DECOMPOSE
		bool inverse_status = Tempest::MatrixCholeskyDecomposition(damp_approx_hessian_matrix, parameter_count, &approx_hessian_inverse_matrix);
	#else
        Tempest::MatrixInverse(damp_approx_hessian_matrix, parameter_count, &approx_hessian_inverse_matrix);
		bool inverse_status = std::isfinite(approx_hessian_inverse_matrix[0]);
	#endif
		if(!inverse_status)
        {
            damping *= damping_multiplier;
            damping_multiplier *= damping_multiplier;
        }
        else // damped Gauss-Newton step
        {
		#ifdef CHOLESKY_DECOMPOSE
			Tempest::MatrixTriangularSolve(approx_hessian_inverse_matrix, parameter_count, gradient, parameter_count, &param_step);
			Tempest::MatrixTriangularTransposeSolve(approx_hessian_inverse_matrix, parameter_count, param_step, parameter_count, &param_step);
		#else
            Tempest::MatrixTransformVector(approx_hessian_inverse_matrix, parameter_count, parameter_count, gradient, parameter_count, &param_step);
		#endif
            TFloat step_len = Tempest::VectorLength(param_step, parameter_count);

            Tempest::VectorAdd(*opt_parameters, parameter_count, break_criteria, &tmp_parameters);

            TFloat parameter_mag = Tempest::VectorLength(tmp_parameters, parameter_count);

            if(step_len <= break_criteria*(parameter_mag + break_criteria)) // Give up
                break;

            if(!accept_step(parameter_count, *opt_parameters, param_step))
            {
                damping *= damping_multiplier;
                damping_multiplier *= damping_multiplier;
                continue;
            }

            Tempest::VectorSubtract(*opt_parameters, parameter_count, param_step, parameter_count, &tmp_parameters);

            TFloat residual_sq_sum = 0.0f;
            eval_residual.computeResidual(tmp_parameters, &residual_sq_sum, &residual_vec);

            Tempest::VectorMultiply(param_step, parameter_count, damping, &tmp_diff);
            Tempest::VectorAdd(tmp_diff, parameter_count, gradient, parameter_count, &tmp_diff);

            TFloat predicted_gain = 0.5f*Tempest::VectorDot(param_step, parameter_count, tmp_diff, parameter_count);
            TFloat gain_ratio = 0.5f*(last_residual_sq_sum - residual_sq_sum)/predicted_gain;

            if(gain_ratio > 0.0f)
            {
                ++effective_iter;

                std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
                last_residual_sq_sum = residual_sq_sum;
                if(last_residual_sq_sum/value_pair_count <= mean_sq_error)
                    break;

                eval_residual.computeJacobianMatrix(*opt_parameters, parameter_count, residual_vec, &jacob_transpose_matrix);

				#if defined(BFGS)
					std::swap(gradient, gradient_step);
				#endif

				Tempest::MatrixTransformVector(jacob_transpose_matrix, parameter_count, value_pair_count, residual_vec, value_pair_count, &gradient);
				TFloat grad_len = Tempest::VectorLengthSquared(gradient, parameter_count);
                if(grad_len <= break_criteria*break_criteria)
                    break;

				#ifdef DFP
					Tempest::VectorNegate(param_step, parameter_count, &param_step);
					Tempest::VectorSubtract(gradient, parameter_count, gradient_step, parameter_count, &gradient_step);

					Tempest::MatrixIdentity(parameter_count, parameter_count, &hessian_step);

					TFloat coef = 1.0f/Tempest::VectorDot(gradient_step, parameter_count, param_step, parameter_count);
					Tempest::VectorOuterProduct(gradient_step, parameter_count, param_step, parameter_count, &coef_matrix);
					Tempest::MatrixMultiplyAdd(-coef, coef_matrix, parameter_count, parameter_count, hessian_step, parameter_count, parameter_count, &coef_matrix);
					Tempest::MatrixMultiply(coef_matrix, parameter_count, parameter_count, approx_hessian_matrix, parameter_count, parameter_count, &approx_hessian_matrix);
					Tempest::MatrixMultiply(approx_hessian_matrix, parameter_count, parameter_count, coef_matrix, parameter_count, parameter_count, &approx_hessian_matrix);
					Tempest::VectorOuterProduct(gradient_step, parameter_count, gradient_step, parameter_count, &hessian_step);
					Tempest::MatrixMultiplyAdd(coef, hessian_step, parameter_count, parameter_count, approx_hessian_matrix, parameter_count, parameter_count, &approx_hessian_matrix);
				#elif defined(BFGS)
					Tempest::VectorNegate(param_step, parameter_count, &param_step);
					Tempest::VectorSubtract(gradient, parameter_count, gradient_step, parameter_count, &gradient_step);
					
					Tempest::VectorOuterProduct(gradient_step, parameter_count, gradient_step, parameter_count, &hessian_step);
					TFloat term0_denom = Tempest::VectorDot(gradient_step, parameter_count, param_step, parameter_count);
					Tempest::MatrixMultiplyAdd(1.0f/term0_denom, hessian_step, parameter_count, parameter_count, approx_hessian_matrix, parameter_count, parameter_count, &approx_hessian_matrix);

					Tempest::MatrixTransformVector(approx_hessian_matrix, parameter_count, parameter_count, param_step, parameter_count, &tmp_diff);
					Tempest::VectorTransposeMatrixTransform(param_step, parameter_count, approx_hessian_matrix, parameter_count, parameter_count, &tmp_diff2);

					Tempest::VectorOuterProduct(tmp_diff, parameter_count, tmp_diff2, parameter_count, &hessian_step);
					TFloat term1_denom = Tempest::VectorDot(param_step, parameter_count, tmp_diff, parameter_count);
					Tempest::MatrixMultiplyAdd(-1.0f/term1_denom, hessian_step, parameter_count, parameter_count, approx_hessian_matrix, parameter_count, parameter_count, &approx_hessian_matrix);
				#else // Gauss-Newton
					Tempest::MatrixSquare(jacob_transpose_matrix, parameter_count, value_pair_count, &approx_hessian_matrix);
				#endif

				TFloat min_mul = 1.0f/3.0f;
                TFloat multiplier = std::fmax(min_mul, 1.0f - std::pow((2.0f*gain_ratio - 1.0f), 3.0f));
                damping *= multiplier;
                damping_multiplier = 2.0f;
            }
            else
            {
                damping *= damping_multiplier;
                damping_multiplier *= damping_multiplier;
            }
        }
    }
}

template<class TFloat, class TResidualEvaluator, class TMaxStep>
void GradientDescentCurveFit(TResidualEvaluator& eval_residual,
                             TFloat mean_sq_error, uint32_t max_steps,
                             const TFloat* input_parameters,
                             uint32_t parameter_count,
                             TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                             TFloat** opt_parameters)
{
//#define ARMIJO_GOLDSTEIN
    auto value_pair_count = eval_residual.SampleCount;
    const uint32_t jacobian_matrix_size = parameter_count*value_pair_count;

	uint32_t total_data_size = value_pair_count + 2*parameter_count + jacobian_matrix_size;

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
	auto* residual_vec = data_offset;
	data_offset += value_pair_count;
    auto* jacob_transpose_matrix = data_offset;
    data_offset += jacobian_matrix_size;
    auto* param_step = data_offset;
	data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

    eval_residual.computeResidual(*opt_parameters, &last_residual_sq_sum, &residual_vec);

    for(uint32_t iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
        eval_residual.computeJacobianMatrix(*opt_parameters, parameter_count, residual_vec, &jacob_transpose_matrix);

        // Gradient-Descent step
        Tempest::MatrixTransformVector(jacob_transpose_matrix, parameter_count, value_pair_count, residual_vec, value_pair_count, &param_step);

        const TFloat tau = 0.5f;
        
        TFloat step_size = max_step(parameter_count, *opt_parameters, param_step);
        TFloat break_criteria = 1e-8f; // Horribly bad method - expect really slow convergence because it tends to crash in the walls of the valley

        TFloat residual_sq_sum = 0.0f;
        do
        {
            // Stationary point reached - completely pointless to continue
            if(step_size < break_criteria)
            {
                return;
            }

            for(uint32_t param_idx = 0; param_idx < parameter_count; ++param_idx)
            {
                tmp_parameters[param_idx] = (*opt_parameters)[param_idx] - step_size*param_step[param_idx];
            }
                        
            eval_residual.computeResidual(tmp_parameters, &residual_sq_sum, &residual_vec);

            step_size *= tau;
        } while(last_residual_sq_sum < residual_sq_sum);
        last_residual_sq_sum = residual_sq_sum;

        std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
    }
}

template<class TFloat, class TResidualEvaluator, class TMaxStep>
void GaussNewtonCurveFit(TResidualEvaluator& eval_residual,
                         TFloat mean_sq_error, uint32_t max_steps,
                         const TFloat* input_parameters,
                         uint32_t parameter_count,
                         TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                         TFloat** opt_parameters)
{
//#define ARMIJO_GOLDSTEIN
    auto value_pair_count = eval_residual.SampleCount;

    const uint32_t jacobian_matrix_size = parameter_count*value_pair_count;
    const uint32_t jacobian_sq_matrix_size = parameter_count*parameter_count;
    const uint32_t inverse_matrix_size = jacobian_sq_matrix_size;

	uint32_t total_data_size = value_pair_count + 3*parameter_count + jacobian_matrix_size + jacobian_sq_matrix_size + inverse_matrix_size;

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
	auto* residual_vec = data_offset;
	data_offset += value_pair_count;
    auto* jacob_transpose_matrix = data_offset;
    data_offset += jacobian_matrix_size;
    auto* approx_hessian_matrix = data_offset;
    data_offset += jacobian_sq_matrix_size;
    auto* approx_hessian_inverse_matrix = data_offset;
    data_offset += inverse_matrix_size;
    auto* param_step = data_offset;
	data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;
    auto* gradient = data_offset;
    data_offset += parameter_count;

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

    eval_residual.computeResidual(*opt_parameters, &last_residual_sq_sum, &residual_vec);

    for(uint32_t iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
        eval_residual.computeJacobianMatrix(*opt_parameters, parameter_count, residual_vec, &jacob_transpose_matrix);

        Tempest::MatrixSquare(jacob_transpose_matrix, parameter_count, value_pair_count, &approx_hessian_matrix);
        Tempest::MatrixInverse(approx_hessian_matrix, parameter_count, &approx_hessian_inverse_matrix);
        if(!std::isfinite(approx_hessian_inverse_matrix[0]))
        {
            // Gradient-Descent step
            Tempest::MatrixTransformVector(jacob_transpose_matrix, parameter_count, value_pair_count, residual_vec, value_pair_count, &param_step);
        }
        else // Gauss-Newton step
        {
            Tempest::MatrixTransformVector(jacob_transpose_matrix, parameter_count, value_pair_count, residual_vec, value_pair_count, &gradient);
            Tempest::MatrixTransformVector(approx_hessian_inverse_matrix, parameter_count, parameter_count, gradient, parameter_count, &param_step);
        }
        const TFloat tau = 0.5f;
        
        TFloat step_size = max_step(parameter_count, *opt_parameters, param_step);
        TFloat break_criteria = 1e-6f;

        TFloat residual_sq_sum = 0.0f;
        do
        {
            // Stationary point reached - completely pointless to continue
            if(step_size < break_criteria)
            {
                return;
            }

            for(uint32_t param_idx = 0; param_idx < parameter_count; ++param_idx)
            {
                tmp_parameters[param_idx] = (*opt_parameters)[param_idx] - step_size*param_step[param_idx];
            }
                        
            eval_residual.computeResidual(tmp_parameters, &residual_sq_sum, &residual_vec);

            step_size *= tau;
        } while(last_residual_sq_sum <= residual_sq_sum);
        last_residual_sq_sum = residual_sq_sum;

    /*
        // Armijo-Goldstein
        bool continue_iterate = false;
        TFloat residual_sq_sum = 0.0f;
        const TFloat control_parameter = 0.5f;
        Tempest::MatrixMultiply(param_step, 1, parameter_count, jacob_transpose_matrix, parameter_count, value_pair_count, &local_slope);
        do
        {
            // Stationary point reached - completely pointless to continue
            if(step_size < break_criteria)
            {
                return;
            }

            for(uint32_t param_idx = 0; param_idx < parameter_count; ++param_idx)
            {
                tmp_parameters[param_idx] = (*opt_parameters)[param_idx] - step_size*param_step[param_idx];
            }

            residual_sq_sum = 0.0f;

            continue_iterate = false;
            for(uint32_t sample_idx = 0; sample_idx < value_pair_count; ++sample_idx)
            {
                TFloat output_value = output_values[sample_idx];
                TFloat func_value = func(sample_idx, tmp_parameters);
                TFloat residual = tmp_residual_vec[sample_idx] = output_value - func_value;
                residual_sq_sum += residual*residual;

                TFloat residual_change = residual_vec[sample_idx] - residual;
                TFloat step_threshold = step_size*control_parameter*local_slope[sample_idx];

				TFloat sign = Sign(step_threshold);
                if(sign*residual_change < sign*step_threshold)
                {
                    continue_iterate = true;
                    break;
                }
            }
            step_size *= tau;
        } while(continue_iterate);
        std::copy_n(tmp_residual_vec, value_pair_count, residual_vec);
        last_residual_sq_sum = residual_sq_sum;
    */

        std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
    }
}

template<class TFunctionFloat, class TReturnFloat, class TFunction>
struct ParallelFDComputeResidualJacobianCenterPoint
{
    TFunctionFloat* OutputValues;
    uint32_t		ParameterCount;
    TReturnFloat    FiniteDifferenceStep;
    TFunction		Function;
        
    void operator()(uint32_t worker_id, uint32_t idx, const TReturnFloat* input_params, TReturnFloat res_value, TReturnFloat** out_values)
    {
        TReturnFloat* tmp_params = TGE_TYPED_ALLOCA(TReturnFloat, ParameterCount);
        std::copy_n(input_params, ParameterCount, tmp_params);
        TReturnFloat* result = *out_values;
        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
            TReturnFloat prev_param_value = tmp_params[param_idx];
            TReturnFloat fd_step = (std::fabs(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;
            tmp_params[param_idx] = prev_param_value + fd_step;
            TFunctionFloat next_residual = -Function(worker_id, idx, tmp_params); // ignore constant
            tmp_params[param_idx] = prev_param_value - fd_step;
            TFunctionFloat prev_residual =- Function(worker_id, idx, tmp_params); // ignore constant
            result[param_idx] = (next_residual - prev_residual)/(2.0f*fd_step);
            tmp_params[param_idx] = prev_param_value;
        }
    }
};

template<class TFunctionFloat, class TReturnFloat, class TFunction>
struct ParallelFDComputeResidualJacobianNextPoint
{
    TFunctionFloat* OutputValues;
    uint32_t		ParameterCount;
    TReturnFloat    FiniteDifferenceStep;
    TFunction		Function;
        
    void operator()(uint32_t worker_id, uint32_t idx, const TReturnFloat* input_params, TReturnFloat res_value, TReturnFloat** out_values)
    {
        TReturnFloat* tmp_params = TGE_TYPED_ALLOCA(TReturnFloat, ParameterCount);
        std::copy_n(input_params, ParameterCount, tmp_params);
        TReturnFloat start_residual = res_value;
        TReturnFloat* result = *out_values;
        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
            TReturnFloat prev_param_value = tmp_params[param_idx];
            TReturnFloat fd_step = (std::fabs(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;
            tmp_params[param_idx] += fd_step;
            TFunctionFloat next_residual = OutputValues[idx] - Function(worker_id, idx, tmp_params);
            result[param_idx] = (next_residual - start_residual)/fd_step;
            tmp_params[param_idx] = prev_param_value;
        }
    }
};

#define ParallelFDComputeResidualJacobian ParallelFDComputeResidualJacobianCenterPoint

template<class TFunctionFloat, class TReturnFloat = TFunctionFloat, class TFunction>
ParallelFDComputeResidualJacobian<TFunctionFloat, TReturnFloat, TFunction> CreateParallelFDComputeResidualJacobian(TFunctionFloat* output_values, uint32_t parameter_count, TReturnFloat finite_diff_step, TFunction& func)
{
    return ParallelFDComputeResidualJacobian<TFunctionFloat, TReturnFloat, TFunction>{ output_values, parameter_count, finite_diff_step, func };
}

template<class TFunctionFloat, class TReturnFloat, class TFunction>
struct FDComputeResidualJacobianCenterPoint
{
    TFunctionFloat* OutputValues;
    uint32_t		ParameterCount;
    TReturnFloat	FiniteDifferenceStep;
    TFunction		Function;
        
    inline void operator()(uint32_t idx, const TReturnFloat* input_params, TReturnFloat res_value, TReturnFloat** out_values)
    {
        TReturnFloat* tmp_params = TGE_TYPED_ALLOCA(TReturnFloat, ParameterCount);
        std::copy_n(input_params, ParameterCount, tmp_params);
        TReturnFloat* result = *out_values;
        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
            TReturnFloat prev_param_value = tmp_params[param_idx];
            TReturnFloat fd_step = (fabsf(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;
            tmp_params[param_idx] = prev_param_value + fd_step;
            TFunctionFloat next_residual = -Function(idx, tmp_params); // ignore constant
            tmp_params[param_idx] = prev_param_value - fd_step;
            TFunctionFloat prev_residual =- Function(idx, tmp_params); // ignore constant
            result[param_idx] = (next_residual - prev_residual)/(2.0f*fd_step);
            tmp_params[param_idx] = prev_param_value;
        }
    }
};

template<class TFunctionFloat, class TReturnFloat, class TFunction>
struct FDComputeResidualJacobianNextPoint
{
    TFunctionFloat* OutputValues;
    uint32_t		ParameterCount;
    TReturnFloat    FiniteDifferenceStep;
    TFunction		Function;
    
    inline void operator()(uint32_t idx, const TReturnFloat* input_params, TReturnFloat res_value, TReturnFloat** out_values)
    {
        TReturnFloat* tmp_params = TGE_TYPED_ALLOCA(TReturnFloat, ParameterCount);
        std::copy_n(input_params, ParameterCount, tmp_params);
        TReturnFloat start_residual = res_value;
        TReturnFloat* result = *out_values;
        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
            TReturnFloat prev_param_value = tmp_params[param_idx];
            TReturnFloat fd_step = (fabsf(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;
            tmp_params[param_idx] += fd_step;
            TFunctionFloat next_residual = OutputValues[idx] - Function(idx, tmp_params);
            result[param_idx] = (next_residual - start_residual)/fd_step;
            tmp_params[param_idx] = prev_param_value;
        }
    }
};

#define FDComputeResidualJacobian FDComputeResidualJacobianCenterPoint

template<class TFunctionFloat, class TReturnFloat = TFunctionFloat, class TFunction>
FDComputeResidualJacobian<TFunctionFloat, TReturnFloat, TFunction> CreateFDComputeResidualJacobian(TFunctionFloat* output_values, uint32_t parameter_count, TReturnFloat finite_diff_step, TFunction& func)
{
    return FDComputeResidualJacobian<TFunctionFloat, TReturnFloat, TFunction>{ output_values, parameter_count, finite_diff_step, func };
}
template<class TFloat, class TFunction, class TMaxStep>
void ParallelFDGaussNewtonCurveFit(uint32_t id, ThreadPool& pool,
                                   TFloat mean_sq_error, uint32_t max_steps,
                                   TFloat* output_values,
                                   uint32_t value_pair_count,
                                   const TFloat* input_parameters,
                                   uint32_t parameter_count,
                                   TFunction func, TFloat finite_diff_step,
                                   TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                                   TFloat** opt_parameters)
{
    auto approx_jacobian = CreateParallelFDComputeResidualJacobian(output_values, parameter_count, finite_diff_step, func);

    auto eval = CreateParallelResidualEvaluator(id, pool, output_values, value_pair_count, func, approx_jacobian);
    
    GaussNewtonCurveFit(eval,
                        mean_sq_error, max_steps,
                        input_parameters,
                        parameter_count,
                        max_step,
                        opt_parameters);
}

template<class TFloat, class TFunction, class TMaxStep>
void ParallelFDGradientDescentCurveFit(uint32_t id, ThreadPool& pool,
                                       TFloat mean_sq_error, uint32_t max_steps,
                                       TFloat* output_values,
                                       uint32_t value_pair_count,
                                       const TFloat* input_parameters,
                                       uint32_t parameter_count,
                                       TFunction func, TFloat finite_diff_step,
                                       TMaxStep max_step, // = ConstantMinimumStep(0.75f)
                                       TFloat** opt_parameters)
{
    auto approx_jacobian = CreateParallelFDComputeResidualJacobian(output_values, parameter_count, finite_diff_step, func);

    GradientDescentCurveFit(CreateParallelResidualEvaluator(id, pool, output_values, value_pair_count, func, approx_jacobian),
                            mean_sq_error, max_steps,
                            input_parameters,
                            parameter_count,
                            max_step,
                            opt_parameters);
}

template<class TFloat, class TFunction, class TAcceptStep>
void ParallelFDLevenbergMarquardtCurveFit(uint32_t id, ThreadPool& pool,
                                          TFloat mean_sq_error, uint32_t max_steps,
                                          TFloat* output_values,
                                          uint32_t value_pair_count,
                                          const TFloat* input_parameters,
                                          uint32_t parameter_count,
                                          TFunction func, TFloat finite_diff_step,
                                          TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
                                          TFloat** opt_parameters)
{
    auto approx_jacobian = CreateParallelFDComputeResidualJacobian(output_values, parameter_count, finite_diff_step, func);

    auto eval = CreateParallelResidualEvaluator(id, pool, output_values, value_pair_count, func, approx_jacobian);
    
    LevenbergMarquardtCurveFit(eval,
                               mean_sq_error, max_steps,
                               input_parameters,
                               parameter_count,
                               accept_step,
                               opt_parameters);
}
}

#endif // _TEMPEST_FITTING_HH_

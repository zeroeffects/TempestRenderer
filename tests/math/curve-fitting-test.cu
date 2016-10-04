#define _SCL_SECURE_NO_WARNINGS

#include "tempest/utils/testing.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/fitting.hh"
#include "tempest/compute/fitting-cuda.hh"

#include <limits>
#include <memory>
#include <algorithm>

#include <cuda_runtime_api.h>

const float CurveConst[] = { 0.1f, -0.1f, 3.0f, 10.0f };
const float XStepSize = 0.1f;
const float XLimit = 14.0f;
const uint32_t XStepCount = (uint32_t)(XLimit/XStepSize);

inline EXPORT_CUDA float SumExponential(float value, const float* input_params)
{
    return input_params[2]*expf(input_params[0]*value) + input_params[3]*expf(input_params[1]*value);
}

inline EXPORT_CUDA void SumExponentialResidualJacobian(float value, const float* input_params, float** out_values)
{
    auto* out = *out_values;
    out[0] = -input_params[2]*value*expf(input_params[0]*value);
    out[1] = -input_params[3]*value*expf(input_params[1]*value);
    out[2] = -expf(input_params[0]*value);
    out[3] = -expf(input_params[1]*value);
}

struct SumExponentialFunctor
{
    float* m_InputData;
    inline EXPORT_CUDA float operator()(uint32_t, uint32_t idx, const float* input_params) { return SumExponential(m_InputData[idx], input_params); }
};

struct SumExponentialJacobianFunctor
{
    float* m_InputData;
    inline EXPORT_CUDA void operator()(float*, uint32_t, uint32_t idx, const float* input_params, float, float** out_values) { return SumExponentialResidualJacobian(m_InputData[idx], input_params, out_values); }
};

TGE_TEST("Testing generic curve fitting functions")
{
    std::unique_ptr<float[]> input_data(new float[XStepCount]);
    std::unique_ptr<float[]> output_data(new float[XStepCount]);
    uint32_t count = 0;
    auto* curve_const_orig = CurveConst;
    std::generate(input_data.get(), input_data.get() + XStepCount, [&count]() { return XStepSize*(count++); });
    for(uint32_t sample_idx = 0; sample_idx < XStepCount; ++sample_idx)
    {
        output_data[sample_idx] = SumExponential(input_data[sample_idx], curve_const_orig);
    }
    Tempest::ThreadPool pool;
    auto thread_id = pool.allocateThreadNumber();

    float opt_parameters[TGE_FIXED_ARRAY_SIZE(CurveConst)];
    float* out_opt_params = opt_parameters;

    float skewed_params[TGE_FIXED_ARRAY_SIZE(CurveConst)];
    for(uint32_t param_idx = 0; param_idx < TGE_FIXED_ARRAY_SIZE(skewed_params); ++param_idx)
    {
        skewed_params[param_idx] = CurveConst[param_idx] + (param_idx & 1) * 0.05f;
    }

    auto* input_ptr = input_data.get();

    auto gpu_input_data = CREATE_SCOPED(float*, ::cudaFree);
    size_t data_size = XStepCount*sizeof(gpu_input_data[0]);
    auto status = cudaMalloc(reinterpret_cast<void**>(&gpu_input_data), data_size);
    TGE_CHECK(status == cudaSuccess, "Failed to allocate memory");
    status = cudaMemcpy(gpu_input_data.get(), input_ptr, data_size, cudaMemcpyHostToDevice);
    TGE_CHECK(status == cudaSuccess, "Failed to copy data");

    SumExponentialFunctor sum_exponential_func_cuda{ gpu_input_data.get() };
    SumExponentialJacobianFunctor sum_exponential_jacobian_cuda{ gpu_input_data.get() };

    auto sum_exponential_func = [input_ptr](uint32_t idx, const float* input_params) { return SumExponential(input_ptr[idx], input_params); };
    auto sum_exponential_jacobian = [input_ptr](uint32_t idx, const float* input_params, float, float** out_values) { SumExponentialResidualJacobian(input_ptr[idx], input_params, out_values); };

    auto sum_exponential_func_parallel = [input_ptr](uint32_t worker_id, uint32_t idx, const float* input_params) { return SumExponential(input_ptr[idx], input_params); };
    auto sum_exponential_jacobian_parallel = [input_ptr](uint32_t worker_id, uint32_t idx, const float* input_params, float, float** out_values) { SumExponentialResidualJacobian(input_ptr[idx], input_params, out_values); };

    auto evaluator = Tempest::CreateResidualEvaluator(output_data.get(), XStepCount, sum_exponential_func, sum_exponential_jacobian);
    auto parallel_evaluator = Tempest::CreateParallelResidualEvaluator(thread_id, pool, output_data.get(), XStepCount, sum_exponential_func_parallel, sum_exponential_jacobian_parallel);
#ifndef LINUX
    auto cuda_evaluator = Tempest::CreateCUDAResidualEvaluator(output_data.get(), XStepCount, TGE_FIXED_ARRAY_SIZE(CurveConst), sum_exponential_func_cuda, sum_exponential_jacobian_cuda); 
#endif
    std::unique_ptr<float[]> residual_vec_single(new float[XStepCount]),
                             residual_vec_parallel(new float[XStepCount]),
                             residual_vec_cuda(new float[XStepCount]),
                             jacobian_single(new float[XStepCount*TGE_FIXED_ARRAY_SIZE(CurveConst)]),
                             jacobian_parallel(new float[XStepCount*TGE_FIXED_ARRAY_SIZE(CurveConst)]),
                             jacobian_cuda(new float[XStepCount*TGE_FIXED_ARRAY_SIZE(CurveConst)]);

    float* residual_vec_single_ptr = residual_vec_single.get(),
         * residual_vec_parallel_ptr = residual_vec_parallel.get(),
         * residual_vec_cuda_ptr = residual_vec_cuda.get(),
         * jacobian_single_ptr = jacobian_single.get(),
         * jacobian_parallel_ptr = jacobian_parallel.get(),
         * jacobian_cuda_ptr = jacobian_cuda.get();

    float residual_sq_sum_single,
          residual_sq_sum_parallel,
          residual_sq_sum_cuda;

    evaluator.computeResidual(skewed_params, &residual_sq_sum_single, &residual_vec_single_ptr);
    parallel_evaluator.computeResidual(skewed_params, &residual_sq_sum_parallel, &residual_vec_parallel_ptr);
#ifndef LINUX
    cuda_evaluator.computeResidual(skewed_params, &residual_sq_sum_cuda, &residual_vec_cuda_ptr);
#endif
    evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_single_ptr, &jacobian_single_ptr);
    parallel_evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_parallel_ptr, &jacobian_parallel_ptr);
#ifndef LINUX
    cuda_evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_cuda_ptr, &jacobian_cuda_ptr);
#endif

    TGE_CHECK(Tempest::ApproxEqual(residual_sq_sum_single, residual_sq_sum_parallel, 1e-3f), "Broken parallel residual evaluation function");
    TGE_CHECK(Tempest::ApproxEqual(residual_sq_sum_parallel, residual_sq_sum_cuda, 1e-3f), "Broken CUDA residual evaluation function");

    for(uint32_t sample_idx = 0; sample_idx < XStepCount; ++sample_idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(residual_vec_single_ptr[sample_idx], residual_vec_parallel_ptr[sample_idx], 1e-4f), "Broken parallel residual evaluation function");
        TGE_CHECK(Tempest::ApproxEqual(residual_vec_parallel_ptr[sample_idx], residual_vec_cuda_ptr[sample_idx], 1e-4f), "Broken CUDA residual evaluation function");

        for(uint32_t parameter_idx = 0; parameter_idx < TGE_FIXED_ARRAY_SIZE(skewed_params); ++parameter_idx)
        {
            uint32_t idx = sample_idx*TGE_FIXED_ARRAY_SIZE(skewed_params) + parameter_idx;
            TGE_CHECK(Tempest::ApproxEqual(jacobian_single_ptr[idx], jacobian_parallel_ptr[idx], 1e-4f), "Broken parallel jacobian evaluation function");
            TGE_CHECK(Tempest::ApproxEqual(jacobian_single_ptr[idx], jacobian_cuda_ptr[idx], 1e-4f), "Broken CUDA jacobian evaluation function");
        }
    }

    auto approx_jacobian = Tempest::CreateFDComputeResidualJacobian(input_data.get(), TGE_FIXED_ARRAY_SIZE(skewed_params), 1e-3f, sum_exponential_func);
    auto approx_jacobian_parallel = Tempest::CreateParallelFDComputeResidualJacobian(input_data.get(), TGE_FIXED_ARRAY_SIZE(skewed_params), 1e-3f, sum_exponential_func_parallel);
    auto approx_jacobian_cuda = Tempest::CreateFDComputeResidualJacobianCuda<float, float>(input_data.get(), TGE_FIXED_ARRAY_SIZE(skewed_params), 1e-3f, sum_exponential_func_cuda);

	auto fd_evaluator = Tempest::CreateResidualEvaluator(output_data.get(), XStepCount, sum_exponential_func, approx_jacobian);
	auto fd_parallel_evaluator = Tempest::CreateParallelResidualEvaluator(thread_id, pool, output_data.get(), XStepCount, sum_exponential_func_parallel, approx_jacobian_parallel);
#ifndef LINUX
	auto fd_cuda_evaluator = Tempest::CreateCUDAResidualEvaluator(output_data.get(), XStepCount, TGE_FIXED_ARRAY_SIZE(CurveConst), sum_exponential_func_cuda, approx_jacobian_cuda); 
#endif

    fd_evaluator.computeResidual(skewed_params, &residual_sq_sum_single, &residual_vec_single_ptr);
    fd_parallel_evaluator.computeResidual(skewed_params, &residual_sq_sum_parallel, &residual_vec_parallel_ptr);
#ifndef LINUX
    fd_cuda_evaluator.computeResidual(skewed_params, &residual_sq_sum_cuda, &residual_vec_cuda_ptr);
#endif

    fd_evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_single_ptr, &jacobian_single_ptr);
    fd_parallel_evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_parallel_ptr, &jacobian_parallel_ptr);
#ifndef LINUX
    fd_cuda_evaluator.computeJacobianMatrix(skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params), residual_vec_cuda_ptr, &jacobian_cuda_ptr);
#endif

    for(uint32_t sample_idx = 0; sample_idx < XStepCount; ++sample_idx)
    {
        TGE_CHECK(Tempest::ApproxEqual(residual_vec_single_ptr[sample_idx], residual_vec_parallel_ptr[sample_idx], 1e-4f), "Broken parallel residual evaluation function");
        TGE_CHECK(Tempest::ApproxEqual(residual_vec_parallel_ptr[sample_idx], residual_vec_cuda_ptr[sample_idx], 1e-4f), "Broken CUDA residual evaluation function");

        for(uint32_t parameter_idx = 0; parameter_idx < TGE_FIXED_ARRAY_SIZE(skewed_params); ++parameter_idx)
        {
            uint32_t idx = sample_idx*TGE_FIXED_ARRAY_SIZE(skewed_params) + parameter_idx;
            TGE_CHECK(Tempest::ApproxEqual(jacobian_single_ptr[idx], jacobian_parallel_ptr[idx], 1e-4f), "Broken parallel jacobian evaluation function");
            TGE_CHECK(Tempest::ApproxEqual(jacobian_single_ptr[idx], jacobian_cuda_ptr[idx], 5e-2f), "Broken CUDA jacobian evaluation function");
        }
    }

    Tempest::GaussNewtonCurveFit(evaluator, 1e-3f, 1024, CurveConst, TGE_FIXED_ARRAY_SIZE(CurveConst),
                                 Tempest::ConstantStep<float>(0.75f),
                                 &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params), "Curve fitting of exact curve should not result in change");
       
    Tempest::GaussNewtonCurveFit(parallel_evaluator, 1e-3f, 1024, CurveConst, TGE_FIXED_ARRAY_SIZE(CurveConst),
                                 Tempest::ConstantStep<float>(0.75f),
                                 &out_opt_params);



    Tempest::GaussNewtonCurveFit(evaluator, 1e-7f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                 Tempest::ConstantStep<float>(0.75f),
                                 &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 1e-3f); }), "Broken curve fitting algorithm");

    Tempest::GaussNewtonCurveFit(parallel_evaluator, 1e-7f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                 Tempest::ConstantStep<float>(0.75f),
                                 &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 1e-3f); }), "Broken curve fitting algorithm");

    Tempest::GradientDescentCurveFit(parallel_evaluator, 1e-12f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                     Tempest::ConstantStep<float>(0.75f),
                                     &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 5e-2f); }), "Broken curve fitting algorithm");

    Tempest::LevenbergMarquardtCurveFit(parallel_evaluator, 1e-12f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                        Tempest::ConstantStep<float>(0.75f),
                                        &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 5e-2f); }), "Broken curve fitting algorithm");

#ifndef LINUX
    Tempest::LevenbergMarquardtCurveFit(cuda_evaluator, 1e-12f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                        Tempest::ConstantStep<float>(0.75f),
                                        &out_opt_params);
#endif

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 5e-2f); }), "Broken curve fitting algorithm");

#ifndef LINUX
	Tempest::LevenbergMarquardtCurveFitCuda(cuda_evaluator, 1e-12f, 1024, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
											Tempest::ConstantStep<float>(0.75f),
											&out_opt_params);
#endif

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 5e-2f); }), "Broken curve fitting algorithm");

    Tempest::ParallelFDGaussNewtonCurveFit(thread_id, pool, 1e-12f, 1024, output_data.get(), XStepCount, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                           [input_ptr](uint32_t worker_id, uint32_t idx, const float* input_params) { return SumExponential(input_ptr[idx], input_params); },
                                           1e-5f, 
                                           Tempest::ConstantStep<float>(0.75f),
                                           &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 1e-4f); }), "Broken curve fitting algorithm");

    Tempest::ParallelFDLevenbergMarquardtCurveFit(thread_id, pool, 1e-12f, 1024, output_data.get(), XStepCount, skewed_params, TGE_FIXED_ARRAY_SIZE(skewed_params),
                                                  [input_ptr](uint32_t worker_id, uint32_t idx, const float* input_params) { return SumExponential(input_ptr[idx], input_params); },
                                                  1e-5f, 
                                                  Tempest::ConstantStep<float>(0.75f),
                                                  &out_opt_params);

    TGE_CHECK(std::equal(CurveConst, CurveConst + TGE_FIXED_ARRAY_SIZE(CurveConst), out_opt_params, [](float lhs, float rhs) { return Tempest::ApproxEqual(lhs, rhs, 1e-4f); }), "Broken curve fitting algorithm");
}
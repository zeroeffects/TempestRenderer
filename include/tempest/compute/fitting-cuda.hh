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

#ifndef _TEMPEST_FITTING_CUDA_HH_
#define _TEMPEST_FITTING_CUDA_HH_

#define CUDA_NO_HALF

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "tempest/math/matrix-variadic.hh"
#include "tempest/math/fitting.hh"

namespace Tempest
{
#define WARP_SIZE            32

template<class TFloat>
struct EvalResidualParameters
{
    TFloat*  OutputValues;
    uint32_t SampleCount;
    TFloat*  Parameters;
    TFloat*  ResidualVector;
    TFloat*  PartialResidualSquaredSum;
};

template<class TDataFloat, class TReturnFloat, class TFunction>
__global__ void EvaluateResidualMatrixElementCuda(TFunction func, TDataFloat*  output_values, uint32_t sample_count,
												  TReturnFloat* parameters, TReturnFloat* residual_vector, TReturnFloat* partial_residual_sq_sum)
{
    unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int worker_id = blockIdx.x;

    if(sample_idx >= sample_count)
        return;

    TDataFloat output_value = output_values[sample_idx];
    TDataFloat func_value = func(threadIdx.x, sample_idx, parameters);
    TDataFloat residual = output_value - func_value;

	residual_vector[sample_idx] = residual;

    TReturnFloat sum = residual*residual;
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    {
        sum += __shfl_down(sum, offset);
    }

    if(threadIdx.x == 0) partial_residual_sq_sum[worker_id] = sum;
}

template<class TFloat>
__global__ void ReduceResidualSquaredSumCuda(TFloat* partial_residual_sq_sum, size_t reduce_size)
{
	unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    TFloat sum = sample_idx < reduce_size ? partial_residual_sq_sum[sample_idx] : 0;
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    {
        sum += __shfl_down(sum, offset);
    }

    if(threadIdx.x == 0) partial_residual_sq_sum[blockIdx.x] = sum;
}

template<class TFloat, class TJacobian>
__global__ void EvaluateResidualJacobianMatrixElementCuda(TJacobian jacob, TFloat* parameters, uint32_t parameter_count,
														  TFloat* residual_vector, uint32_t sample_count, TFloat* jacobian_matrix_transpose)
{
	extern __shared__ TFloat delta_params[];

    unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(sample_idx >= sample_count)
        return;

    TFloat* res_matrix = jacobian_matrix_transpose + sample_idx*parameter_count;
	auto* delta_params_ptr = delta_params; 
    jacob(delta_params_ptr, threadIdx.x, sample_idx, parameters, residual_vector[sample_idx], &res_matrix);
}

template<class TDataFloat, class TReturnFloat, class TFunction, class TJacobian>
struct ResidualEvaluatorCuda;

template<class TDataFloat, class TReturnFloat = TDataFloat, class TFunction, class TJacobian>
void ComputeResidualCudaImpl(ResidualEvaluatorCuda<TDataFloat, TReturnFloat, TFunction, TJacobian>* evaluator, TFunction& func, TReturnFloat* parameters);

template<class TDataFloat, class TReturnFloat, class TFunction, class TJacobian>
struct ResidualEvaluatorCuda
{
    uint32_t    SampleCount;
    uint32_t    ParameterCount;
    TFunction   Function;
    TJacobian   Jacobian;

    TReturnFloat* GPUParameters;
    TDataFloat*   GPUOutputValues;
    TReturnFloat* GPUResidual;
    TReturnFloat* GPUJacobianMatrix;
    TReturnFloat* GPUPartialResidualSquaredSum;

#ifndef NDEBUG
    TReturnFloat* DbgResidual = nullptr;
#endif

    ResidualEvaluatorCuda(TDataFloat* output_values, uint32_t sample_count, uint32_t parameter_count, TFunction func, TJacobian jacob)
        :   SampleCount(sample_count),
            ParameterCount(parameter_count),
            Function(func),
            Jacobian(jacob)
    {
        size_t parameter_vec_size = ParameterCount*sizeof(GPUParameters[0]),
               value_vec_size = SampleCount*sizeof(GPUResidual[0]),
               jacobian_matrix_size = SampleCount*ParameterCount*sizeof(GPUJacobianMatrix[0]),
			   reduce_sum_count = (sample_count + WARP_SIZE - 1) / WARP_SIZE,
			   partial_sum_size = reduce_sum_count*sizeof(GPUPartialResidualSquaredSum[0]);

        auto status = cudaMalloc(reinterpret_cast<void**>(&GPUParameters), parameter_vec_size);
        TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory");
        cudaMalloc(reinterpret_cast<void**>(&GPUOutputValues), value_vec_size);
        TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory");
        cudaMalloc(reinterpret_cast<void**>(&GPUResidual), value_vec_size);
        TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory");
        cudaMalloc(reinterpret_cast<void**>(&GPUJacobianMatrix), jacobian_matrix_size);
        TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory");
        cudaMalloc(reinterpret_cast<void**>(&GPUPartialResidualSquaredSum), partial_sum_size);
        TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory");

        status = cudaMemcpy(GPUOutputValues, output_values, value_vec_size, cudaMemcpyHostToDevice);
        TGE_ASSERT(status == cudaSuccess, "Failed to copy data");
    }

    ResidualEvaluatorCuda(const ResidualEvaluatorCuda&)=delete;
    ResidualEvaluatorCuda& operator=(const ResidualEvaluatorCuda&)=delete;

    ~ResidualEvaluatorCuda()
    {
        cudaFree(GPUParameters);
        cudaFree(GPUOutputValues);
        cudaFree(GPUResidual);
        cudaFree(GPUJacobianMatrix);
        cudaFree(GPUPartialResidualSquaredSum);
    }
    
    void computeResidualGPU(TReturnFloat* parameters)
    {
        auto status = cudaMemcpy(GPUParameters, parameters, ParameterCount*sizeof(parameters[0]), cudaMemcpyHostToDevice);
        TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

        dim3 group_size(WARP_SIZE, 1, 1);
        dim3 eval_thread_groups((SampleCount + WARP_SIZE - 1)/WARP_SIZE, 1, 1);

		size_t reduce_sum_count = (SampleCount + WARP_SIZE - 1) / WARP_SIZE;
		if(SampleCount % WARP_SIZE)
		{
			cudaMemset(GPUPartialResidualSquaredSum, 0, reduce_sum_count*sizeof(GPUPartialResidualSquaredSum[0]));
		}

        EvaluateResidualMatrixElementCuda<<<eval_thread_groups, group_size>>>(Function, GPUOutputValues, SampleCount, GPUParameters, GPUResidual, GPUPartialResidualSquaredSum);

		for(uint32_t reduce_counter = static_cast<uint32_t>(reduce_sum_count); reduce_counter > 1;)
		{
			uint32_t reduced_elem_count = reduce_counter;
			reduce_counter = (reduce_counter + WARP_SIZE - 1) / WARP_SIZE;
			dim3 reduce_thread_groups(reduce_counter, 1, 1);
			ReduceResidualSquaredSumCuda<<<reduce_thread_groups, group_size>>>(GPUPartialResidualSquaredSum, reduced_elem_count);
		}        
	}

	void computeResidual(TReturnFloat* parameters, TReturnFloat* out_residual_sq_sum, TReturnFloat** out_residual_vec)
    {
		computeResidualGPU(parameters);

        auto status = cudaMemcpy(out_residual_sq_sum, GPUPartialResidualSquaredSum, sizeof(*out_residual_sq_sum), cudaMemcpyDeviceToHost);
        TGE_ASSERT(status == cudaSuccess, "Failed to copy data");
        status = cudaMemcpy(*out_residual_vec, GPUResidual, SampleCount*sizeof(GPUResidual[0]), cudaMemcpyDeviceToHost);
        TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

    #ifndef NDEBUG
        DbgResidual = *out_residual_vec;
    #endif
    }

    void computeJacobianMatrixGPU()
    {
        dim3 group_size(WARP_SIZE, 1, 1);
        dim3 eval_thread_groups((SampleCount + WARP_SIZE - 1)/WARP_SIZE, 1, 1);
        EvaluateResidualJacobianMatrixElementCuda<<<eval_thread_groups, group_size, ParameterCount*sizeof(TReturnFloat)>>>(Jacobian, GPUParameters, ParameterCount, GPUResidual, SampleCount, GPUJacobianMatrix);
	}

	void computeJacobianMatrix(TReturnFloat* parameters, uint32_t parameter_count, TReturnFloat* residual_vec, TReturnFloat** out_jacob_transpose_matrix)
    {
    #ifndef NDEBUG
        TGE_ASSERT(DbgResidual == residual_vec, "Residual is automatically reused. Don't change it!");
    #endif

		computeJacobianMatrixGPU();

		auto status = cudaMemcpy(*out_jacob_transpose_matrix, GPUJacobianMatrix, ParameterCount*SampleCount*sizeof(GPUJacobianMatrix[0]), cudaMemcpyDeviceToHost);
        TGE_ASSERT(status == cudaSuccess, "Failed to copy data");
    }
};

template<class TDataFloat, class TReturnFloat = TDataFloat, class TFunction, class TJacobian>
ResidualEvaluatorCuda<TDataFloat, TReturnFloat, TFunction, TJacobian> CreateCUDAResidualEvaluator(TDataFloat* output_values, uint32_t sample_count, uint32_t parameter_count, TFunction func, TJacobian jacob)
{
    return ResidualEvaluatorCuda<TDataFloat, TReturnFloat, TFunction, TJacobian>(output_values, sample_count, parameter_count, func, jacob);
}

#define FDComputeResidualJacobianCuda FDComputeResidualJacobianCenterPointCuda

// TODO: Try shared memory schemas
template<class TDataFloat, class TReturn, class TFunction>
struct FDComputeResidualJacobianCenterPointCuda
{
    TDataFloat*    OutputValues;
    uint32_t	   ParameterCount;
    TReturn        FiniteDifferenceStep;
    TFunction	   Function;
 
    void __device__ operator()(TReturn* parameter_matrix_slice, uint32_t worker_id, uint32_t sample_idx, const TReturn* input_params, TReturn res_value, TReturn** out_values)
    {
		TReturn* result = *out_values;

		for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
		{
			 parameter_matrix_slice[param_idx] = input_params[param_idx];
		}

		__syncthreads();

        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
			TReturn prev_param_value = parameter_matrix_slice[param_idx];

            TReturn fd_step = (fabs(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;
            
			parameter_matrix_slice[param_idx] = prev_param_value + fd_step; 

			__syncthreads();

            TDataFloat next_residual = -Function(worker_id, sample_idx, parameter_matrix_slice); // ignore constant
			
			__syncthreads();
			
			parameter_matrix_slice[param_idx] = prev_param_value - fd_step; 

			__syncthreads();

            TDataFloat prev_residual = -Function(worker_id, sample_idx, parameter_matrix_slice); // ignore constant
            result[param_idx] = static_cast<TReturn>((next_residual - prev_residual)/(2.0f*fd_step));
            
			__syncthreads();
			
			parameter_matrix_slice[param_idx] = prev_param_value;
		}
    }
};

template<class TDataFloat, class TReturn, class TFunction>
struct FDComputeResidualJacobianNextPointCuda
{
	TReturn*	   ParameterMatrix;
	TDataFloat*    OutputValues;
    uint32_t	   ParameterCount;
    TReturn        FiniteDifferenceStep;
    TFunction	   Function;

    void __device__ operator()(uint32_t worker_id, uint32_t sample_idx, const TDataFloat* input_params, TDataFloat res_value, TReturn** out_values)
    {
		TReturn start_residual = res_value;
        TReturn* result = *out_values;

		TReturn* parameter_matrix_slice = ParameterMatrix + sample_idx*ParameterCount;

		for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
		{
			 parameter_matrix_slice[param_idx] = input_params[param_idx];
		}

        for(uint32_t param_idx = 0; param_idx < ParameterCount; ++param_idx)
        {
			TReturn prev_param_value = parameter_matrix_slice[param_idx];

			TReturn fd_step = (fabs(prev_param_value) + FiniteDifferenceStep)*FiniteDifferenceStep;

			parameter_matrix_slice[param_idx] = prev_param_value + fd_step; 
            TDataFloat next_residual = OutputValues[sample_idx] - Function(worker_id, sample_idx, parameter_matrix_slice);
            result[param_idx] = static_cast<TReturn>((next_residual - start_residual)/fd_step);

            parameter_matrix_slice[param_idx] = prev_param_value;
        }
    }
};

template<class TFloat>
__global__ void ReduceDampingCoefficient(TFloat initial_value, TFloat* hessian_matrix, uint32_t parameter_count, TFloat* partial_max_diag)
{
	unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    TFloat max_value = sample_idx < parameter_count ? hessian_matrix[sample_idx*parameter_count + sample_idx] : 0;
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    {
        max_value = Maxf(max_value, __shfl_down(max_value, offset));
    }

    if(threadIdx.x == 0) partial_max_diag[blockIdx.x] = Maxf(initial_value, max_value);
}

template<class TFloat>
__global__ void ReduceMaxValue(TFloat* partial_max_diag)
{
	unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	TFloat max_value = partial_max_diag[sample_idx];
	for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    {
        max_value = Maxf(max_value, __shfl_down(max_value, offset));
    }

	if(threadIdx.x == 0) partial_max_diag[blockIdx.x] = max_value;
}

template<class TFloat>
__global__ void MatrixDamping(TFloat* in_matrix, TFloat coefficient, uint32_t diagonal, uint32_t bounds, TFloat* out_matrix)
{
	unsigned int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(sample_idx >= bounds)
		return;

	TFloat elem = in_matrix[sample_idx];
	if((sample_idx % diagonal) == 0)
		elem += coefficient;

	out_matrix[sample_idx] = elem;
}

cublasStatus_t Xgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t Xgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
	return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t Xgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
	return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t Xgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
	return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}


template<class TFloat, class TResidualEvaluator, class TAcceptStep>
void LevenbergMarquardtCurveFitCudaSoft(TResidualEvaluator& eval_residual,
										TFloat mean_sq_error, uint32_t max_steps,
										const TFloat* input_parameters,
										uint32_t parameter_count,
										TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
										TFloat** opt_parameters,
										FitStatistics<TFloat>* stats = nullptr,
										LevenbergMarquardtSettings<TFloat>* in_settings = nullptr)
{
    auto value_pair_count = eval_residual.SampleCount;
    const uint32_t jacobian_sq_matrix_size = parameter_count*parameter_count;
    const uint32_t inverse_matrix_size = jacobian_sq_matrix_size;

	auto cublas_handle = CREATE_SCOPED(cublasHandle_t, ::cublasDestroy);
	auto cublas_status = cublasCreate(&cublas_handle);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to initialize cuBLAS");

	uint32_t total_data_size = 4*parameter_count + jacobian_sq_matrix_size + inverse_matrix_size;

	LevenbergMarquardtSettings<TFloat> settings;
	if(in_settings)
	{
		settings = *in_settings;
	}

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
    TFloat* param_step = data_offset;
	data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;
    auto* tmp_step = data_offset;
    data_offset += parameter_count;
    auto* tmp_diff = data_offset;
    data_offset += parameter_count;

    auto* damp_approx_hessian_matrix = data_offset;
    data_offset += jacobian_sq_matrix_size;
    auto* approx_hessian_inverse_matrix = data_offset;
    data_offset += inverse_matrix_size;

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    auto gradient = CREATE_SCOPED(TFloat*, ::cudaFree);
    auto status = cudaMallocHost(reinterpret_cast<void**>(&gradient), parameter_count*sizeof(gradient[0]));
    TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient");

    auto approx_hessian_matrix = CREATE_SCOPED(TFloat*, ::cudaFree);
    status = cudaMallocHost(reinterpret_cast<void**>(&approx_hessian_matrix), jacobian_sq_matrix_size*sizeof(approx_hessian_matrix[0]));
    TGE_ASSERT(status == cudaSuccess, "Failed to allocate hessian matrix");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

	auto approx_hessian_matrix_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&approx_hessian_matrix_cuda), jacobian_sq_matrix_size*sizeof(approx_hessian_matrix_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate Hessian matrix");

	auto gradient_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&gradient_cuda), parameter_count*sizeof(gradient_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient memory");

	auto partial_max_value = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&partial_max_value), ((parameter_count + WARP_SIZE - 1)/WARP_SIZE)*sizeof(partial_max_value[0]));

    auto copy_stream = CREATE_SCOPED(cudaStream_t, ::cudaStreamDestroy);
    status = cudaStreamCreate(&copy_stream);
    TGE_ASSERT(status == cudaSuccess, "Failed to create stream");

    eval_residual.computeResidualGPU(*opt_parameters);

    status = cudaMemcpyAsync(&last_residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(last_residual_sq_sum), cudaMemcpyDeviceToHost, copy_stream);
    TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

	eval_residual.computeJacobianMatrixGPU();
	
	const TFloat alpha = 1.0f;
	const TFloat beta = 0.0f;

	cublas_status = Xgemm(cublas_handle,
						  CUBLAS_OP_N, CUBLAS_OP_T,
						  parameter_count, parameter_count, value_pair_count,
						  &alpha,
						  eval_residual.GPUJacobianMatrix, parameter_count,
						  eval_residual.GPUJacobianMatrix, parameter_count,
						  &beta,
						  approx_hessian_matrix_cuda, parameter_count);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

    status = cudaMemcpyAsync(approx_hessian_matrix, approx_hessian_matrix_cuda, jacobian_sq_matrix_size*sizeof(approx_hessian_matrix[0]), cudaMemcpyDeviceToHost, copy_stream);
	TGE_ASSERT(status == cudaSuccess, "Failed to copy hessian");

	cublas_status = Xgemv(cublas_handle, CUBLAS_OP_N,
						  parameter_count, value_pair_count,
						  &alpha,
						  eval_residual.GPUJacobianMatrix, parameter_count,
						  eval_residual.GPUResidual, 1,
						  &beta,
						  gradient_cuda, 1);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");
    
    status = cudaMemcpyAsync(gradient, gradient_cuda, parameter_count*sizeof(gradient[0]), cudaMemcpyDeviceToHost, copy_stream);
	TGE_ASSERT(status == cudaSuccess, "Failed to copy gradient");

	TFloat damping_multiplier = settings.DampingFailureMultiplier;
    const TFloat tau = settings.InitialDampingMultiplier;
    const TFloat break_criteria = settings.BreakOnSmallStep;

    TFloat damping = settings.InitialMinDamping;

	uint32_t thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;

	dim3 rd_thread_groups(thread_group_count, 1, 1);
	dim3 reduce_group_size(WARP_SIZE, 1, 1);
	ReduceDampingCoefficient<<<rd_thread_groups, reduce_group_size>>>(damping, approx_hessian_matrix_cuda.get(), parameter_count, partial_max_value.get());
	
	while(thread_group_count > 1)
	{
		thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;
		dim3 reduce_thread_groups(thread_group_count, 1, 1);
		ReduceMaxValue<<<reduce_thread_groups, reduce_group_size>>>(partial_max_value.get());
	}
	
	status = cudaMemcpyAsync(&damping, partial_max_value, sizeof(damping), cudaMemcpyDeviceToHost, copy_stream);
	TGE_ASSERT(status == cudaSuccess, "Failed to transfer damping coefficient");

    status = cudaStreamSynchronize(copy_stream);
    TGE_ASSERT(status == cudaSuccess, "Failed to synchronize");

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

        Tempest::MatrixInverse(damp_approx_hessian_matrix, parameter_count, &approx_hessian_inverse_matrix);
        if(!std::isfinite(approx_hessian_inverse_matrix[0]))
        {
            damping *= damping_multiplier;
            damping_multiplier *= damping_multiplier;
			continue;
        }

        Tempest::MatrixTransformVector(approx_hessian_inverse_matrix, parameter_count, parameter_count, gradient.get(), parameter_count, &param_step);

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

		eval_residual.computeResidualGPU(tmp_parameters);
		
        Tempest::VectorMultiply(param_step, parameter_count, damping, &tmp_step);
        Tempest::VectorAdd(tmp_step, parameter_count, gradient.get(), parameter_count, &tmp_diff);

        TFloat predicted_gain = 0.5f*Tempest::VectorDot(param_step, parameter_count, tmp_diff, parameter_count);

        status = cudaMemcpy(&residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(residual_sq_sum), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

        TFloat gain_ratio = 0.5f*(last_residual_sq_sum - residual_sq_sum)/predicted_gain;

        if(gain_ratio > 0.0f)
        {
			++effective_iter;

            std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
            last_residual_sq_sum = residual_sq_sum;
            if(last_residual_sq_sum/value_pair_count <= mean_sq_error)
                break;

            eval_residual.computeJacobianMatrixGPU();

			cublas_status = Xgemm(cublas_handle,
								  CUBLAS_OP_N, CUBLAS_OP_T,
								  parameter_count, parameter_count, value_pair_count,
								  &alpha,
								  eval_residual.GPUJacobianMatrix, parameter_count,
								  eval_residual.GPUJacobianMatrix, parameter_count,
								  &beta,
								  approx_hessian_matrix_cuda, parameter_count);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

            status = cudaMemcpyAsync(approx_hessian_matrix, approx_hessian_matrix_cuda, jacobian_sq_matrix_size*sizeof(approx_hessian_matrix[0]), cudaMemcpyDeviceToHost, copy_stream);
			TGE_ASSERT(status == cudaSuccess, "Failed to copy hessian");

			cublas_status = Xgemv(cublas_handle, CUBLAS_OP_N,
								  parameter_count, value_pair_count,
								  &alpha,
								  eval_residual.GPUJacobianMatrix, parameter_count,
								  eval_residual.GPUResidual, 1,
								  &beta,
								  gradient_cuda, 1);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");

			status = cudaMemcpyAsync(gradient, gradient_cuda, parameter_count*sizeof(gradient[0]), cudaMemcpyDeviceToHost, copy_stream);
			TGE_ASSERT(status == cudaSuccess, "Failed to copy gradient");

            status = cudaStreamSynchronize(copy_stream);
            TGE_ASSERT(status == cudaSuccess, "Failed to synchronize");

			TFloat gradient_length = Tempest::VectorLengthSquared(gradient.get(), parameter_count);
            if(gradient_length <= break_criteria*break_criteria)
                break;

			TFloat min_multiplier = 1.0f/3.0f;
            TFloat multiplier = std::fmax(min_multiplier, 1.0f - std::pow((2.0f*gain_ratio - 1.0f), 3.0f));
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


template<class TFloat, class TResidualEvaluator, class TAcceptStep>
void LevenbergMarquardtCurveFitCuda(TResidualEvaluator& eval_residual,
									TFloat mean_sq_error, uint32_t max_steps,
									const TFloat* input_parameters,
									uint32_t parameter_count,
									TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
									TFloat** opt_parameters,
									FitStatistics<TFloat>* stats = nullptr)
{
    auto value_pair_count = eval_residual.SampleCount;
    const uint32_t jacobian_sq_matrix_size = parameter_count*parameter_count;

	uint32_t total_data_size = 5*parameter_count;

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
    TFloat* param_step = data_offset;
	data_offset += parameter_count;
    auto* gradient = data_offset;
    data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;
    auto* tmp_step = data_offset;
    data_offset += parameter_count;
    auto* tmp_diff = data_offset;
    data_offset += parameter_count;

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

	auto cusolver_handle = CREATE_SCOPED(cusolverDnHandle_t, ::cusolverDnDestroy);
	auto cusolver_status = cusolverDnCreate(&cusolver_handle);
	TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to initialize cuSOLVER");

	auto cublas_handle = CREATE_SCOPED(cublasHandle_t, ::cublasDestroy);
	auto cublas_status = cublasCreate(&cublas_handle);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to initialize cuBLAS");

	auto copy_stream = CREATE_SCOPED(cudaStream_t, ::cudaStreamDestroy);
	auto status = cudaStreamCreate(&copy_stream);
	bool desync = true;

    eval_residual.computeResidualGPU(*opt_parameters);

	status = cudaMemcpyAsync(&last_residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(last_residual_sq_sum), cudaMemcpyDeviceToHost, copy_stream);
    TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

	eval_residual.computeJacobianMatrixGPU();

	auto approx_hessian_matrix_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&approx_hessian_matrix_cuda), jacobian_sq_matrix_size*sizeof(approx_hessian_matrix_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate Hessian matrix");

	auto damp_approx_hessian_matrix_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&damp_approx_hessian_matrix_cuda), jacobian_sq_matrix_size*sizeof(damp_approx_hessian_matrix_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate Hessian matrix");

	auto gradient_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&gradient_cuda), parameter_count*sizeof(gradient_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient memory");

	auto param_step_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&param_step_cuda), parameter_count*sizeof(param_step_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient memory");

	auto partial_max_value = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&partial_max_value), ((parameter_count + WARP_SIZE - 1)/WARP_SIZE)*sizeof(partial_max_value[0]));

	auto dev_info = CREATE_SCOPED(int*, ::cudaFree);
	status = cudaMallocHost(reinterpret_cast<void**>(&dev_info), sizeof(int));

	

	int cholesky_work_size = 0;
	auto cholesky_workspace_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);

	const TFloat alpha = 1.0f;
	const TFloat beta = 0.0f;

	cublas_status = cublasSgemm(cublas_handle,
								CUBLAS_OP_N, CUBLAS_OP_T,
								parameter_count, parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUJacobianMatrix, parameter_count,
								&beta,
								approx_hessian_matrix_cuda, parameter_count);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

	cublas_status = cublasSgemv(cublas_handle, CUBLAS_OP_N,
								parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUResidual, 1,
								&beta,
								gradient_cuda, 1);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");
    
    TFloat damping_multiplier = 2.0f;
    const TFloat tau = 0.75f;
    const TFloat break_criteria = 1e-8f;

	uint32_t thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;

	dim3 rd_thread_groups(thread_group_count, 1, 1);
	dim3 reduce_group_size(WARP_SIZE, 1, 1);
	TFloat init_damp = 1.0f;
	ReduceDampingCoefficient<<<rd_thread_groups, reduce_group_size>>>(init_damp, approx_hessian_matrix_cuda.get(), parameter_count, partial_max_value.get());
	
	while(thread_group_count > 1)
	{
		thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;
		dim3 reduce_thread_groups(thread_group_count, 1, 1);
		ReduceMaxValue<<<reduce_thread_groups, reduce_group_size>>>(partial_max_value.get());
	}
	
	TFloat damping;
	status = cudaMemcpy(&damping, partial_max_value, sizeof(damping), cudaMemcpyDeviceToHost);
	TGE_ASSERT(status == cudaSuccess, "Failed to transfer damping coefficient");

	cudaStreamSynchronize(copy_stream);

	status = cudaMemcpyAsync(gradient, gradient_cuda, parameter_count*sizeof(gradient[0]), cudaMemcpyDeviceToHost, copy_stream);
	desync = true;

    damping *= tau;

	uint32_t iter_step;
	auto at_exit = CreateAtScopeExit([&last_residual_sq_sum, &iter_step, value_pair_count, stats]
	{
		if(stats)
		{
			stats->IterationCount = iter_step;
			stats->MeanSequaredError = last_residual_sq_sum/value_pair_count;
		}
	});

    for(iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
		const uint32_t damp_base_size = 64;
		dim3 damp_thread_group((jacobian_sq_matrix_size + damp_base_size - 1)/damp_base_size, 1, 1);
		dim3 damp_group_size(damp_base_size, 1, 1);
		uint32_t diag = parameter_count + 1;
		MatrixDamping<<<damp_thread_group, damp_group_size>>>(approx_hessian_matrix_cuda.get(), damping, diag, jacobian_sq_matrix_size, damp_approx_hessian_matrix_cuda.get());

		int work_size;
		cusolver_status = cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, &work_size);
		TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to compute workspace size for cholesky decomposition");

		if(cholesky_work_size < work_size)
		{
			cholesky_workspace_cuda.reset();
			status = cudaMalloc(reinterpret_cast<void**>(&cholesky_workspace_cuda), work_size);
			TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory for Cholesky decomposition");
			cholesky_work_size = work_size;
		}

		cusolver_status = cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
										   parameter_count,
										   damp_approx_hessian_matrix_cuda, parameter_count,
										   cholesky_workspace_cuda, work_size, dev_info);
		TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to execute Cholesky decomposition");

        if(*dev_info < 0)
        {
            damping *= damping_multiplier;
            damping_multiplier *= damping_multiplier;
			continue;
        }

		cublas_status = cublasScopy(cublas_handle, parameter_count, gradient_cuda, 1, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to initialize gradient");
		cublas_status = cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to solve linear system");

		cublas_status = cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to solve linear system");

		status = cudaMemcpy(param_step, param_step_cuda, parameter_count*sizeof(param_step_cuda[0]), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed transfer parameter step");

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
    
        eval_residual.computeResidualGPU(tmp_parameters);
		
		if(desync)
		{
			cudaStreamSynchronize(copy_stream);
			desync = false;
		}

        Tempest::VectorMultiply(param_step, parameter_count, damping, &tmp_step);
        Tempest::VectorAdd(tmp_step, parameter_count, gradient, parameter_count, &tmp_diff);

        TFloat predicted_gain = 0.5f*Tempest::VectorDot(param_step, parameter_count, tmp_diff, parameter_count);
 
		TFloat residual_sq_sum = 0.0f;
		status = cudaMemcpy(&residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(residual_sq_sum), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed to copy data");
		
		TFloat gain_ratio = 0.5f*(last_residual_sq_sum - residual_sq_sum)/predicted_gain;

        if(gain_ratio > 0.0f)
        {
            std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
            last_residual_sq_sum = residual_sq_sum;
            if(last_residual_sq_sum/value_pair_count <= mean_sq_error)
                break;

            eval_residual.computeJacobianMatrixGPU();

			cublas_status = cublasSgemm(cublas_handle,
								CUBLAS_OP_N, CUBLAS_OP_T,
								parameter_count, parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUJacobianMatrix, parameter_count,
								&beta,
								approx_hessian_matrix_cuda, parameter_count);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

			cublas_status = cublasSgemv(cublas_handle, CUBLAS_OP_N,
										parameter_count, value_pair_count,
										&alpha,
										eval_residual.GPUJacobianMatrix, parameter_count,
										eval_residual.GPUResidual, 1,
										&beta,
										gradient_cuda, 1);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");

			TFloat gradient_length;
			cublas_status = cublasSdot(cublas_handle, parameter_count, gradient_cuda, 1, gradient_cuda, 1, &gradient_length);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to apply dot product to gradient");

			status = cudaMemcpyAsync(gradient, gradient_cuda, parameter_count*sizeof(gradient[0]), cudaMemcpyDeviceToHost, copy_stream);
			desync = true;
			TGE_ASSERT(status == cudaSuccess, "Failed to copy gradient");

            if(gradient_length <= break_criteria*break_criteria)
                break;

            TFloat multiplier = Maxf(1.0f/3.0f, 1.0f - powf((2.0f*gain_ratio - 1.0f), 3.0f));
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

template<class TFloat, class TResidualEvaluator, class TAcceptStep>
void LevenbergMarquardtCurveFitCudaAggressive(TResidualEvaluator& eval_residual,
											  TFloat mean_sq_error, uint32_t max_steps,
											  const TFloat* input_parameters,
											  uint32_t parameter_count,
											  TAcceptStep accept_step, // = ConstantMinimumStep(0.75f)
											  TFloat** opt_parameters)
{
    auto value_pair_count = eval_residual.SampleCount;
    const uint32_t jacobian_sq_matrix_size = parameter_count*parameter_count;

	uint32_t total_data_size = 2*parameter_count;

    std::unique_ptr<TFloat[]> data(new TFloat[total_data_size]);

    TFloat* data_offset = data.get();
    TFloat* param_step = data_offset;
	data_offset += parameter_count;
    auto* tmp_parameters = data_offset;
    data_offset += parameter_count;

	TGE_ASSERT(data_offset - data.get() == total_data_size, "Incorrectly populated data pointers");

    std::copy_n(input_parameters, parameter_count, *opt_parameters);

    TFloat last_residual_sq_sum = 0;

	auto cusolver_handle = CREATE_SCOPED(cusolverDnHandle_t, ::cusolverDnDestroy);
	auto cusolver_status = cusolverDnCreate(&cusolver_handle);
	TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to initialize cuSOLVER");

	auto cublas_handle = CREATE_SCOPED(cublasHandle_t, ::cublasDestroy);
	auto cublas_status = cublasCreate(&cublas_handle);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to initialize cuBLAS");

    eval_residual.computeResidualGPU(*opt_parameters);

	eval_residual.computeJacobianMatrixGPU();

	auto approx_hessian_matrix_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	auto status = cudaMalloc(reinterpret_cast<void**>(&approx_hessian_matrix_cuda), jacobian_sq_matrix_size*sizeof(approx_hessian_matrix_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate Hessian matrix");

	auto damp_approx_hessian_matrix_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&damp_approx_hessian_matrix_cuda), jacobian_sq_matrix_size*sizeof(damp_approx_hessian_matrix_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate Hessian matrix");

	auto gradient_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&gradient_cuda), parameter_count*sizeof(gradient_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient memory");

	auto param_step_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&param_step_cuda), parameter_count*sizeof(param_step_cuda[0]));
	TGE_ASSERT(status == cudaSuccess, "Failed to allocate gradient memory");

	auto partial_max_value = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&partial_max_value), ((parameter_count + WARP_SIZE - 1)/WARP_SIZE)*sizeof(partial_max_value[0]));

	auto gain_vec_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&gain_vec_cuda), parameter_count*sizeof(gain_vec_cuda[0]));

	auto dev_info = CREATE_SCOPED(int*, ::cudaFree);
	status = cudaMalloc(reinterpret_cast<void**>(&dev_info), sizeof(int));

	int cholesky_work_size = 0;
	auto cholesky_workspace_cuda = CREATE_SCOPED(TFloat*, ::cudaFree);

	const TFloat alpha = 1.0f;
	const TFloat beta = 0.0f;

	cublas_status = cublasSgemm(cublas_handle,
								CUBLAS_OP_N, CUBLAS_OP_T,
								parameter_count, parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUJacobianMatrix, parameter_count,
								&beta,
								approx_hessian_matrix_cuda, parameter_count);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

	cublas_status = cublasSgemv(cublas_handle, CUBLAS_OP_N,
								parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUResidual, 1,
								&beta,
								gradient_cuda, 1);
	TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");
    
    TFloat damping_multiplier = 2.0f;
    const TFloat tau = 0.75f;
    const TFloat break_criteria = 1e-8f;

	uint32_t thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;

	dim3 rd_thread_groups(thread_group_count, 1, 1);
	dim3 reduce_group_size(WARP_SIZE, 1, 1);
	ReduceDampingCoefficient<<<rd_thread_groups, reduce_group_size>>>(1.0f, approx_hessian_matrix_cuda, parameter_count, partial_max_value);
	
	while(thread_group_count > 1)
	{
		thread_group_count = (parameter_count + WARP_SIZE - 1)/WARP_SIZE;
		dim3 reduce_thread_groups(thread_group_count, 1, 1);
		ReduceMaxValue<<<reduce_thread_groups, reduce_group_size>>>(partial_max_value);
	}
	
	status = cudaMemcpy(&last_residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(last_residual_sq_sum), cudaMemcpyDeviceToHost);
    TGE_ASSERT(status == cudaSuccess, "Failed to copy data");

	TFloat damping;
	status = cudaMemcpy(&damping, partial_max_value, sizeof(damping), cudaMemcpyDeviceToHost);
	TGE_ASSERT(status == cudaSuccess, "Failed to transfer damping coefficient");

    damping *= tau;

    for(uint32_t iter_step = 0; iter_step < max_steps && mean_sq_error < last_residual_sq_sum/value_pair_count; ++iter_step)
    {
		const uint32_t damp_base_size = 64;
		dim3 damp_thread_group((jacobian_sq_matrix_size + damp_base_size - 1)/damp_base_size, 1, 1);
		dim3 damp_group_size(damp_base_size, 1, 1);
		MatrixDamping<<<damp_thread_group, damp_group_size>>>(approx_hessian_matrix_cuda, damping, parameter_count + 1, jacobian_sq_matrix_size, damp_approx_hessian_matrix_cuda);

		int work_size;
		cusolver_status = cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, &work_size);
		TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to compute workspace size for cholesky decomposition");

		if(cholesky_work_size < work_size)
		{
			cholesky_workspace_cuda.reset();
			status = cudaMalloc(reinterpret_cast<void**>(&cholesky_workspace_cuda), work_size);
			TGE_ASSERT(status == cudaSuccess, "Failed to allocate memory for Cholesky decomposition");
			cholesky_work_size = work_size;
		}

		int cholesky_info;
		cusolver_status = cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
										   parameter_count,
										   damp_approx_hessian_matrix_cuda, parameter_count,
										   cholesky_workspace_cuda, work_size, dev_info);
		TGE_ASSERT(cusolver_status == CUSOLVER_STATUS_SUCCESS, "Failed to execute Cholesky decomposition");

		status = cudaMemcpy(&cholesky_info, dev_info, sizeof(cholesky_info), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed to copy info about Cholesky decomposition");

        if(cholesky_info < 0)
        {
            damping *= damping_multiplier;
            damping_multiplier *= damping_multiplier;
			continue;
        }

		cublas_status = cublasScopy(cublas_handle, parameter_count, gradient_cuda, 1, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to initialize gradient");
		cublas_status = cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to solve linear system");

		cublas_status = cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, parameter_count, damp_approx_hessian_matrix_cuda, parameter_count, param_step_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to solve linear system");

		status = cudaMemcpy(param_step, param_step_cuda, parameter_count*sizeof(param_step_cuda[0]), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed transfer parameter step");

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
    
        eval_residual.computeResidualGPU(tmp_parameters);

		cublas_status = cublasScopy(cublas_handle, parameter_count, gradient_cuda, 1, gain_vec_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to copy vector");

		cublas_status = cublasSaxpy(cublas_handle, parameter_count, &damping, param_step_cuda, 1, gain_vec_cuda, 1);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to add vectors");
		
		TFloat predicted_gain;
		cublas_status = cublasSdot(cublas_handle, parameter_count, param_step_cuda, 1, gain_vec_cuda, 1, &predicted_gain);
		TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to apply dot product to gradient");

		TFloat residual_sq_sum = 0.0f;
		status = cudaMemcpy(&residual_sq_sum, eval_residual.GPUPartialResidualSquaredSum, sizeof(residual_sq_sum), cudaMemcpyDeviceToHost);
		TGE_ASSERT(status == cudaSuccess, "Failed to copy data");
		
		TFloat gain_ratio = (last_residual_sq_sum - residual_sq_sum)/predicted_gain;

        if(gain_ratio > 0.0f)
        {
            std::copy_n(tmp_parameters, parameter_count, *opt_parameters);
            last_residual_sq_sum = residual_sq_sum;
            if(last_residual_sq_sum/value_pair_count <= mean_sq_error)
                break;

            eval_residual.computeJacobianMatrixGPU();

			cublas_status = cublasSgemm(cublas_handle,
								CUBLAS_OP_N, CUBLAS_OP_T,
								parameter_count, parameter_count, value_pair_count,
								&alpha,
								eval_residual.GPUJacobianMatrix, parameter_count,
								eval_residual.GPUJacobianMatrix, parameter_count,
								&beta,
								approx_hessian_matrix_cuda, parameter_count);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to execute multiplication");

			cublas_status = cublasSgemv(cublas_handle, CUBLAS_OP_N,
										parameter_count, value_pair_count,
										&alpha,
										eval_residual.GPUJacobianMatrix, parameter_count,
										eval_residual.GPUResidual, 1,
										&beta,
										gradient_cuda, 1);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to multiply vector");

			TFloat gradient_length;
			cublas_status = cublasSdot(cublas_handle, parameter_count, gradient_cuda, 1, gradient_cuda, 1, &gradient_length);
			TGE_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS, "Failed to apply dot product to gradient");

            if(gradient_length <= break_criteria*break_criteria)
                break;

            TFloat multiplier = Maxf(1.0f/3.0f, 1.0f - powf((2.0f*gain_ratio - 1.0f), 3.0f));
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


template<class TDataFloat, class TReturnFloat = TDataFloat, class TFunction>
FDComputeResidualJacobianCuda<TDataFloat, TReturnFloat, TFunction> CreateFDComputeResidualJacobianCuda(TDataFloat* output_values, uint32_t parameter_count, TReturnFloat finite_diff_step, TFunction& func)
{
    return FDComputeResidualJacobianCuda<TDataFloat, TReturnFloat, TFunction>{ output_values, parameter_count, finite_diff_step, func };
}
}

#endif // _TEMPEST_FITTING_CUDA_HH_
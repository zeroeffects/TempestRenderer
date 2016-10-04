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

#ifndef _TEMPEST_COMPUTE_CONVENIENCE_HH_
#define _TEMPEST_COMPUTE_CONVENIENCE_HH_

#include "tempest/utils/config.hh"


#include "tempest/compute/ray-tracing-cuda.hh"
#include "tempest/mesh/obj-loader.hh"

namespace Tempest
{
template<class TRendenrer>
inline void RebindMaterialsToGPU(TRendenrer* rt_scene, RTMeshBlob& mesh_blob)
{
	std::unique_ptr<Tempest::RTMicrofacetMaterial[]> rt_materials(new Tempest::RTMicrofacetMaterial[mesh_blob.MaterialCount]);
	for(uint32_t material_idx = 0; material_idx < mesh_blob.MaterialCount; ++material_idx)
	{
		auto& cpu_mat = mesh_blob.Materials[material_idx];
		auto& rt_mat = rt_materials[material_idx];
		rt_mat = cpu_mat;
		if(cpu_mat.DiffuseMap)
		{
			rt_mat.DiffuseMap = rt_scene->bindTexture(reinterpret_cast<const Tempest::Texture*>(cpu_mat.DiffuseMap));
		}
		if(cpu_mat.SpecularMap)
		{
			rt_mat.SpecularMap = rt_scene->bindTexture(reinterpret_cast<const Tempest::Texture*>(cpu_mat.SpecularMap));
		}
	}
	for(uint32_t submesh_idx = 0; submesh_idx < mesh_blob.SubmeshCount; ++submesh_idx)
	{
		auto& submesh = mesh_blob.Submeshes[submesh_idx];
		submesh.Material = static_cast<Tempest::RTMaterial*>(&rt_materials[static_cast<Tempest::RTMicrofacetMaterial*>(submesh.Material) - mesh_blob.Materials]);
	}
	delete[] mesh_blob.Materials;
	mesh_blob.Materials = rt_materials.release();
}

#ifndef DISABLE_CUDA
#ifdef __CUDACC__
template<class TLoopBody>
__global__ void ParallelForLoop2DImpl(uint32_t width, uint32_t height, TLoopBody body)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int worker_id = threadIdx.y * blockDim.x + threadIdx.x;

	if(x >= width || y >= height)
		return;

	body(worker_id, x, y);
}

template<class TPool, class TLoopBody>
inline void ExecuteParallelForLoop2DGPU(uint32_t, TPool&, uint32_t width, uint32_t height, TLoopBody& body)
{
	dim3 group_size(8, 8, 1);
    dim3 thread_groups((width + group_size.x - 1)/group_size.x,
                       (height + group_size.y - 1)/group_size.y, 1);

	ParallelForLoop2DImpl<<<thread_groups, group_size>>>(width, height, body);

#ifndef NDEBUG
    cudaThreadSynchronize();
    auto status = cudaGetLastError();
    TGE_ASSERT(status == cudaSuccess, "Failed to launch kernel");
#endif
}

template<class TLoopBody>
__global__ void ParallelForLoop2DImpl(uint32_t size, TLoopBody body)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int worker_id = threadIdx.x;

	if(idx >= size)
		return;

	body(worker_id, idx);
}

template<class TPool, class TLoopBody>
inline void ExecuteParallelForLoopGPU(uint32_t, TPool&, uint32_t size, TLoopBody& body)
{
	dim3 group_size(64, 1, 1);
    dim3 thread_groups((size + group_size.x - 1)/group_size.x, 1, 1);

	ParallelForLoop2DImpl<<<thread_groups, group_size>>>(size, body);

#ifndef NDEBUG
    cudaThreadSynchronize();
    auto status = cudaGetLastError();
    TGE_ASSERT(status == cudaSuccess, "Failed to launch kernel");
#endif
}
#endif

template<class TPool, class TLoopBody>
inline void ExecuteParallelForLoop2DCPU(uint32_t id, TPool& pool, uint32_t width, uint32_t height, TLoopBody& body)
{
	auto func = Tempest::CreateParallelForLoop2D(width, height, 16, body);
    pool.enqueueTask(&func);
	pool.waitAndHelp(id, &func);
}
template<class TPool, class TLoopBody>
inline void ExecuteParallelForLoopCPU(uint32_t id, TPool& pool, uint32_t size, TLoopBody& body)
{
	auto func = Tempest::CreateParallelForLoop(size, 16, 
        [&body](unsigned worker_id, unsigned start_idx, unsigned chunk_size)
        {   
            for(unsigned idx = start_idx, idx_end = start_idx + chunk_size; idx < idx_end; ++idx)
            {
                body(worker_id, idx);
            }
        });
    pool.enqueueTask(&func);
	pool.waitAndHelp(id, &func);
}
#endif
}

#endif // _TEMPEST_COMPUTE_CONVENIENCE_HH_
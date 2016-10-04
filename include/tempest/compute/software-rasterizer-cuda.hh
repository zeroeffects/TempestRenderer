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

#ifndef _TEMPEST_CUDA_SOFTWARE_RASTERIZER_HH_
#define _TEMPEST_CUDA_SOFTWARE_RASTERIZER_HH_

#include "tempest/math/vector2.hh"

#include <vector_types.h>

namespace Tempest
{
namespace RasterizerCuda
{
template<class TShaderFunc>
__global__ void RasterizeDisk2Kernel(uint32_t x_off, uint32_t xend, uint32_t y_off, uint32_t yend,
                                     uint32_t width, uint32_t height, Disk2 disk, TShaderFunc shader)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + x_off;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + y_off;

    if(x >= xend || y >= yend)
        return;

    Vector2 tc{ (float)x, (float)y };

    Vector2 dist = tc - disk.Center;
    float len = Length(dist);
    if(disk.InnerRadius > len || len > disk.OuterRadius)
        return;

    shader(x, y, width, height, dist);
}

template<class TShaderFunc>
void RasterizeDisk2(const Disk2& disk, uint32_t width, uint32_t height, const TShaderFunc& shader)
{
    auto center_unorm = disk.Center;
    auto rad = ToVector2(disk.OuterRadius);
    auto min_corner = center_unorm - rad;
    auto max_corner = center_unorm + rad;

    min_corner = Vector2Max(min_corner, 0.0f);
    Vector2 box_size{ (float)width, (float)height };
    max_corner = Vector2Min(Vector2Ceil(max_corner), box_size);

    uint32_t x = (uint32_t)min_corner.x, xend = (uint32_t)max_corner.x, xrange = xend - x;
    uint32_t y = (uint32_t)min_corner.y, yend = (uint32_t)max_corner.y, yrange = yend - y;

    dim3 group_size(16, 16, 1);
    dim3 thread_groups((xrange + group_size.x - 1)/group_size.x,
                       (yrange + group_size.y - 1)/group_size.y, 1);

    RasterizeDisk2Kernel<<<thread_groups, group_size>>>(x, xend, y, yend, width, height, disk, shader);

#ifndef NDEBUG
    cudaThreadSynchronize();
    auto err = cudaGetLastError();
    TGE_ASSERT(err == cudaSuccess, "Broken kernel");
#endif
}

template<class TShaderFunc>
__global__ void RasterizeCapsule2Kernel(uint32_t x_off, uint32_t xend, uint32_t y_off, uint32_t yend,
                                        uint32_t width, uint32_t height,
                                        Vector2 corner0, Vector2 corner1, Vector2 dir, float len, float radius2, TShaderFunc shader)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + x_off;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + y_off;

    if(x >= xend || y >= yend)
        return;

    Vector2 tc{ (float)x, (float)y };

    Vector2 dist = {};
	Vector2 dist_to_point0 = tc - corner0;
	float parallel_dist = Dot(dir, dist_to_point0);
	if(parallel_dist < 0.0f)
	{
        dist = dist_to_point0;
		if(Dot(dist_to_point0, dist_to_point0) > radius2)
			return;
	}
	else if(parallel_dist > len)
	{
		Vector2 dist_to_point1 = tc - corner1;
        dist = dist_to_point1;
        if(Dot(dist_to_point1, dist_to_point1) > radius2)
			return;
	}
	else
	{
        Vector2 proj_vector = dir*Dot(dir, dist_to_point0);

        dist = dist_to_point0 - proj_vector;
        float dist_to_line = Length(dist);

        float dbg_dist_to_line = WedgeZ(dir, dist_to_point0);
        TGE_ASSERT(fabsf(Length(dist) - fabsf(dbg_dist_to_line)) < 1e-2f, "Invalid distance");

		if(dist_to_line*dist_to_line > radius2)
			return;
	}
            
    float tangent_ratio = parallel_dist/len;

    shader(x, y, width, height, dist, tangent_ratio);
}


template<class TShaderFunc>
void RasterizeCapsule2(const Capsule2& capsule, uint32_t width, uint32_t height, const TShaderFunc& shader)
{
    auto corner0 = capsule.Center[0],
		 corner1 = capsule.Center[1];
	
	auto min_corner = corner0,
		 max_corner = corner1;

	SwapBounds(min_corner.x, max_corner.x);
	SwapBounds(min_corner.y, max_corner.y);

	auto rad = ToVector2(capsule.Radius);
	min_corner -= rad;
	max_corner += rad;

	auto unorm_dir = corner1 - corner0;
	float len2 = Dot(unorm_dir, unorm_dir);
	float len = sqrtf(len2);
    Vector2 dir{ 1.0f, 0.0f };
    if(len2)
        dir = unorm_dir/len;

	float radius2 = capsule.Radius*capsule.Radius;

    min_corner = Vector2Max(min_corner, 0.0f);
    Vector2 box_size{ (float)width, (float)height };
    max_corner = Vector2Min(Vector2Ceil(max_corner), box_size);

    uint32_t x = (uint32_t)min_corner.x, xend = (uint32_t)max_corner.x, xrange = xend - x;
    uint32_t y = (uint32_t)min_corner.y, yend = (uint32_t)max_corner.y, yrange = yend - y;

    dim3 group_size(16, 16, 1);
    dim3 thread_groups((xrange + group_size.x - 1)/group_size.x,
                       (yrange + group_size.y - 1)/group_size.y, 1);
    RasterizeCapsule2Kernel<<<thread_groups, group_size>>>(x, xend, y, yend, width, height, corner0, corner1, dir, len, radius2, shader);
}
}
}

#endif // _TEMPEST_CUDA_SOFTWARE_RASTERIZER_HH_

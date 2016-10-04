/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_SOFTWARE_RASTERIZER_HH_
#define _TEMPEST_SOFTWARE_RASTERIZER_HH_

#include "tempest/math/vector2.hh"
#include <cstdint>
#include "tempest/math/shapes.hh"

namespace Tempest
{
namespace Rasterizer
{
// Far from optimal, but it would be easier to port it to the GPU
// TODO: Make optimal scan-line solution

template<class TShaderFunc>
void RasterizeDisk2(const Disk2& disk, uint32_t width, uint32_t height, TShaderFunc shader)
{
    auto center_unorm = disk.Center;
    auto rad = ToVector2(disk.OuterRadius);
    auto min_corner = center_unorm - rad;
    auto max_corner = center_unorm + rad;

    min_corner = Vector2Max(min_corner, 0.0f);
    Vector2 box_size{ (float)width, (float)height };
    max_corner = Vector2Min(Vector2Ceil(max_corner), box_size);

    for(uint32_t y = (uint32_t)min_corner.y, yend = (uint32_t)max_corner.y; y < yend; ++y)
    {
        for(uint32_t x = (uint32_t)min_corner.x, xend = (uint32_t)max_corner.x; x < xend; ++x)
        {
            Vector2 tc{ (float)x, (float)y };

            Vector2 dist = tc - center_unorm;
            float len = Length(dist);
            if(disk.InnerRadius > len || len > disk.OuterRadius)
                continue;

             shader(x, y, width, height, dist);
        }
    }
}

template<class TShaderFunc>
void RasterizeCapsule2(const Capsule2& capsule, uint32_t width, uint32_t height, TShaderFunc shader)
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

    for(uint32_t y = (uint32_t)min_corner.y, yend = (uint32_t)max_corner.y; y < yend; ++y)
    {
        for(uint32_t x = (uint32_t)min_corner.x, xend = (uint32_t)max_corner.x; x < xend; ++x)
        {
            Vector2 tc{ (float)x, (float)y };

            Vector2 dist = {};
			Vector2 dist_to_point0 = tc - corner0;
			float parallel_dist = Dot(dir, dist_to_point0);
			if(parallel_dist < 0.0f)
			{
                dist = dist_to_point0;
				if(Dot(dist_to_point0, dist_to_point0) > radius2)
					continue;
			}
			else if(parallel_dist > len)
			{
				Vector2 dist_to_point1 = tc - corner1;
                dist = dist_to_point1;
                if(Dot(dist_to_point1, dist_to_point1) > radius2)
					continue;
			}
			else
			{
                Vector2 proj_vector = dir*Dot(dir, dist_to_point0);

                dist = dist_to_point0 - proj_vector;
                float dist_to_line = Length(dist);

                float dbg_dist_to_line = WedgeZ(dir, dist_to_point0);
                TGE_ASSERT(fabsf(Length(dist) - fabsf(dbg_dist_to_line)) < 1e-2f, "Invalid distance");

				if(dist_to_line*dist_to_line > radius2)
					continue;
			}
            
            float tangent_ratio = parallel_dist/len;

            shader(x, y, width, height, dist, tangent_ratio);
        }
    }
}
}
}

#endif // _TEMPEST_SOFTWARE_RASTERIZER_HH_

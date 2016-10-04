/*   
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

#ifndef _TEMPEST_GABOR_NOISE_HH_
#define _TEMPEST_GABOR_NOISE_HH_

#include "tempest/math/sampling1.hh"
#include "tempest/math/morton.hh"

// Procedural Noise using Sparse Gabor Convolution.
// Ares Lagae, Sylvain Lefebvre, George Drettakis and Philip Dutre.
// ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH 2009) 28(3), 2009.
// http://graphics.cs.kuleuven.be/publications/LLDD09PNSGC/

namespace Tempest
{
float GaborKernel(float strength, float bandwidth, float frequency, float cos_omega, float sin_omega, float x, float y)
{
    float gaussian = strength * expf(-Tempest::MathPi*(bandwidth*bandwidth)*(x*x + y*y));
    float cos_wave = cosf(2.0f*Tempest::MathPi*frequency*(x*cos_omega + y*sin_omega));
    return gaussian*cos_wave;
}

inline unsigned Poisson(float mean, unsigned* seed)
{
    float g = expf(-mean);
    unsigned em = 0;
    float t = FastFloatRand(*seed);
    while(t > g)
    {
        ++em;
        t *= FastFloatRand(*seed);
    }
    return em;
}

inline float GaborNoise(float strength, float bandwidth, float frequency, float cos_omega, float sin_omega, unsigned num_impulses, float x, float y, unsigned random_offset = 0)
{
    float rcp_kernel_radius = bandwidth*(1.0f/0.9765097024771845f); // sqrtf(-logf(0.05)/math.pi)
    x *= rcp_kernel_radius;
    y *= rcp_kernel_radius;
    float kernel_radius = 1.0f/rcp_kernel_radius;
    int x_trunc = Tempest::FastFloorToInt(x),
        y_trunc = Tempest::FastFloorToInt(y);
    float frac_x = x - x_trunc,
          frac_y = y - y_trunc;
    float noise = 0.0f;
    float number_of_impulses_per_cell = num_impulses*(1.0f/Tempest::MathPi);

    
    for(int dj = -1; dj <= 1; ++dj)
    {
        for(int di = -1; di <= 1; ++di)
        {
            float cur_x = frac_x - di;
            float cur_y = frac_y - dj;

            int sub_x = x_trunc + di;
            int sub_y = y_trunc + dj;

            unsigned seed = Tempest::EncodeMorton2(sub_x, sub_y) + random_offset; // TODO: What about -1?
            //seed = Tempest::Hash32(seed);

            unsigned num_impulses_gen = Poisson(number_of_impulses_per_cell, &seed);

            for(unsigned impulse = 0; impulse < num_impulses_gen; ++impulse)
            {
                float x_i = Tempest::FastFloatRand(seed);
                float y_i = Tempest::FastFloatRand(seed);
                float w_i = Tempest::FastFloatRand(seed)*2.0f - 1.0f;
                
                float x_i_x = cur_x - x_i;
                float y_i_y = cur_y - y_i;
                if(((x_i_x*x_i_x) + (y_i_y*y_i_y)) < 1.0)
                {
                    noise += w_i*GaborKernel(strength, bandwidth, frequency, cos_omega, sin_omega, x_i_x*kernel_radius, y_i_y*kernel_radius);
                }
            }
        }
    }

    return noise;
}

float GaborVariance(float strength, float spread, float base_frequency, unsigned num_impulses)
{
    float kernel_radius = 0.9765097024771845f/spread;
    float integral = ((strength*strength)/(4.0f*spread*spread)) * (1.0f + expf(-(2.0f*MathPi*base_frequency*base_frequency) / (spread*spread)));
    return (1.0f / 3.0f) * integral * num_impulses / (MathPi * kernel_radius * kernel_radius);
}
}

#endif // _TEMPEST_GABOR_NOISE_HH_
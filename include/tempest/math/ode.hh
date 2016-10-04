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

#ifndef _TEMPEST_ODE_HH_
#define _TEMPEST_ODE_HH_

#include "tempest/math/vector2.hh"

namespace Tempest
{
#ifdef __CUDA_ARCH__
#	define EXPORT_CONSTANT __constant__
#else
#	define EXPORT_CONSTANT const
#endif

EXPORT_CONSTANT float SimulationEpsilon = 1e-6f;

struct EulerODESolver
{
	// a x' + b x + c = 0
	static EXPORT_CUDA float solveFirstOrderLinearODE(float a, float b, float c, float initial_value, float time, float step_size)
	{
		float derivative = (-b*initial_value - c)/a;

		return initial_value + derivative*step_size;
	}

	// a x'' + b x ' + c x + d = 0
	static EXPORT_CUDA void solveSecondOrderLinearODE(float a, float b, float c, float d, float step_size, float* inout_value, float* inout_first_deriv)
	{
		float initial_first_deriv = *inout_first_deriv;
		*inout_value += step_size*initial_first_deriv;
		*inout_first_deriv = initial_first_deriv - step_size*(b*initial_first_deriv + c*(*inout_value) + d)/a;
	}
};

struct RK4ODESolver
{
	static EXPORT_CUDA float solveFirstOrderLinearODE(float a, float b, float c, float initial_value, float time, float step_size)
	{
		float const_factor = c/a;

		float x1 = initial_value;
		float v1 = -b*x1/a - const_factor;

		float half_step = step_size*0.5f;

		float x2 = x1 + v1*half_step;
		float v2 = -b*x2/a - const_factor;

		float x3 = x1 + v2*half_step;
		float v3 = -b*x3/a - const_factor;

		float x4 = x1 + v3*step_size;
		float v4 = -b*x4/a - const_factor;

		return x1 + step_size*(1.0f/6.0f)*(v1 + 2.0f*(v2 + v3) + v4);
	}

	static EXPORT_CUDA void solveSecondOrderLinearODE(float a, float b, float c, float d, float step_size, float* inout_value, float* inout_first_deriv)
	{
		float const_factor = d/a;

		float x1 = *inout_value;
		float v1 = *inout_first_deriv;
		float a1 = -(b*v1 + c*x1)/a - const_factor;

		float half_step = step_size*0.5f;

		float x2 = x1 + v1*half_step;
		float v2 = v1 + a1*half_step;
		float a2 = -(b*v2 + c*x2)/a - const_factor;

		float x3 = x1 + v2*half_step;
		float v3 = v1 + a2*half_step;
		float a3 = -(b*v3 + c*x3)/a - const_factor;
    
		float x4 = x1 + v3*step_size;
		float v4 = v1 + a3*step_size;
		float a4 = -(b*v4 + c*x4)/a - const_factor;

		*inout_value = x1 + step_size*(1.0f/6.0f)*(v1 + 2.0f*(v2 + v3) + v4);
		*inout_first_deriv = v1 + step_size*(1.0f/6.0f)*(a1 + 2.0f*(a2 + a3) + a4);
	}
};

// TODO: Support non-linear
EXPORT_CUDA void BackwardEulerMethodSolveSecondOrderLinearODE(float a, float b, float c, float d, float step_size, float* inout_value, float* inout_first_deriv)
{
	float initial_first_deriv = *inout_first_deriv;
	float C = -c/a,
		  B = -b/a;
	*inout_value += initial_first_deriv*step_size;
	float delta_v = step_size*(*inout_value + step_size*C*initial_first_deriv)/(1 - step_size*B - step_size*step_size*C);
	float first_deriv = *inout_first_deriv += delta_v;
}

template<class T = RK4ODESolver>
EXPORT_CUDA void SolveSecondOrderLinearODE(float a, float b, float c, float d, float step_size, Vector2* inout_value, Vector2* inout_first_deriv)
{
    float dist = Length(*inout_value);
    Vector2 norm_spring_dir;
    if(dist > SimulationEpsilon)
    {
        norm_spring_dir = *inout_value/dist;
    }
    else
    {
        auto fd_len = Length(*inout_first_deriv);
        if(fd_len < SimulationEpsilon)
            return;
        norm_spring_dir = *inout_first_deriv/fd_len;
    }
    float orig_vel = Dot(norm_spring_dir, *inout_first_deriv);
    float result_vel = orig_vel;

    T::solveSecondOrderLinearODE(a, b, c, d, step_size, &dist, &result_vel);
    *inout_first_deriv += (result_vel - orig_vel)*norm_spring_dir;
    *inout_value = dist*norm_spring_dir;
}

template<class T = RK4ODESolver>
EXPORT_CUDA void SolveSecondOrderLinearODE(float a, float b, float c, float d, float step_size, float rest_distance, const Vector2& next_pos, Vector2* inout_pos, Vector2* inout_first_deriv)
{
    Vector2 dist_vec = *inout_pos - next_pos;
    Vector2 norm_dist_vec = dist_vec;
    NormalizeSelf(&norm_dist_vec);

    Vector2 offset_vec = rest_distance*norm_dist_vec;
    dist_vec -= offset_vec;

    SolveSecondOrderLinearODE<T>(a, b, c, d, step_size, &dist_vec, inout_first_deriv);
    *inout_pos = dist_vec + next_pos + offset_vec;
}

template<class T = RK4ODESolver>
EXPORT_CUDA void SolveSecondOrderLinearODEStretch(float a, float b, float c, float d, float step_size, const Vector2& next_stretch, Vector2* inout_stretch, Vector2* inout_first_deriv)
{
    Vector2 dist_vec = *inout_stretch - next_stretch;
    SolveSecondOrderLinearODE<T>(a, b, c, d, step_size, &dist_vec, inout_first_deriv);
    *inout_stretch = dist_vec + next_stretch;
}
}

#endif // _TEMPEST_ODE_HH_

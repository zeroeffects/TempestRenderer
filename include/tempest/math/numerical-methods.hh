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

#ifndef _TEMPEST_NUMERICAL_METHODS_HH_
#define _TEMPEST_NUMERICAL_METHODS_HH_

#include "tempest/math/functions.hh"
#include "tempest/math/sampling3.hh"
#include "tempest/utils/threads.hh"
#include "tempest/utils/memory.hh"

namespace Tempest
{
template<class TFunc, class TDeriv>
EXPORT_CUDA float NewtonRaphsonMethod(float guess, float epsilon, TFunc func, TDeriv deriv)
{
    float value;
    while(fabsf(value = func(guess)) > epsilon)
    {
        float deriv_value = deriv(guess);
        guess -= value/deriv_value;
    }
    return guess;
}

template<class TFunc>
EXPORT_CUDA float BrentMethod(float a, float b, float epsilon,
#ifdef DEBUG_NUMERICAL
                 size_t* dbg_steps,
#endif
                 TFunc f)
{
#ifdef DEBUG_NUMERICAL
    *dbg_steps = 0;
#endif
    float fa = f(a);
    float fb = f(b);

    TGE_ASSERT(fa * fb <= 0.0f, "Not bracketed input!");
    
    if(fabs(fa) < fabs(fb))
    {
        Swap(a, b);
        Swap(fa, fb);
    }
    float c = a, s = a, d = a;
    bool mflag = true;
    for(float fs = fa, fc = fa; fabs(fb) > epsilon && fabs(b - a) >= epsilon;)
    {
        if(ApproxNotEqual(fa, fc) && ApproxNotEqual(fb, fc))
        {
            float rcp_fab = 1.0f/(fa - fb);
            float rcp_fac = 1.0f/(fa - fc);
            float rcp_fcb = 1.0f/(fc - fb);
            s = a * fb * fc * rcp_fab * rcp_fac +
                b * fa * fc * rcp_fab * rcp_fcb - 
                c * fa * fb * rcp_fac * rcp_fcb;
        }
        else
        {
            b -= fb * (b - a) / (fb - fa);
        }

        mflag = (3.0f*a + b)*0.25f > s || s > b ||
                 ( mflag && (fabs(s - b) >= fabs(b - c)*0.5f ||
                             fabs(b - c) < epsilon) ||
                  !mflag && (fabs(s - b) >= fabs(c - d)*0.5f ||
                             fabs(c - d) < epsilon)
                 );
        if(mflag)
        {
            s = (a + b)*0.5f;
        }
        fs = f(s);
        d = c;
        c = b;
        if(fa * fs < 0)
        {
            b = s;
        }
        else
        {
            a = s;
        }

        fa = f(a);
        fb = f(b);
        fc = f(c);

        if(fabs(fa) < fabs(fb))
        {
            Swap(a, b);
            Swap(fa, fb);
        }

#ifdef DEBUG_NUMERICAL
        ++*dbg_steps;
#endif
    }
    return b;
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn SimpsonsCompositeRuleQuadratureIntegrator(float a, float b, uint32_t samples, TFunc func)
{
    float step = (b - a) / (samples + 1), simp_factor = 4, theta = a + step;

	TReturn integral = func(a) + func(b);
	for(uint32_t i = 0; i < samples; ++i)
    {
		TReturn value = func(theta);
		integral += value * simp_factor;
		theta += step;
		simp_factor = 6 - simp_factor;
	}

    return integral * step * (1.0f/3.0f);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn MonteCarloIntegrator(float a, float b, uint32_t samples, TFunc func)
{
    TReturn total{};

    unsigned seed = 1;

    float pdf = 1.0f/(b - a);

    for(uint32_t i = 0; i < samples; ++i)
    {
        float t = a + (b - a)*Tempest::FastFloatRand(seed);

		total += func(t);
	}

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn StratifiedMonteCarloIntegrator(float a, float b, uint32_t samples, TFunc func)
{
    TReturn total{};
    
    unsigned seed = 1;

    float pdf = 1.0f/(b - a);

    for(uint32_t i = 0; i < samples; ++i)
    {
        float mix = (float)(i + Tempest::FastFloatRand(seed))/samples;
        float t = a + (b - a)*mix;

		total += func(t);
	}

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn StratifiedMonteCarloIntegratorSphere(uint32_t samples, TFunc func)
{
    TReturn total{};
    
    unsigned seed = 1;

    float pdf = 1.0f/(4.0f*MathPi);

    float sqrt_samples = sqrtf((float)samples);
    uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
    TGE_ASSERT((samples % sqrt_samples_i) == 0, "Invalid sample count");
    
    for(uint32_t i = 0; i < samples; ++i)
    {
        uint32_t x_idx = i % sqrt_samples_i;
        uint32_t y_idx = i / sqrt_samples_i;
        float x = (x_idx + Tempest::FastFloatRand(seed))/sqrt_samples;
        float y = (y_idx + Tempest::FastFloatRand(seed))/sqrt_samples;

        auto dir = Tempest::UniformSampleSphere(x, y);
        
		total += func(dir);
	}

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn StratifiedMonteCarloIntegratorHemisphere(uint32_t samples, TFunc func)
{
    TReturn total{};
    
    unsigned seed = 1;

    float pdf = 1.0f/(2.0f*MathPi);

    float sqrt_samples = sqrtf((float)samples);
    uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
    TGE_ASSERT((samples % sqrt_samples_i) == 0, "Invalid sample count");
    
    for(uint32_t i = 0; i < samples; ++i)
    {
        uint32_t x_idx = i % sqrt_samples_i;
        uint32_t y_idx = i / sqrt_samples_i;
        float x = (x_idx + Tempest::FastFloatRand(seed))/sqrt_samples;
        float y = (y_idx + Tempest::FastFloatRand(seed))/sqrt_samples;

        auto dir = Tempest::UniformSampleHemisphere(x, y);
        
		total += func(dir);
	}

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn StratifiedMonteCarloIntegratorSphericalCone(uint32_t samples, float cos_angle, TFunc func)
{
    TReturn total{};
    
    unsigned seed = 1;

    float pdf = Tempest::UniformSphericalConePDF(cos_angle);

    float sqrt_samples = sqrtf((float)samples);
    uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
    TGE_ASSERT((samples % sqrt_samples_i) == 0, "Invalid sample count");
    
    for(uint32_t i = 0; i < samples; ++i)
    {
        uint32_t x_idx = i % sqrt_samples_i;
        uint32_t y_idx = i / sqrt_samples_i;
        float x = (x_idx + Tempest::FastFloatRand(seed))/sqrt_samples;
        float y = (y_idx + Tempest::FastFloatRand(seed))/sqrt_samples;

        auto dir = Tempest::UniformSampleSphericalCone(cos_angle, x, y);
        
		total += func(dir);
	}

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
TReturn ParallelStratifiedMonteCarloIntegratorHemisphere(uint32_t id, ThreadPool& pool, uint32_t samples, uint32_t chunk_size, TFunc func)
{
    unsigned seed = 1;

    float pdf = 1.0f/(2.0f*MathPi);

    float sqrt_samples = sqrtf((float)samples);
    uint32_t sqrt_samples_i = (uint32_t)(int32_t)sqrt_samples;
    
    TGE_ASSERT((samples % sqrt_samples_i) == 0, "Invalid sample count");
    
    auto thread_count = pool.getThreadCount();
    TReturn* partial_total = TGE_TYPED_ALLOCA(TReturn, thread_count);
    memset(partial_total, 0, thread_count*sizeof(TReturn));

    auto parallel = Tempest::CreateParallelForLoop2D(sqrt_samples_i, sqrt_samples_i, chunk_size,
        [&seed, sqrt_samples, &func, partial_total](uint32_t worker_id, uint32_t x_idx, uint32_t y_idx)
        {
            float xf = (x_idx + Tempest::FastFloatRand(seed))/sqrt_samples;
            float yf = (y_idx + Tempest::FastFloatRand(seed))/sqrt_samples;

            auto dir = Tempest::UniformSampleHemisphere(xf, yf);
        
		    partial_total[worker_id] += func(dir);
        });

    pool.enqueueTask(&parallel);
    pool.waitAndHelp(id, &parallel);

    TReturn total = partial_total[0];
    for(uint32_t thread_idx = 1; thread_idx < thread_count; ++thread_idx)
    {
        total += partial_total[thread_idx];
    }

    return total / ((float)samples*pdf);
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn CircularAreaIntegratorTessellationApproximation(float a, float b, uint32_t samples, TFunc func)
{
    float step = (b - a) / (samples - 1), theta;

    TReturn fprev = func(a);
    TReturn integral = 0.0f;
    float sin_step = sinf(step);
    for(uint32_t i = 1; i < samples; ++i)
    {
        theta = i*step;
		TReturn fcur = func(theta);

        TReturn segment_size = fabsf(fcur*fprev)*sin_step*0.5f;
		integral += segment_size; // half area of the parallelogram
        fprev = fcur;
	}
    return integral;
}

template<class TReturn = float, class TFunc>
EXPORT_CUDA TReturn ArcLengthIntegrator(float a, float b, uint32_t samples, TFunc func)
{
    float step = (b - a) / (samples - 1), theta;

    TReturn fprev = func(a);
    TReturn integral = 0.0f;
    float cos_step = cosf(step);

    for(uint32_t i = 1; i < samples; ++i)
    {
        theta = i*step;
		TReturn fcur = func(theta);

        TReturn segment_size = sqrtf(fprev*fprev + fcur*fcur - 2.0f*fcur*fprev*cos_step);

		integral += segment_size; // half area of the parallelogram
        fprev = fcur;
	}
    return integral;
}
}

#endif // _TEMPEST_NUMERICAL_METHODS_HH_

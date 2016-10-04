/*   The MIT License
 *
 *   Tempest Engine
 *   Copyright (c) 2015 Sebastian Werner
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

#ifndef WINDOWING_HH
#define WINDOWING_HH

#include "tempest/utils/assert.hh"
#include "tempest/math/functions.hh"

namespace Tempest
{

inline EXPORT_CUDA float DiscreteTukeyWindow(int currentIndex, int sampleNumber, float alpha) {
    /*DOCSTRING: Returns the Tukey window value for the given currentIndex with respect to an array with sampleNumber elements.
     * alpha gives the steepness of the window at the edges, where alpha=0 is a rectangle and alpha=1 is the Hann-window
    */
    TGE_ASSERT(currentIndex > 0, "Index needs to be non-negative");
    if (currentIndex <= alpha * (sampleNumber - 1)/2.0f) {
        return 0.5f * (1.0f + cosf(MathPi * (2.0f * currentIndex/(alpha * (float)(sampleNumber - 1) ) - 1.0f)));
    }
    if (currentIndex <= ((1.0f - alpha/2.0f) * (sampleNumber - 1))) {
        return 1.0f;
    }
    TGE_ASSERT(currentIndex < sampleNumber, "Index must not exceed total number of elements");
    return 0.5f * (1.0f + cosf(MathPi * (2.0f * currentIndex/(alpha * (float)((sampleNumber - 1) - 2.0f/alpha + 1.0f) ) - 1.0f)));
}

inline EXPORT_CUDA float ContinuousTukeyWindow(float position, float length, float alpha) {
    /*DOCSTRING: Returns the Tukey window value for the given position (1D) with respect to an given maximal length.
     * alpha gives the steepness of the window at the edges, where alpha=0 is a rectangle and alpha=1 is the Hann-window
    */
    TGE_ASSERT(position > 0, "Index needs to be non-negative");
    if (position <= alpha * length/2.0f) {
        return 0.5f * (1.0f + cosf(MathPi * (2.0f * position/(alpha * length ) - 1.0f)));
    }
    if (position <= ((1.0f - alpha/2.0f) * length)) {
        return 1.0f;
    }
    TGE_ASSERT(position <= length, "Index must not exceed total length");
    return 0.5f * (1.0f + cosf(MathPi * (2.0f * position/(alpha * (float)(length - 2.0f/alpha + 1.0f) ) - 1.0f)));
}

inline EXPORT_CUDA float ContinuousGaussianWindow(float position, float mean, float stdev) {
    /*DOCSTRING: Returns the Gaussian window value for the given position (1D) with respect to an given mean.
     * stdev is the standard deviation of the gaussian
    */
    float expArg = (position - mean)/stdev;
    return (expf(-0.5f * expArg * expArg));
}

inline EXPORT_CUDA float ContinuousNormalDistribution(float position, float mean, float stdev) {
    /*DOCSTRING: Returns the Gaussian window value for the given position (1D) with respect to an given mean.
     * stdev is the standard deviation of the gaussian
    */
    float invFactor = 1.0f/(sqrtf(2.0f * MathPi) * stdev);
    float expArg = (position - mean)/stdev;
    return invFactor * (expf(-0.5f * expArg * expArg));
}

inline EXPORT_CUDA float ContinuousTukeyWindowFT(float position, float width, float alpha)
{
    /*DOCSTRING: Returns the analytic Fourier transform for a Tukey window with given width and alpha value.
     * The limit cases are considered and alpha is clamped to [0.0, 1.0], ensuring safe evaluation.
    */
    if (alpha <= 0.0f) {
        return Sinc(MathPi * width * position);
    }

    alpha = Maxf(alpha, 1.0f);
    //Calculate u**2/(u**2 - (1/alpa * width)**2)
    float uSquared = position * position;

    float invAlphaWidthSquared = 1.0f/(alpha * width);
    invAlphaWidthSquared *= invAlphaWidthSquared;

    float invSubtract = (uSquared)/(uSquared - invAlphaWidthSquared);

    if (alpha == 1.0f ) {
        float result = Sinc(MathPi * width * position) * 0.5f * width * (1.0f - invSubtract);
        return result;
    }

    float oneSubAlpha = (1.0f - alpha);
    float result = Sinc(MathPi * width * position) * 0.5f * width * (1.0f - invSubtract);
    result += Sinc(MathPi * width * position * oneSubAlpha) * 0.5f * width * oneSubAlpha * (1.0f - invSubtract);

    return result;
}

}
#endif //WINDOWING_HH

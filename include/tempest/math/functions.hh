/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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

#ifndef _TEMPEST_MATH_FUNCTIONS_HH_
#define _TEMPEST_MATH_FUNCTIONS_HH_

#include <limits>
#include <cmath>

namespace Tempest
{
//! Mathematical constant
constexpr float math_pi = 3.141592f;

inline float approx_eq(float lhs, float rhs)
{
    return fabs(rhs - lhs) <= std::numeric_limits<float>::epsilon();
}

inline float approx_neq(float lhs, float rhs)
{
    return fabs(rhs - lhs) > std::numeric_limits<float>::epsilon();
}

//! Converts degrees to radians
/*!
    \param val a floating-point number argument
    \return the angle in radians
*/
inline float to_radians(float val) { return (val * math_pi) / 180.0f; }

inline float to_degrees(float val) { return (val * 180.0f) / math_pi; }
}

#endif // _TEMPEST_MATH_FUNCTIONS_HH_
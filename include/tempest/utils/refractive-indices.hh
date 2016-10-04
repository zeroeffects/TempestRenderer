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

#ifndef _TEMPEST_REFRACTION_INDICES_HH_
#define _TEMPEST_REFRACTION_INDICES_HH_

#include "tempest/math/vector2.hh"

namespace Tempest
{
// If you need more just go to this website and add it below:
// http://refractiveindex.info/
const Vector2 SilverRefractiveIndex{ 0.15f, 3.4727f };
const Vector2 AluminiumRefractiveIndex{ 1.0972f, 6.7943f };
const Vector2 GoldRefractiveIndex{ 1.5249f, 1.9600f };
const Vector2 ChromiumRefractiveIndex{ 0.75924f, 2.3692f };
const Vector2 IronRefractiveIndex{ 2.8735f, 3.3590f };
const Vector2 CelluloseRefractiveIndex{ 1.4613f, 0.0f };
}

#endif
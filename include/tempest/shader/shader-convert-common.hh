/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2013 Zdravko Velinov
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

#ifndef _SHADER_CONVERT_COMMON_HH_
#define _SHADER_CONVERT_COMMON_HH_

#include "tempest/utils/types.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
namespace Shader
{
class Type;
class Buffer;
class Variable;
class BufferDescription;
class EffectDescription;

uint32 ConvertStructBuffer(const string* opts, size_t opts_count, uint64 settings, const Variable* var, Shader::EffectDescription* fx_desc);
void ConvertBuffer(const string* opts, size_t opts_count, uint64 settings, const Buffer* buffer, Shader::EffectDescription* fx_desc);
}
}

#endif // _SHADER_CONVERT_COMMON_HH_
/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2010-2014 Zdravko Velinov
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

#ifndef _TEMPEST_SHADER_HH_
#define _TEMPEST_SHADER_HH_

#include "tempest/utils/patterns.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
struct DataDescription
{
    string           Name;
    UniformValueType Type;
    uint16           ElementSize;
    uint16           ElementCount;
    uint32           Offset;
};

struct ResourceTableDescription
{
    string                       Name;
    uint32                       BindPoint;
    uint32                       BufferSize;
    uint32                       ExtendablePart;
    PACKED_DATA(DataDescription) Uniforms;

    ResourceTableDescription(const ResourceTableDescription&) = delete;
    ResourceTableDescription& operator=(const ResourceTableDescription&) = delete;

private:
    ResourceTableDescription(uint32 count, uint32 extendable_part, string name, uint32 bind_point)
        :   ExtendablePart(extendable_part),
            Name(name),
            BindPoint(bind_point),
            Uniforms(count) {}
    ~ResourceTableDescription() = default;
};

struct ResourceIndex
{
    uint32 ResourceTableIndex = 0;
    uint32 BaseOffset = 0;
};
}

#endif // _TEMPEST_SHADER_HH_
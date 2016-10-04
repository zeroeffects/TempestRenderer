
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

#include "tempest/graphics/rendering-definitions.hh"

#include <unordered_map>

namespace Tempest
{
DataFormat TranslateDataFormat(const std::string& name)
{
    #define ELEMENT(fmt) { #fmt, DataFormat::fmt }
    static std::unordered_map<std::string, DataFormat> fmt_map
    {
        ELEMENT(R32F),
        ELEMENT(RG32F),
        ELEMENT(RGB32F),
        ELEMENT(RGBA32F),
        ELEMENT(R16F),
        ELEMENT(RG16F),
        //  ELEMENT(RGB16F),
        ELEMENT(RGBA16F),
        ELEMENT(R32),
        ELEMENT(RG32),
        ELEMENT(RGB32),
        ELEMENT(RGBA32),
        ELEMENT(R16),
        ELEMENT(RG16),
        //  ELEMENT(RGB16),
        ELEMENT(RGBA16),
        ELEMENT(R8),
        ELEMENT(RG8),
        //  ELEMENT(RGB8),
        ELEMENT(RGBA8),
        ELEMENT(uR32),
        ELEMENT(uRG32),
        ELEMENT(uRGB32),
        ELEMENT(uRGBA32),
        ELEMENT(uR16),
        ELEMENT(uRG16),
        //  ELEMENT(uRGB16),
        ELEMENT(uRGBA16),
        ELEMENT(uR8),
        ELEMENT(uRG8),
        //  ELEMENT(uRGB8),
        ELEMENT(uRGBA8),

        ELEMENT(R16SNorm),
        ELEMENT(RG16SNorm),
        //  ELEMENT(RGB16SNorm),
        ELEMENT(RGBA16SNorm),
        ELEMENT(R8SNorm),
        ELEMENT(RG8SNorm),
        //  ELEMENT(RGB8SNorm),
        ELEMENT(RGBA8SNorm),
        ELEMENT(R16UNorm),
        ELEMENT(RG16UNorm),
        //  ELEMENT(RGB16UNorm),
        ELEMENT(RGBA16UNorm),
        ELEMENT(R8UNorm),
        ELEMENT(RG8UNorm),
        //  ELEMENT(RGB8UNorm),
        ELEMENT(RGBA8UNorm),

        ELEMENT(D16),
        ELEMENT(D24S8),
        ELEMENT(D32),

        ELEMENT(R10G10B10A2),
        ELEMENT(uR10G10B10A2)
    };

    auto iter = fmt_map.find(name);
    TGE_ASSERT(iter != fmt_map.end(), "Unknown data format");
    if(iter == fmt_map.end())
        return DataFormat::Unknown;
    return iter->second;
}
}
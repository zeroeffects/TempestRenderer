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

#include "tempest/image/image.hh"
#include "tempest/image/tga-image.hh"
#include "tempest/image/png-image.hh"
#include "tempest/image/exr-image.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/system.hh"

namespace Tempest
{
Texture* LoadImage(const Path& file_path)
{
    auto ext = file_path.extension();
    for(auto& c : ext)
        c = tolower(c);
    if(ext == "tga")
    {
        return LoadTGAImage(file_path);
    }
    else if(ext == "png")
    {
        return LoadPNGImage(file_path);
    }
    else if(ext == "exr")
    {
        return LoadEXRImage(file_path);
    }
    else
    {
        Log(LogLevel::Error, "Unsupported file format: ", ext);
        return nullptr;
    }
}

bool SaveImage(const TextureDescription& tex, const void* data, const Path& file_path)
{
    auto ext = file_path.extension();
    for(auto& c : ext)
        c = tolower(c);
    if(ext == "tga")
    {
        return SaveTGAImage(tex, data, file_path);
    }
    else if(ext == "png")
    {
        return SavePNGImage(tex, data, file_path);
    }
    else if(ext == "exr")
    {
        return SaveEXRImage(tex, data, file_path);
    }
    else
    {
        Log(LogLevel::Error, "Unsupported file format: ", ext);
        return nullptr;
    }
    System::Touch(file_path.get());
}
}
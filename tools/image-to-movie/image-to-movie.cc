/*   The MIT License
*
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

#include "tempest/utils/parse-command-line.hh"
#include "tempest/utils/video-encode.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/image/image.hh"

#include <cstdlib>

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("image-to-movie", true);
    parser.createOption('r', "repeat-frames", "Repeat images for how many frames", "1");
    parser.createOption('o', "output", "Specify output video file", true, "video.ivf");

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto image_count = parser.getUnassociatedCount();
    if(image_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify at least one image");
    }

    auto texname = parser.getUnassociatedArgument(0);
    Tempest::TexturePtr tex(Tempest::LoadImage(Tempest::Path(texname)));
    if(!tex)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load texture: ", texname);
        return EXIT_FAILURE;
    }

    auto& first_hdr = tex->getHeader();

    auto width = first_hdr.Width,
         height = first_hdr.Height;

    Tempest::VideoInfo video_info;
    video_info.FileName = parser.extractString("output");
    video_info.Width = width;
    video_info.Height = height;
    video_info.Bitrate = 50000;

    Tempest::VPXVideoEncoder video_enc;
    if(!video_enc.openStream(video_info))
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to open video stream: ", video_info.FileName);
        return EXIT_FAILURE;
    }

    auto repeat_count = parser.extract<size_t>("repeat-frames");
    for(size_t repeat = 0; repeat < repeat_count; ++repeat)
    {
        video_enc.submitFrame(*tex);
    }

    for(size_t image_idx = 1; image_idx < image_count; ++image_idx)
    {
        auto texname = parser.getUnassociatedArgument(image_idx);
        tex = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(texname)));
        if(!tex)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load texture: ", texname);
            return EXIT_FAILURE;
        }

        auto& hdr = tex->getHeader();
        if(width != hdr.Width ||
           height != hdr.Height)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Incompatible size: (", width, ", ", height, ") compared to (", hdr.Width, ", ", hdr.Height, ")");
            return EXIT_FAILURE;
        }

        for(size_t repeat = 0; repeat < repeat_count; ++repeat)
        {
            video_enc.submitFrame(*tex);
        }
    }

    return EXIT_SUCCESS;
}
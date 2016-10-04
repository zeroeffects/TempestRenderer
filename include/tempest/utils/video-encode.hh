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

#ifndef _TEMPEST_VIDEO_ENCODE_HH_
#define _TEMPEST_VIDEO_ENCODE_HH_

#include <string>
#include <fstream>
#include <memory>

#include "tempest/graphics/texture.hh"

#include "libvpx/vpx/vpx_codec.h"

#define INPUT_BUFFER_COUNT 32

struct _NV_ENCODE_API_FUNCTION_LIST;

struct vpx_image;
struct vpx_codec_ctx;
struct VpxVideoWriterStruct;

namespace Tempest
{
struct VideoInfo
{
    std::string FileName;
    uint32_t    Width;
    uint32_t    Height;
    uint32_t    FPS = 30;
    uint32_t    Bitrate = 200;
};

class NVVideoEncoder
{
    std::fstream                 m_Stream;
    std::string                  m_Filename;
    Library                      m_Library;

    VideoInfo                    m_VideoInfo;

    struct _NV_ENCODE_API_FUNCTION_LIST* m_APIList = nullptr;
    void*                        m_Device = nullptr;
    void*                        m_Encoder = nullptr;

    struct
    {
        void*                    InputBuffer;
        void*                    BitstreamBuffer;
    }                            m_InputBuffers[INPUT_BUFFER_COUNT];

    uint32_t                     m_BufferInUse = 0;

    uint32_t                     m_FrameIndex = 0;
    uint32_t                     m_FrameNumber = 0;

public:
    NVVideoEncoder();
    ~NVVideoEncoder();
    
    bool openStream(VideoInfo video_info);
    void endStream();

    bool submitFrame(const Texture& tex);

private:
    void processOutput(uint32_t idx);
};

class VPXVideoEncoder
{
    std::string                  m_Filename;

    VideoInfo                    m_VideoInfo;

	struct vpx_image*			 m_VPXImageSurface = nullptr;
	vpx_codec_ctx_t				 m_Codec;
	struct VpxVideoWriterStruct* m_Writer = nullptr;

	std::unique_ptr<uint8_t[]>   m_YUVImage;

	uint32_t                     m_FrameNumber = 0;

public:
    VPXVideoEncoder();
    ~VPXVideoEncoder();
    
    bool openStream(VideoInfo video_info);
    void endStream();

    bool submitFrame(const Texture& tex);

private:
    void processOutput(uint32_t idx);
};
}

#endif // _TEMPEST_VIDEO_ENCODE_HH_
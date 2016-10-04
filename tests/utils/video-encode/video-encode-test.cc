#include "tempest/image/image.hh"
#include "tempest/utils/video-encode.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/testing.hh"

#include <memory>

#if 0
#	define VIDEO_FILENAME "videoenc.m4v"
#	define VIDEO_ENCODER Tempest::VideoEncoder
#else
#	define VIDEO_FILENAME "videoenc.ivf"
#	define VIDEO_ENCODER Tempest::VPXVideoEncoder
#endif

const uint32_t EncodeNumberOfSeconds = 20;

TGE_TEST("Testing video encoding capabilities")
{
    std::unique_ptr<Tempest::Texture> tex(Tempest::LoadImage(Tempest::Path(CURRENT_SOURCE_DIR "/Mandrill.tga")));

    auto& tex_hdr = tex->getHeader();

    Tempest::VideoInfo video_info;
    video_info.FileName = VIDEO_FILENAME;
    video_info.FPS = 30;
    video_info.Width = tex_hdr.Width;
    video_info.Height = tex_hdr.Height;

    VIDEO_ENCODER video_enc;
    auto status = video_enc.openStream(video_info);
    TGE_CHECK(status, "Failed to open video stream for encoding");

    for(uint32_t frame_idx = 0, frame_idx_end = EncodeNumberOfSeconds*video_info.FPS;
        frame_idx < frame_idx_end; ++frame_idx)
    {
        auto status = video_enc.submitFrame(*tex);
        TGE_CHECK(status, "Failed to submit new frame");
    }

    video_enc.endStream();
}
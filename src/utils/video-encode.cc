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

#include "tempest/utils/video-encode.hh"

#ifdef DISABLE_CUDA
#   include <cuda_runtime.h>
#   include "nvenc/nvEncodeAPI.h"
#   include "nvenc/NvHWEncoder.h"
#endif

#include "vpx/vpx_encoder.h"
#include "tools_common.h"
#include "video_writer.h"

#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024

#ifdef DISABLE_CUDA
/*
const NV_ENC_BUFFER_FORMAT BufferFormat = NV_ENC_BUFFER_FORMAT_YUV444_PL;
/*/
const NV_ENC_BUFFER_FORMAT BufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;
//*/
#endif

extern "C"
{
#define main not_main
#include "libvpx/tools_common.c"
#include "libvpx/video_writer.c"
#include "libvpx/ivfenc.c"
#include "libvpx/examples/set_maps.c"
}

namespace Tempest
{
#ifdef DISABLE_CUDA
typedef NVENCSTATUS (NVENCAPI *NvEncodeAPICreateInstanceProc)(NV_ENCODE_API_FUNCTION_LIST *functionList);

NVVideoEncoder::NVVideoEncoder() {}

NVVideoEncoder::~NVVideoEncoder()
{
    endStream();
    cuCtxDestroy(reinterpret_cast<CUcontext>(m_Device));
	m_Device = nullptr;
	delete m_APIList;
	m_APIList = nullptr;
}

bool NVVideoEncoder::openStream(VideoInfo video_info)
{
    m_VideoInfo = video_info;
    m_Stream.open(video_info.FileName.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);

    if(!m_Library.loaded())
    {
    #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        typedef HMODULE CUDADRIVER;
    #else
        typedef void *CUDADRIVER;
    #endif

        auto cu_res = cuInit(0);
        if(cu_res != CUDA_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to initialize CUDA context");
            return false;
        }

        int device_count = 0;
        cu_res = cuDeviceGetCount(&device_count);
        if(cu_res != CUDA_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to get CUDA device count");
            return false;
        }

        int deviceID = 0;
        if(deviceID >= device_count)
        {
            Log(LogLevel::Error, "Invalid CUDA capable device. Current available CUDA devices: ", device_count);
            return false;
        }

        CUdevice device;
        cu_res = cuDeviceGet(&device, deviceID);
        if(cu_res != CUDA_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to get CUDA capable device: Error Code ", cu_res);
            return false;
        }

        int major, minor;
        cu_res = cuDeviceComputeCapability(&major, &minor, device);
        if(cu_res != CUDA_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to get CUDA capabilities of device ", deviceID, ": Error Code ", cu_res);
            return false;
        }

        if (((major << 4) + minor) < 0x30)
        {
            Log(LogLevel::Error, "The specified GPU does not have NVENC capabilities ", deviceID);
            return NV_ENC_ERR_NO_ENCODE_DEVICE;
        }

        cu_res = cuCtxCreate(reinterpret_cast<CUcontext*>(&m_Device), 0, device);
        if(cu_res != CUDA_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to initialize CUDA context: Error Code ", cu_res);
            return false;
        }

         if(!m_Library.load("nvEncodeAPI64.dll"))
        {
            Log(LogLevel::Error, "Failed to load NVENC library");
            return false;
        }

        auto nvEncodeAPICreateInstance = (NvEncodeAPICreateInstanceProc)m_Library.getProcAddress("NvEncodeAPICreateInstance");
        if(nvEncodeAPICreateInstance == nullptr)
        {
            Log(LogLevel::Error, "Failed to load NVENC library: nvEncodeAPICreateInstance is missing");
            return false;
        }

        m_APIList = new NV_ENCODE_API_FUNCTION_LIST{};
        m_APIList->version = NV_ENCODE_API_FUNCTION_LIST_VER;
        auto status = nvEncodeAPICreateInstance(m_APIList);
        if(status != NV_ENC_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to load NVENC library: nvEncodeAPICreateInstance failed: Error Code ", status);
            return false;
        }
    }

    if(m_Encoder)
    {
        endStream();
    }

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS open_session_params{};
    open_session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    open_session_params.device = m_Device;
    open_session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    open_session_params.reserved = NULL;
    open_session_params.apiVersion = NVENCAPI_VERSION;

    auto status = m_APIList->nvEncOpenEncodeSessionEx(&open_session_params, &m_Encoder);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to open NVENC session: Error Code ", status);
        return false;
    }

    NV_ENC_PRESET_CONFIG preset_cfg{};
    preset_cfg.version = NV_ENC_PRESET_CONFIG_VER;
    preset_cfg.presetCfg.version = NV_ENC_PRESET_CONFIG_VER;
	/*preset_cfg.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
	preset_cfg.presetCfg.gopLength = -1;
	preset_cfg.presetCfg.rcParams.constQP.qpInterP = 28;
    preset_cfg.presetCfg.rcParams.constQP.qpInterB = 28;
    preset_cfg.presetCfg.rcParams.constQP.qpIntra = 28;
    */
    NV_ENC_INITIALIZE_PARAMS enc_params{};
    enc_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
    enc_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
    //enc_params.encodeGUID = NV_ENC_CODEC_HEVC_GUID;
    enc_params.presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
    enc_params.encodeWidth = video_info.Width;
    enc_params.encodeHeight = video_info.Height;
    enc_params.darWidth = video_info.Width;
    enc_params.darHeight = video_info.Height;
    enc_params.frameRateNum = video_info.FPS;
    enc_params.frameRateDen = 1;
    enc_params.enableEncodeAsync = 0; // TODO: Should I care?
    enc_params.enablePTD = 1;
    enc_params.reportSliceOffsets = 0;
    enc_params.enableSubFrameWrite = 0;
    enc_params.encodeConfig = &preset_cfg.presetCfg;
    enc_params.maxEncodeWidth = video_info.Width;
    enc_params.maxEncodeHeight = video_info.Height;

    status = m_APIList->nvEncGetEncodePresetConfig(m_Encoder, enc_params.encodeGUID, enc_params.presetGUID, &preset_cfg);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to get encode preset: Error Code ", status);
        return false;
    }

    status = m_APIList->nvEncInitializeEncoder(m_Encoder, &enc_params);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to initialize encoder: Error Code ", status);
        return false;
    }

    for(auto& input_buffer : m_InputBuffers)
    {
        NV_ENC_CREATE_INPUT_BUFFER input_buf_params{};
        input_buf_params.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
        input_buf_params.width = video_info.Width;
        input_buf_params.height = video_info.Height;
        input_buf_params.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
        input_buf_params.bufferFmt = BufferFormat;

        status = m_APIList->nvEncCreateInputBuffer(m_Encoder, &input_buf_params);
        if(status != NV_ENC_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to create input buffer: Error Code ", status);
        }

        input_buffer.InputBuffer = input_buf_params.inputBuffer;

        NV_ENC_CREATE_BITSTREAM_BUFFER bitstream_buf_params{};
        bitstream_buf_params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        bitstream_buf_params.size = BITSTREAM_BUFFER_SIZE;
        bitstream_buf_params.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
        status = m_APIList->nvEncCreateBitstreamBuffer(m_Encoder, &bitstream_buf_params);
        if(status != NV_ENC_SUCCESS)
        {
            Log(LogLevel::Error, "Failed to create bitstream buffer: Error Code ", status);
            return false;
        }

        input_buffer.BitstreamBuffer = bitstream_buf_params.bitstreamBuffer;
    }

    return true;
}

void NVVideoEncoder::endStream()
{
	if(m_Encoder == nullptr)
		return;

    NV_ENC_PIC_PARAMS enc_pic_params{};
    enc_pic_params.version = NV_ENC_PIC_PARAMS_VER;
    enc_pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    enc_pic_params.completionEvent = nullptr;
    auto status = m_APIList->nvEncEncodePicture(m_Encoder, &enc_pic_params);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to insert end-of-stream packet: Error Code ", status);
    }

    for(uint32_t proc_idx = m_FrameIndex; m_BufferInUse; proc_idx = (proc_idx + 1)%INPUT_BUFFER_COUNT)
    {
        uint32_t frame_mask = (1 << proc_idx);
        if(m_BufferInUse & frame_mask)
        {
            processOutput(proc_idx);
        }
    }

    for(auto& buffer_info : m_InputBuffers)
    {
        auto status = m_APIList->nvEncDestroyInputBuffer(m_Encoder, buffer_info.InputBuffer);
        TGE_ASSERT(status == NV_ENC_SUCCESS, "Failed to destroy input buffer");
        buffer_info.InputBuffer = nullptr;

        status = m_APIList->nvEncDestroyBitstreamBuffer(m_Encoder, buffer_info.BitstreamBuffer);
        TGE_ASSERT(status == NV_ENC_SUCCESS, "Failed to destroy bitstream");
        buffer_info.BitstreamBuffer = nullptr;
    }

    m_APIList->nvEncDestroyEncoder(m_Encoder);
	m_Encoder = nullptr;
}

void NVVideoEncoder::processOutput(uint32_t proc_idx)
{
    uint32_t frame_mask = (1 << proc_idx);

    NV_ENC_LOCK_BITSTREAM bitstream_data{};
    bitstream_data.version = NV_ENC_LOCK_BITSTREAM_VER;
    bitstream_data.outputBitstream = m_InputBuffers[proc_idx].BitstreamBuffer;
    bitstream_data.doNotWait = false;

    auto status = m_APIList->nvEncLockBitstream(m_Encoder, &bitstream_data);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to lock bitstream for reading encoded result: Error Code ", status);
        return;
    }

    m_Stream.write(reinterpret_cast<const char*>(bitstream_data.bitstreamBufferPtr), bitstream_data.bitstreamSizeInBytes);

    status = m_APIList->nvEncUnlockBitstream(m_Encoder, bitstream_data.outputBitstream);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to unlock bitstream after reading encoded data: Error Code ", status);
    }
    
    m_BufferInUse &= ~frame_mask;
}

bool NVVideoEncoder::submitFrame(const Texture& tex)
{
    NV_ENC_LOCK_INPUT_BUFFER lock_buf_params{};
    lock_buf_params.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
    auto& buf_info = m_InputBuffers[m_FrameIndex];
    auto input_buffer = lock_buf_params.inputBuffer = buf_info.InputBuffer;
    
    uint32_t frame_mask = (1 << m_FrameIndex);
    if(m_BufferInUse & frame_mask)
    {
        processOutput(m_FrameIndex);
    }

    auto status = m_APIList->nvEncLockInputBuffer(m_Encoder, &lock_buf_params);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to lock NVENC buffer: Error Code ", status);
        return false;
    }

    auto& hdr = tex.getHeader();
    if(hdr.Width != m_VideoInfo.Width ||
       hdr.Height != m_VideoInfo.Height)
        return false;

    uint32_t pitch = lock_buf_params.pitch;
    uint8_t* input_luma = reinterpret_cast<uint8_t*>(lock_buf_params.bufferDataPtr);
    uint32_t color_plane_offset = (m_VideoInfo.Height*pitch);

    if(BufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_PL)
    {
        uint8_t* input_cb = input_luma + color_plane_offset;
        uint8_t* input_cr = input_cb + color_plane_offset;

        for (uint32_t y = 0 ; y < m_VideoInfo.Height; ++y)
        {
            for(uint32_t x = 0; x < m_VideoInfo.Width; ++x)
            {
                auto color = tex.fetchRGB(x, y);
                auto ycbcr = RGBToYCbCr(color);
                input_luma[y*pitch + x] = static_cast<uint8_t>(ycbcr.Color.x*255.0f);
                input_cb[y*pitch + x] = static_cast<uint8_t>(ycbcr.Color.y*255.0f);
                input_cr[y*pitch + x] = static_cast<uint8_t>(ycbcr.Color.z*255.0f);
            }
        }
    }
    else
    {
        uint8_t* input_chroma = input_luma + color_plane_offset;
        for (uint32_t y = 0 ; y < m_VideoInfo.Height; ++y)
        {
            for(uint32_t x = 0; x < m_VideoInfo.Width; ++x)
            {
                auto color = tex.fetchRGB(x, y);
                auto ycbcr = RGBToYCbCr(color);
                input_luma[y*pitch + x] = static_cast<uint8_t>(ycbcr.Color.x*255.0f);
                if((x % 2) == 0 && (y % 2) == 0)
                {
                    input_chroma[y*pitch/2 + x] = static_cast<uint8_t>(ycbcr.Color.y*255.0f);
                    input_chroma[y*pitch/2 + x + 1] = static_cast<uint8_t>(ycbcr.Color.z*255.0f);
                }
            }
        }
    }

    status = m_APIList->nvEncUnlockInputBuffer(m_Encoder, lock_buf_params.inputBuffer);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to unlock NVENC buffer: Error Code ", status);
        return false;
    }

    NV_ENC_PIC_PARAMS pic_params{};
    pic_params.version = NV_ENC_PIC_PARAMS_VER;
    pic_params.inputBuffer = input_buffer;
    pic_params.inputWidth = m_VideoInfo.Width;
    pic_params.inputHeight = m_VideoInfo.Height;
	pic_params.bufferFmt = BufferFormat;
    pic_params.outputBitstream = buf_info.BitstreamBuffer;
    pic_params.completionEvent = nullptr;
    pic_params.inputTimeStamp = m_FrameNumber;
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic_params.qpDeltaMap = nullptr;
    pic_params.qpDeltaMapSize = 0;

    status = m_APIList->nvEncEncodePicture(m_Encoder, &pic_params);
    if(status != NV_ENC_SUCCESS)
    {
        Log(LogLevel::Error, "Failed to encode a frame: Error Code ", status);
        return false;
    }

    m_BufferInUse |= frame_mask; 
    m_FrameIndex = (m_FrameIndex + 1)%INPUT_BUFFER_COUNT;
    ++m_FrameNumber;
    return true;
}
#endif
VPXVideoEncoder::VPXVideoEncoder()
    :   m_Codec{}
{
}

VPXVideoEncoder::~VPXVideoEncoder()
{
	endStream();
}

bool VPXVideoEncoder::openStream(VideoInfo video_info)
{
	m_VideoInfo = video_info;

	if(m_VPXImageSurface)
	{
		endStream();
	}

	m_VPXImageSurface = vpx_img_alloc(nullptr, VPX_IMG_FMT_YV12, video_info.Width, video_info.Height, 1);
	if(m_VPXImageSurface == nullptr)
	{
		Log(LogLevel::Error, "Failed to allocate VPX image surface");
		return false;
	}

	const VpxInterface* vpx_enc = get_vpx_encoder_by_name("vp9");

	vpx_codec_enc_cfg_t cfg{};
	auto err = vpx_codec_enc_config_default(vpx_enc->codec_interface(), &cfg, 0);
	if(err)
	{
		Log(LogLevel::Error, "Failed to get codec config");
		return false;
	}

	VpxVideoInfo info{};
	info.codec_fourcc = vpx_enc->fourcc;
	info.frame_width = video_info.Width;
	info.frame_height = video_info.Height;
	info.time_base.numerator = 1;
	info.time_base.denominator = video_info.FPS;

	m_Writer = vpx_video_writer_open(video_info.FileName.c_str(), kContainerIVF, &info);
	if(m_Writer == nullptr)
	{
		Log(LogLevel::Error, "Failed to open VPX writer");
		return false;
	}

	cfg.g_w = video_info.Width;
	cfg.g_h = video_info.Height;
	cfg.g_timebase.num = 1;
	cfg.g_timebase.den = video_info.FPS;
	cfg.rc_target_bitrate = video_info.Bitrate;
	cfg.g_error_resilient = 0;

	m_YUVImage = std::unique_ptr<uint8_t[]>(new uint8_t[video_info.Width*video_info.Height*3/2]);

	err = vpx_codec_enc_init(&m_Codec, vpx_enc->codec_interface(), &cfg, 0);
	if(err)
	{
		Log(LogLevel::Error, "Failed to init VPX codec");
		return false;
	}

	return true;
}

void VPXVideoEncoder::endStream()
{
	if(!m_VPXImageSurface)
		return;
			
	if(m_Writer && m_VPXImageSurface && m_Codec.name)
	{
		bool active;
		do
		{
			active = false;
			vpx_codec_iter_t iter = nullptr;
			const vpx_codec_cx_pkt_t *pkt = nullptr;
			const vpx_codec_err_t res = vpx_codec_encode(&m_Codec, nullptr, -1, 1,
														 0, VPX_DL_GOOD_QUALITY);
			if(res != VPX_CODEC_OK)
			{
				Log(LogLevel::Error, "Failed to encode video frame in VPX format");
				break;
			}

			while ((pkt = vpx_codec_get_cx_data(&m_Codec, &iter)) != nullptr)
			{
				active = 1;

				if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
				{
					const int keyframe = (pkt->data.frame.flags & VPX_FRAME_IS_KEY) != 0;
					auto status = vpx_video_writer_write_frame(m_Writer,
															   reinterpret_cast<uint8_t*>(pkt->data.frame.buf),
															   pkt->data.frame.sz,
															   pkt->data.frame.pts);
					if(!status)
					{
						Log(LogLevel::Error, "Failed to write frame");
						active = 0;
						break;
					}
				}
			}
		} while(active);
	}

	if(m_VPXImageSurface)
	{
		vpx_img_free(m_VPXImageSurface);
		m_VPXImageSurface = nullptr;
	}

	if(m_Codec.name)
	{
		vpx_codec_destroy(&m_Codec);
		m_Codec = {};
	}

	if(m_Writer)
	{
		vpx_video_writer_close(m_Writer);
		m_Writer = nullptr;
	}
}

bool VPXVideoEncoder::submitFrame(const Texture& tex)
{
	auto& hdr = tex.getHeader();
	if(hdr.Width != m_VideoInfo.Width ||
	   hdr.Height != m_VideoInfo.Height)
	{
		Log(LogLevel::Error, "Encoded frame does not match backbuffer size");
		return false;
	}

	uint8_t* input_luma = m_YUVImage.get();
	uint8_t* input_cr = input_luma + m_VideoInfo.Width*m_VideoInfo.Height;
    uint8_t* input_cb = input_cr + m_VideoInfo.Width*m_VideoInfo.Height/4;

	uint32_t luma_pitch = m_VideoInfo.Width;
	uint32_t chroma_pitch = m_VideoInfo.Width/2;

    for (uint32_t y = 0 ; y < m_VideoInfo.Height; ++y)
    {
        uint32_t y_read = m_VideoInfo.Height - 1 - y;
        for(uint32_t x = 0; x < m_VideoInfo.Width; ++x)
        {
            auto color = tex.fetchRGB(x, y_read);
            auto ycbcr = RGBToYCbCr(color);
            input_luma[y*luma_pitch + x] = static_cast<uint8_t>(ycbcr.Color.x*255.0f);
            if((x % 2) == 0 && (y % 2) == 0)
            {
                input_cb[(y*chroma_pitch + x)/2] = static_cast<uint8_t>(ycbcr.Color.y*255.0f);
                input_cr[(y*chroma_pitch + x)/2] = static_cast<uint8_t>(ycbcr.Color.z*255.0f);
            }
        }
    }

	auto surf = vpx_img_wrap(m_VPXImageSurface, VPX_IMG_FMT_YV12, m_VideoInfo.Width, m_VideoInfo.Height, 1, const_cast<uint8_t*>(m_YUVImage.get()));
	TGE_ASSERT(surf == m_VPXImageSurface, "Unexpected library behavior");

	int got_pkts = 0;
	vpx_codec_iter_t iter = nullptr;
	const vpx_codec_cx_pkt_t *pkt = nullptr;
	const vpx_codec_err_t res = vpx_codec_encode(&m_Codec, m_VPXImageSurface, m_FrameNumber, 1,
												 0, VPX_DL_GOOD_QUALITY);
	if(res != VPX_CODEC_OK)
	{
		Log(LogLevel::Error, "Failed to encode video frame in VPX format");
		return false;
	}

	while ((pkt = vpx_codec_get_cx_data(&m_Codec, &iter)) != nullptr)
	{
		got_pkts = 1;

		if(pkt->kind == VPX_CODEC_CX_FRAME_PKT)
		{
			const int keyframe = (pkt->data.frame.flags & VPX_FRAME_IS_KEY) != 0;
			auto status = vpx_video_writer_write_frame(m_Writer,
													   reinterpret_cast<uint8_t*>(pkt->data.frame.buf),
													   pkt->data.frame.sz,
													   pkt->data.frame.pts);
			if(!status)
			{
				Log(LogLevel::Error, "Failed to write frame");
				return false;
			}
		}
	}

	m_FrameNumber++;

	return true;
}
}

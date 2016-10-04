/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#ifndef _TEMPEST_RAY_TRACING_CUDA_SYSTEM_HH_
#define _TEMPEST_RAY_TRACING_CUDA_SYSTEM_HH_

#include "tempest/utils/config.hh"

#ifndef DISABLE_CUDA

#include "tempest/compute/ray-tracing-cuda.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "tempest/debug/fps-counter.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "tempest/graphics/state-object.hh"

namespace Tempest
{
// This basically wraps both the ray tracer and the windowing functionality
typedef decltype(Tempest::PreferredSystem().Backend) PreferredBackend;

class RayTracingCudaSystem
{
    static const uint32_t TextureTableSlots = 5;
    static const uint32_t AuxiliarySlots = 1;
    static const uint32_t TotalSlots = TextureTableSlots + AuxiliarySlots;

    PreferredWindow*                            m_Window;
    PreferredBackend*                           m_Backend;
    PreferredShaderCompiler*                    m_ShaderCompiler;
    std::unique_ptr<PreferredSystem>            m_SystemCleanup;
    std::unique_ptr<RayTracerCuda>              m_RayTracer;
    PreferredBackend::TextureType              *m_Texture;
    Tempest::CUDASurfaceResource                m_CudaTexture;
    PreferredShaderCompiler::ShaderProgramType* m_Shader;
    PreferredBackend::StateObjectType*          m_DSState;
    PreferredBackend::CommandBufferType*        m_CommandBuffer;
    PreferredBackend::CommandBufferType*        m_FpsCommandBuffer;
	std::unique_ptr<Texture>					m_BackbufferCopy;
    std::unique_ptr<FpsCounter>                 m_FpsCounter;
    BakedResourceTable                          m_TextureSlots;
    TextureDescription                          m_FrameTexture;
	bool										m_EnableFpsCounter = false;
    uint32_t                                    m_Flags = DIRTY_RECT|UNINITIALIZED_STATE;
public:
    RayTracingCudaSystem(uint32_t width, uint32_t height, const Matrix4& view_proj_inv, const RTSettings& settings = RTSettings(),
                         PreferredWindow* window = nullptr, PreferredBackend* backend = nullptr, PreferredShaderCompiler* shader_compiler = nullptr);
    ~RayTracingCudaSystem();

    Tempest::RayTracerCuda* getRayTracer() { return m_RayTracer.get(); }

    void startRendering()
    {
        m_RayTracer->drawOnce();
    }

    PreferredWindow& getWindow()
    {
        return *m_Window;
    }

    PreferredBackend& getBackend()
    {
        return *m_Backend;
    }

    PreferredShaderCompiler& getShaderCompiler()
    {
        return *m_ShaderCompiler;
    }

    void completeFrame()
    {
    }

    void completeFrameAndRestart(uint32_t width, uint32_t height, const Matrix4& view_proj_inv);

    void saveImage(const Path& file_path);

    void displayImage()
    {
        auto texture = m_Texture;
        auto cuda_texture = m_CudaTexture;

        m_Window->show();

        memset(m_TextureSlots.get(), 0, TotalSlots*4*sizeof(float));
        *reinterpret_cast<uint64_t*>(m_TextureSlots.get() + TextureTableSlots*4*sizeof(float)) = texture->getHandle();

        m_Backend->setActiveTextures(TotalSlots);

        m_Backend->setViewportRect(0, 0, cuda_texture.Texture.Description.Width, cuda_texture.Texture.Description.Height);

        while(!m_Window->isDead())
        {
            m_Backend->clearColorBuffer(0, Tempest::Vector4{1.0f, 0.0f, 0.0f, 0.0f});
            m_Backend->clearDepthStencilBuffer();

            m_Backend->setTextures(&m_TextureSlots);

            m_Backend->submitCommandBuffer(m_CommandBuffer);
        
            m_Window->swapBuffers();
        }
    }

    bool submitFrame()
	{
        auto texture = m_Texture;
        auto& backend = *m_Backend;

        if(m_Window->isDead())
            return false;

        if(m_Flags & UNINITIALIZED_STATE)
        {
            m_Window->show();
            m_Flags &= ~UNINITIALIZED_STATE;
        }

        if(m_Flags & DIRTY_RECT)
        {
            auto& window = *m_Window;
            window.resize(m_FrameTexture.Width, m_FrameTexture.Height);

			m_Flags &= ~DIRTY_RECT;
        }

        backend.setViewportRect(0, 0, m_FrameTexture.Width, m_FrameTexture.Height);

        memset(m_TextureSlots.get(), 0, TotalSlots*4*sizeof(float));
        *reinterpret_cast<uint64_t*>(m_TextureSlots.get() + TextureTableSlots*4*sizeof(float)) = texture->getHandle();

        backend.clearColorBuffer(0, Tempest::Vector4{1.0f, 0.0f, 0.0f, 0.0f});
        backend.clearDepthStencilBuffer();

        backend.setTextures(&m_TextureSlots);

        backend.submitCommandBuffer(m_CommandBuffer);
        
        if(m_FpsCounter->update(m_FrameTexture.Width, m_FrameTexture.Height))
        {
            m_FpsCommandBuffer->clear();
            auto draw_batches = m_FpsCounter->getDrawBatches();
            auto batch_count = m_FpsCounter->getDrawBatchCount();
            for(decltype(batch_count) i = 0; i < batch_count; ++i)
            {
                m_FpsCommandBuffer->enqueueBatch(draw_batches[i]);
            }

            m_FpsCommandBuffer->prepareCommandBuffer();
        }

		if(m_EnableFpsCounter)
		{
			backend.submitCommandBuffer(m_FpsCommandBuffer);
		}

		return true;
	}

    bool presentFrame()
    {
        if(!submitFrame())
			return false;

        m_Window->swapBuffers();

        return true;
    }

	void setEnableFpsCounter(bool fps_counter) { m_EnableFpsCounter = fps_counter; }
    void toggleFpsCounter() { m_EnableFpsCounter = !m_EnableFpsCounter; }
    bool getFpsCounterStatus() { return m_EnableFpsCounter; }

	Texture* getLastFrameTexture();

private:
	void updateBackbufferCopy()
	{
		if(!m_BackbufferCopy)
		{
			m_BackbufferCopy = std::unique_ptr<Texture>(new Texture(m_FrameTexture, new uint8_t[m_FrameTexture.Width*m_FrameTexture.Height*DataFormatElementSize(m_FrameTexture.Format)]));
		}
		else
		{
			auto& hdr = m_BackbufferCopy->getHeader();
			if(hdr.Width != m_FrameTexture.Width ||
			   hdr.Height != m_FrameTexture.Height)
			{
				m_BackbufferCopy->realloc(m_FrameTexture);
			}
		}
	}
};
}

#endif

#endif // _TEMPEST_RAY_TRACING_CUDA_SYSTEM_HH_
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

#ifndef _SCENE_RASTERIZER_SYSTEM_HH_
#define _SCENE_RASTERIZER_SYSTEM_HH_

#include "tempest/graphics/scene-rasterizer.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "tempest/debug/fps-counter.hh"

namespace Tempest
{
class Path;
    
class SceneRasterizerSystem
{
    static const uint32_t TextureTableSlots = 5;
    static const uint32_t AuxiliarySlots = 1;
    static const uint32_t TotalSlots = TextureTableSlots + AuxiliarySlots;

    typedef Tempest::PreferredBackend::IOCommandBufferType IOCommandBufferType;
    typedef Tempest::PreferredBackend::CommandBufferType CommandBufferType;
    typedef Tempest::PreferredBackend::StorageType StorageType;
    typedef Tempest::PreferredBackend::TextureType TextureType;
    typedef Tempest::PreferredBackend::StateObjectType StateObjectType;
    typedef Tempest::PreferredShaderCompiler::ShaderProgramType ShaderType;

    PreferredWindow*                    m_Window;
    PreferredBackend*                   m_Backend;
    PreferredShaderCompiler*            m_ShaderCompiler;

    ShaderType*                         m_Shader = nullptr;
    StateObjectType*                    m_DSState = nullptr;
    CommandBufferType*                  m_CommandBuffer = nullptr;

    Tempest::TextureDescription         m_FrameTexture;
    std::unique_ptr<PreferredSystem>    m_SystemCleanup;
    std::unique_ptr<FpsCounter>         m_FpsCounter;
    PreferredBackend::CommandBufferType* m_FpsCommandBuffer;
    bool								m_EnableFpsCounter = false;

	uint32_t                            m_Flags = DIRTY_RECT;

    std::unique_ptr<SceneRasterizer>    m_Rasterizer;
    BakedResourceTable                  m_TextureSlots;

    uint32_t                            m_Width = 0,
                                        m_Height = 0;
    Matrix4                             m_ViewProjectionInverse;

public:

    SceneRasterizerSystem(uint32_t image_width, uint32_t image_height, const Matrix4& view_proj_inv,
                          PreferredWindow* window = nullptr, PreferredBackend* backend = nullptr, PreferredShaderCompiler* shader_compiler = nullptr);
    ~SceneRasterizerSystem();

    Tempest::SceneRasterizer* getRayTracer() { return m_Rasterizer.get(); }

    void startRendering()
    {
        m_Rasterizer->initWorkers();
        m_Rasterizer->drawOnce();
    }
    
    void completeFrame() {}

    void completeFrameAndRestart(uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
    {
        m_FrameTexture.Width = width;
        m_FrameTexture.Height = height;
        m_Width = width;
        m_Height = height;
        m_ViewProjectionInverse = view_proj_inv;
    }

    void saveImage(const Path& file_path);

    void displayImage()
    {
        auto texture = m_Rasterizer->getBackbuffer();

        m_Window->show();

        memset(m_TextureSlots.get(), 0, TotalSlots*4*sizeof(float));
        *reinterpret_cast<uint64_t*>(m_TextureSlots.get() + TextureTableSlots*4*sizeof(float)) = texture->getHandle();

        m_Backend->setActiveTextures(TotalSlots);

        m_Backend->setViewportRect(0, 0, m_Window->getWidth(), m_Window->getHeight());

        while(!m_Window->isDead())
        {
            m_Backend->clearColorBuffer(0, Tempest::Vector4{0.0f, 0.0f, 0.0f, 0.0f});
            m_Backend->clearDepthStencilBuffer();
            m_Backend->setTextures(&m_TextureSlots);

            m_Backend->submitCommandBuffer(m_CommandBuffer);
        
            m_Window->swapBuffers();
        }
    }

    PreferredWindow& getWindow()
    {
        //updateSystem();
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

    const Texture* getLastFrameTexture() const;

	bool submitFrame()
	{
		auto texture = m_Rasterizer->getBackbuffer();
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

        if(m_Width && m_Height)
            m_Rasterizer->draw(m_Width, m_Height, m_ViewProjectionInverse);

        m_Width = m_Height = 0;

        return true;
    }

    void setEnableFpsCounter(bool fps_counter) { m_EnableFpsCounter = fps_counter; }
    void toggleFpsCounter() { m_EnableFpsCounter = !m_EnableFpsCounter; }
    bool getFpsCounterStatus() { return m_EnableFpsCounter; }
};
}

#endif // _SCENE_RASTERIZER_SYSTEM_HH_
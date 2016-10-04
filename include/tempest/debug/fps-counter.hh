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

#ifndef _TEMPEST_FPS_COUNTER_HH_
#define _TEMPEST_FPS_COUNTER_HH_

#include "tempest/graphics/preferred-backend.hh"
#include "tempest/graphics/shader.hh"

#include "tempest/utils/timer.hh"

namespace Tempest
{
class FpsCounter
{
    PreferredBackend*                     m_Backend = nullptr;
    PreferredShaderCompiler*              m_ShaderCompiler = nullptr;

    PreferredBackend::BufferType*         m_VertexBuffer = nullptr;
    PreferredBackend::BufferType*         m_IndexBuffer = nullptr;
    PreferredBackend::ShaderProgramType*  m_Shader = nullptr;
    PreferredBackend::StateObjectType*    m_StateObject = nullptr;
    BakedResourceTable*                   m_Parameters[8];

    typedef PreferredBackend::ShaderProgramType::ResourceTableType ResourceTableType;
    ResourceTableType*                    m_ResourceTable = nullptr;

    typedef PreferredBackend::CommandBufferType::DrawBatchType DrawBatchType;
    DrawBatchType                 m_DrawBatches[8];
    uint32_t                      m_DrawBatchCount = 0,
                                  m_FrameCount = 0;
    uint64_t                      m_LastUpdate = 0;

    int64_t                       m_FrameRate = 0,
                                  m_FrameTime = 0;

    float                         m_WindowWidth = 0.0f,
                                  m_WindowHeight = 0.0f,
                                  m_HeightPixels = 0.0f;

    Tempest::TimeQuery            m_TimeQuery;
public:
    FpsCounter(PreferredBackend* backend,
               PreferredShaderCompiler* shader_compiler,
               FileLoader* file_loader,
               float height_pixels,
               float width,
               float height);
    ~FpsCounter();

    const DrawBatchType* getDrawBatches() const { return m_DrawBatches; }
    uint32_t getDrawBatchCount() const { return m_DrawBatchCount; }

    bool update(float window_width, float window_height);
};
}

#endif // _TEMPEST_FPS_COUNTER_HH_
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

#include "tempest/debug/fps-counter.hh"
#include "tempest/utils/memory.hh"

#include <algorithm>

namespace Tempest
{
struct FpsVertex
{
    uint16_t position[2];
};

static const FpsVertex FpsVerts[] =
{
    // TOP_SEGMENT
    { 3, 0 },
    { 2, 1 },
    { 3, 2 },
    { 9, 0 },
    { 9, 2 },
    { 10, 1 },

    // TOP_LEFT_SEGMENT
    { 0, 3 },
    { 2, 3 },
    { 1, 2 },
    { 0, 9 },
    { 2, 9 },
    { 1, 10 },

    // TOP_RIGHT_SEGMENT
    { 10, 3 },
    { 12, 3 },
    { 11, 2 },
    { 10, 9 },
    { 12, 9 },
    { 11, 10 },

    // CENTER_SEGMENT
    { 3, 11 },
    { 2, 12 },
    { 3, 13 },
    { 9, 11 },
    { 9, 13 },
    { 10, 12 },

    // BOTTOM_LEFT_SEGMENT
    { 0, 14 },
    { 2, 14 },
    { 1, 13 },
    { 0, 21 },
    { 2, 21 },
    { 1, 22 },

    // BOTTOM_RIGHT_SEGMENT
    { 10, 14 },
    { 12, 14 },
    { 11, 13 },
    { 10, 21 },
    { 12, 21 },
    { 11, 22 },

    // BOTTOM_SEGMENT
    { 3, 23 },
    { 2, 24 },
    { 3, 25 },
    { 9, 23 },
    { 9, 25 },
    { 10, 24 },
};

const size_t MaxSizeX = 12;
const size_t MaxSizeY = 25;

#define TOP_SEGMENT \
    0, 1, 2, \
    0, 2, 3, \
    3, 2, 4, \
    3, 4, 5

#define TOP_LEFT_SEGMENT \
    6, 7, 8, \
    6, 9, 7, \
    7, 9, 10, \
    9, 11, 10

#define TOP_RIGHT_SEGMENT \
    12, 13, 14, \
    12, 15, 13, \
    13, 15, 16, \
    15, 17, 16

#define CENTER_SEGMENT \
    18, 19, 20, \
    18, 20, 21, \
    21, 20, 22, \
    21, 22, 23

#define BOTTOM_LEFT_SEGMENT \
    24, 25, 26, \
    24, 27, 25, \
    25, 27, 28, \
    27, 29, 28

#define BOTTOM_RIGHT_SEGMENT \
    30, 31, 32, \
    30, 33, 31, \
    31, 33, 34, \
    33, 35, 34

#define BOTTOM_SEGMENT \
    36, 37, 38, \
    36, 38, 39, \
    39, 38, 40, \
    39, 40, 41

static const uint16_t DigitZero[] =
{
    TOP_SEGMENT,
    TOP_LEFT_SEGMENT,
    TOP_RIGHT_SEGMENT,
    BOTTOM_LEFT_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitOne[] =
{
    TOP_RIGHT_SEGMENT,
    BOTTOM_RIGHT_SEGMENT
};

static const uint16_t DigitTwo[] =
{
    TOP_SEGMENT,
    TOP_RIGHT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_LEFT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitThree[] =
{
    TOP_SEGMENT,
    TOP_RIGHT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitFour[] =
{
    TOP_LEFT_SEGMENT,
    TOP_RIGHT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_RIGHT_SEGMENT
};

static const uint16_t DigitFive[] =
{
    TOP_SEGMENT,
    TOP_LEFT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitSix[] =
{
    TOP_SEGMENT,
    TOP_LEFT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_LEFT_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitSeven[] =
{
    TOP_SEGMENT,
    TOP_RIGHT_SEGMENT,
    BOTTOM_RIGHT_SEGMENT
};

static const uint16_t DigitEight[] =
{
    TOP_SEGMENT,
    TOP_LEFT_SEGMENT,
    TOP_RIGHT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_LEFT_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

static const uint16_t DigitNine[] =
{
    TOP_SEGMENT,
    TOP_LEFT_SEGMENT,
    TOP_RIGHT_SEGMENT,
    CENTER_SEGMENT,
    BOTTOM_RIGHT_SEGMENT,
    BOTTOM_SEGMENT
};

struct FpsDigitDescription
{
    const uint16_t* Indices;
    uint32_t        IndicesCount;
    uint32_t        Offset;
};

#define FPS_DIGIT(name) FpsDigitDescription{ name, sizeof(name)/sizeof(uint16_t), 0 }

static FpsDigitDescription FpsDigits[] =
{
    FPS_DIGIT(DigitZero),
    FPS_DIGIT(DigitOne),
    FPS_DIGIT(DigitTwo),
    FPS_DIGIT(DigitThree),
    FPS_DIGIT(DigitFour),
    FPS_DIGIT(DigitFive),
    FPS_DIGIT(DigitSix),
    FPS_DIGIT(DigitSeven),
    FPS_DIGIT(DigitEight),
    FPS_DIGIT(DigitNine)
};

const float SafeArea = 25.0f;

FpsCounter::FpsCounter(PreferredBackend* backend, PreferredShaderCompiler* shader_compiler, FileLoader* file_loader, float height_pixels, float window_width, float window_height)
    :   m_Backend(backend),
        m_ShaderCompiler(shader_compiler),
        m_VertexBuffer(backend->createBuffer(sizeof(FpsVerts), Tempest::ResourceBufferType::VertexBuffer, 0, &FpsVerts)),
        m_WindowWidth(window_width),
        m_WindowHeight(window_height),
        m_HeightPixels(height_pixels)
{
    m_Shader = shader_compiler->compileShaderProgram("fps.tfx", file_loader);
    TGE_ASSERT(m_Shader, "Expecting successful compilation");
    typedef PreferredBackend::ShaderProgramType::ResourceTableType ResourceTableType;
    m_ResourceTable = m_Shader->createResourceTable("Globals", 1);
    TGE_ASSERT(m_ResourceTable, "Expecting valid resource table");

	float digit_idx = 1.0f;
    for(auto& params : m_Parameters)
    {
        float max_size_y = (float)std::numeric_limits<uint16_t>::max()/MaxSizeY;

        float text_height = m_HeightPixels*max_size_y/window_height;
        float text_width = text_height*window_height/window_width;

		float offset = m_HeightPixels*MaxSizeX/(MaxSizeY*window_width);

		Tempest::Vector4 transform{1.0f - 2.0f*digit_idx*(offset + SafeArea/window_width), 1.0f - 2.0f*SafeArea/window_height, 2.0f*text_width, -2.0f*text_height};
		++digit_idx;
        m_ResourceTable->setResource("Globals[0].Transform", transform);
        params = m_ResourceTable->extractBakedTable();
        m_ResourceTable->resetBakedTable();
    }
    
    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    
    m_StateObject = backend->createStateObject(&rt_fmt, 1, Tempest::DataFormat::Unknown, m_Shader, Tempest::DrawModes::TriangleList);

	uint32_t indices_count = 0;
    for(auto& digit_desc : FpsDigits)
    {
        // TODO: Compile time generation
        digit_desc.Offset = indices_count;
        indices_count += digit_desc.IndicesCount;
    }

    uint32_t indices_size = indices_count*sizeof(uint16_t);
    auto* fps_digits = TGE_TYPED_ALLOCA(uint16_t, indices_size);

    auto* digit_iter = fps_digits;
    for(auto& digit_desc : FpsDigits)
    {
        std::copy(digit_desc.Indices, digit_desc.Indices + digit_desc.IndicesCount, digit_iter);
        digit_iter += digit_desc.IndicesCount;
    }
    m_IndexBuffer = backend->createBuffer(indices_size, Tempest::ResourceBufferType::IndexBuffer, 0, fps_digits);

    size_t idx = 0;
    for(auto& draw_batch : m_DrawBatches)
    {
        draw_batch.BaseIndex = 0;
        draw_batch.BaseVertex = 0;
        draw_batch.IndexBuffer = m_IndexBuffer;
        draw_batch.VertexBuffers[0].Offset = 0;
        draw_batch.VertexBuffers[0].Stride = sizeof(FpsVertex);
        draw_batch.VertexBuffers[0].VertexBuffer = m_VertexBuffer;
        draw_batch.VertexCount = 0;
        draw_batch.SortKey = ~0;
        draw_batch.ResourceTable = m_Parameters[idx];
        draw_batch.PipelineState = m_StateObject;
    }
}

FpsCounter::~FpsCounter()
{
    m_Backend->destroyRenderResource(m_VertexBuffer);
    m_Backend->destroyRenderResource(m_IndexBuffer);
    m_Backend->destroyRenderResource(m_StateObject);
    m_ShaderCompiler->destroyRenderResource(m_Shader);

    for(auto& baked_res_table : m_Parameters)
    {
        delete baked_res_table;
    }

    delete m_ResourceTable;
}

const uint64_t UpdateTime = 2000000ULL;

bool FpsCounter::update(float window_width, float window_height)
{
    ++m_FrameCount;
    int64_t cur_time = m_TimeQuery.time();
    int64_t diff = cur_time - m_LastUpdate;

    bool update_cmd = false;

    if(m_WindowWidth != window_width ||
       m_WindowHeight != window_height)
    {
        size_t counter = 0;
        const size_t max_chars = 4;
		float digit_idx = 0.0f;
        for(auto& params : m_Parameters)
        {
            float max_size_y = (float)std::numeric_limits<uint16_t>::max()/MaxSizeY;

			float text_height = m_HeightPixels*max_size_y/window_height;
			float text_width = text_height*window_height/window_width;

			float offset = m_HeightPixels*MaxSizeX/(MaxSizeY*window_width);

			Tempest::Vector4 transform{1.0f - 2.0f*digit_idx*(offset + SafeArea/window_width), 1.0f - 2.0f*SafeArea/window_height, 2.0f*text_width, -2.0f*text_height};
			++digit_idx;

            m_ResourceTable->swapBakedTable(*params);
                m_ResourceTable->setResource("Globals[0].Transform", transform);
            m_ResourceTable->swapBakedTable(*params);
        }

        update_cmd = true;

        m_WindowWidth = window_width;
        m_WindowHeight = window_height;
    }

    if(diff > UpdateTime)
    {
        m_FrameTime = diff / m_FrameCount;
        m_FrameRate = std::min(9999ULL, (m_FrameCount * 1000000ULL) / UpdateTime);
        m_FrameCount = 0;
        m_LastUpdate = cur_time;

        auto& draw_batch = m_DrawBatches[0];
        draw_batch.VertexCount = FpsDigits[0].IndicesCount;
        m_DrawBatchCount = 1;

        uint64_t number[4];

        number[0] = (m_FrameRate % 10000) / 1000;
        number[1] = (m_FrameRate % 1000) / 100;
        number[2] = (m_FrameRate % 100) / 10;
        number[3] = m_FrameRate % 10;

        uint32_t non_empty = 0;
		for(; TGE_FIXED_ARRAY_SIZE(number) - 1; ++non_empty)
        {
            if(number[non_empty])
                break;
        }

        for(uint32_t i = 0; i < TGE_FIXED_ARRAY_SIZE(number); ++i)
        {
            auto& draw_batch = m_DrawBatches[i];
            uint64_t num = number[TGE_FIXED_ARRAY_SIZE(number) - 1 - i];
            auto& fps_digit = FpsDigits[num];
            draw_batch.BaseIndex = fps_digit.Offset;
            draw_batch.VertexCount = fps_digit.IndicesCount;
            draw_batch.ResourceTable = m_Parameters[i];
        }
                
        m_DrawBatchCount = TGE_FIXED_ARRAY_SIZE(number) - non_empty;
        update_cmd = true;
    }
    return update_cmd;
}
}
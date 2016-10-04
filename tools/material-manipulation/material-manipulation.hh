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

#ifndef _MATERIAL_MANIPULATION_HH_
#define _MATERIAL_MANIPULATION_HH_

#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/utils/video-encode.hh"

#define ENABLE_DATA_DRIVEN

#ifndef ENABLE_DATA_DRIVEN
//# define ENABLE_PHYSICS
#endif

#define ENABLE_MESH

#define CUDA_ACCELERATED

#ifdef CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#   define RASTERIZER_FUNCTION __device__
#   define RASTERIZER Tempest::RasterizerCuda
#ifdef __CUDA_ARCH__
#	define CONSTANT __constant__
#else
#   define CONSTANT const
#endif

#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#   define RASTERIZER_FUNCTION
#   define RASTERIZER Tempest::Rasterizer
#	define CONSTANT
#endif

enum RecordStatus: uint32_t
{
    Idle,
    Record,
    Playback
};

struct EventRecord
{
    int64_t                    Timestamp;
    Tempest::WindowSystemEvent Event;
};

struct SpringData
{
    Tempest::Vector2 Stretch;
#ifdef ENABLE_PHYSICS
    Tempest::Vector2 Velocity;
#endif
};

struct ClothInteractionShader
{
	Tempest::Vector2 Cursor;
    Tempest::Vector2 ForceDirection;
    float            ForceMagnitude;
    SpringData*      Data;

#if !defined(CUDA_ACCELERATED) || defined(__CUDACC__)
	RASTERIZER_FUNCTION void operator()(uint32_t x, uint32_t y, uint32_t width, uint32_t height, const Tempest::Vector2& dist_vector, float tangent_ratio);
#endif
};

struct ClothDataDrivenInteractionConversion
{
    bool				Paint = false;
    uint32_t			Width;
    uint32_t            Height;
	SpringData*			Data;

    float               RepeatFactor = 16.0f;

    float               DiffuseMultiplier = 1.0f;
    float               SpecularMultiplier = 1.0f;
    float               AnisotropyModifier = 0.5f;
    float               SmoothnessModifier = 0.5f;

    Tempest::Vector4    RemapDiffuseColor = Tempest::Vector4{ 1.0f, 1.0f, 1.0f, 1.0f },
                        RemapSpecularColor = Tempest::Vector4{ 1.0f, 1.0f, 1.0f, 1.0f };

    const void*         DiffuseTextures[8];
    const void*         SpecularTextures[8];
    const void*         StandardDeviationTextures[8];
    const void*         BasisTextures[8];

    void*               MixDiffuseTextureData;
    void*               MixSpecularTextureData;
    void*               MixStandardDeviationTextureData;
  	void*				MixTangentTextureData;

#if !defined(CUDA_ACCELERATED) || defined(__CUDACC__)
    RASTERIZER_FUNCTION void operator()(uint32_t worker_id, uint32_t x, uint32_t y);
#endif
};

struct ClothInteractionConversion
{
    bool				Paint = false;
    uint32_t			Width;
    uint32_t            Height;
	SpringData*			Data;
  	void*				MixTangentTextureData;

#if !defined(CUDA_ACCELERATED) || defined(__CUDACC__)
    RASTERIZER_FUNCTION void operator()(uint32_t worker_id, uint32_t x, uint32_t y);
#endif
};

enum
{
    MOVEMENT_FORWARD            = 1 << 0,
    MOVEMENT_LEFT               = 1 << 1,
    MOVEMENT_RIGHT              = 1 << 2,
    MOVEMENT_BACKWARD           = 1 << 3,
    MOVEMENT_ROTATION           = 1 << 4,
    MOVEMENT_UP                 = 1 << 5,
    MOVEMENT_DOWN               = 1 << 6,
	MOVEMENT_ZOOM_IN            = 1 << 7,
	MOVEMENT_ZOOM_OUT           = 1 << 8,
	MOVEMENT_FLUSH              = 1 << 9,
	MOVEMENT_DRAG               = 1 << 10,
    MOVEMENT_MASK               = (1 << 11) - 1,
    STATUS_ACTIVE               = 1 << 11,
    STATUS_MASK                 = (STATUS_ACTIVE << 1) - 1,

    OPTION_PLAYBACK_AND_RECORD  = 1 << 12,
    OPTION_ROTATION             = 1 << 13,
    OPTION_DISABLE_RECORD       = 1 << 14,
    
    ENQUEUE_SCREENSHOT          = 1 << 15,
};

enum
{
    MATERIAL_MANIPULATION_AREA_LIGHT = 1 << 0,
    MATERIAL_MANIPULATION_LOW_SPEC = 1 << 1,
    MATERIAL_MANIPULATION_GUI_RECORD = 1 << 2
};

class MaterialManipulation
{
    bool                                        m_PlaybackAndRecord = false;
    uint64_t                                    m_PreviousTime = 0;
    uint64_t                                    m_Status = 0;
    Tempest::TimeQuery                          m_Timer;

    Tempest::Matrix4                            m_ViewProjectionInverse;
    std::unique_ptr<RAY_TRACING_SYSTEM>         m_RayTracingSystem;

    RecordStatus                                m_RecordStatus = RecordStatus::Idle;

    EventRecord                                 m_EventRecord;

    Tempest::Vector3                            m_Base = Tempest::Vector3{0.0f, 0.0f, 0.0f};
    Tempest::Vector3                            m_Offset = Tempest::Vector3{0.0f, 0.0f, 5.0f};
    float                                       m_Yaw = 0.0f;
	float                                       m_Roll = (30.0f * Tempest::MathPi) / 180.0f;

    #ifdef ENABLE_DATA_DRIVEN
        ClothDataDrivenInteractionConversion    m_Simulation;
    #else
    #ifdef ENABLE_PHYSICS
	    ClothPhysicsSimulation                  m_Simulation;
    #else
        ClothInteractionConversion              m_Simulation;
    #endif
    #endif

    ClothInteractionShader                      m_Interaction;

    std::fstream                                m_RecordStream;

    Tempest::Vector2                            m_PreviousMousePosition,
                                                m_ImageSize;

    uint64_t                                    m_PlaneID,
                                                m_RecordStart = 0,
                                                m_TestRotationPeriod = 0;

    std::unique_ptr<Tempest::RTSGGXSurface>     m_Material;
    std::unique_ptr<SpringData[]>               m_SpringGrid;

    typedef Tempest::PreferredBackend::ShaderProgramType* ShaderProgramTypePtr;
    typedef Tempest::PreferredBackend::FramebufferType* FramebufferPtr;
    typedef Tempest::PreferredBackend::StorageType* StoragePtr;
    typedef Tempest::PreferredBackend::RenderTargetType* RenderTargetPtr;
    typedef Tempest::PreferredBackend::TextureType* GPUTexturePtr;
    typedef std::unique_ptr<Tempest::PreferredBackend::ShaderProgramType::ResourceTableType> ResourceTablePtr; 
    typedef Tempest::PreferredBackend::CommandBufferType* CommandBufferPtr;
    typedef Tempest::PreferredBackend::IOCommandBufferType* IOCommandBufferPtr;
    typedef std::unique_ptr<Tempest::BakedResourceTable> BakedResourcePtr;
    typedef Tempest::PreferredBackend::StateObjectType* StateObjectPtr;
    
    Tempest::ResourceIndex                      m_TransformIndex;

    Tempest::Vector4                            m_BaseDiffuseColor;
    Tempest::Vector4                            m_BaseSpecularColor;
    
    std::unique_ptr<Tempest::VPXVideoEncoder>   m_VideoEncoder;

    StateObjectPtr                              m_ImageStateObject;
    ShaderProgramTypePtr                        m_ImageShader;
    BakedResourcePtr                            m_TouchBakedTable;
    BakedResourcePtr                            m_RotateBakedTable;
    ResourceTablePtr                            m_ImageResourceTable;
    RenderTargetPtr                             m_IntermediateRenderTarget;
    FramebufferPtr                              m_IntermediateFramebuffer;
    GPUTexturePtr                               m_TouchImage,
                                                m_RotateImage;
    Tempest::TexturePtr                         m_BasisMap,
                                                m_StandardDeviationMap,
                                                m_DiffuseMap,
                                                m_SpecularMap,
                                                m_BackbufferTexture;
    SpringData*                                 m_SpringBuffer;
    StoragePtr                                  m_BackbufferStorage;
    CommandBufferPtr                            m_TouchCommandBuffer;
    IOCommandBufferPtr                          m_CopyCommandBuffer;

public:
    MaterialManipulation(float yaw = 0.0f, float roll = (30.0f * Tempest::MathPi) / 180.0f)
        :   m_Yaw(yaw),
            m_Roll(roll) {}

    ~MaterialManipulation();

    void setPlaybackAndRecord(bool playback_and_record)
    {
        if(playback_and_record)
            m_Status |= OPTION_PLAYBACK_AND_RECORD;
        else
            m_Status &= ~OPTION_PLAYBACK_AND_RECORD;
    }

    // HACK: Qt focus doesn't work properly
    void setActive(bool active)
    {
        if(active)
            m_Status |= STATUS_ACTIVE;
        else
            m_Status &= ~STATUS_ACTIVE;
    }

    void setDisableRecord(bool active)
    {
        if(active)
            m_Status |= OPTION_DISABLE_RECORD;
        else
            m_Status &= ~OPTION_DISABLE_RECORD;
    }

    void takeScreenshot()
    {
        m_Status |= ENQUEUE_SCREENSHOT;
    }

    void setAreaLightSampleCount(uint32_t count) { m_RayTracingSystem->getRayTracer()->setSamplesLocalAreaLight(count); }

    Tempest::Vector4 getDiffuseColor() const { return m_BaseDiffuseColor*m_Simulation.RemapDiffuseColor; }
    Tempest::Vector4 getSpecularColor() const { return m_BaseSpecularColor*m_Simulation.RemapSpecularColor; }

    void setDiffuseColor(const Tempest::Vector4& color) { m_Simulation.RemapDiffuseColor = color/m_BaseDiffuseColor; }
    void setSpecularColor(const Tempest::Vector4& color) { m_Simulation.RemapSpecularColor = color/m_BaseSpecularColor; }

    void setTestRotationPeriod(uint64_t test_period) { m_TestRotationPeriod = test_period; }

    void setDiffuseMultiplier(float mult) { m_Simulation.DiffuseMultiplier = mult; }
    void setSpecularMultiplier(float mult) { m_Simulation.SpecularMultiplier = mult; }
    void setAnisotropyModifier(float mod) { m_Simulation.AnisotropyModifier = mod; }
    void setSmoothnessModifier(float mod) { m_Simulation.SmoothnessModifier = mod; }

    bool init(uint32_t image_width, uint32_t image_height, uint32_t flags,
              Tempest::PreferredWindow* window = nullptr, Tempest::PreferredBackend* backend = nullptr, Tempest::PreferredShaderCompiler* shader_compiler = nullptr);
    bool run(bool swap_buffers = true);
};

#endif // _MATERIAL_MANIPULATION_HH_
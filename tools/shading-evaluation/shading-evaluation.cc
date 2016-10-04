#include "ui_shading-evaluation.h"
#include "tempest/texture/texture-table.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/math/functions.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/cube-map.hh"
#include "tempest/debug/fps-counter.hh"

#include <QApplication>
#include <QMainWindow>

#include <chrono>

typedef decltype(Tempest::PreferredSystem().Backend) BackendType;
typedef decltype(Tempest::PreferredSystem().Library) LibraryType;
typedef decltype(Tempest::PreferredSystem().ShaderCompiler) ShaderCompilerType;
typedef decltype(Tempest::PreferredSystem().Window) WindowType;
typedef BackendType::ShaderProgramType::ResourceTableType ResourceTableType;

const Tempest::Vector3 InitialOffset = Tempest::Vector3{0.0f, 0.0f, 100.0f};
const size_t BaseLayout = 2;
const size_t BaseModels = 3;

#define FREE_MEMBER_RESOURCE(parent, name) if(name) { parent.destroyRenderResource(name); name = nullptr; }

Tempest::Vector4 QColorToVector4(QColor color)
{
    float coef = 1.0f / 255.0f;
    return Tempest::Vector4{color.red() * coef, color.green() * coef, color.blue() * coef, color.alpha() * coef};
}

enum ShadingModel
{
    ADHOC_PHONG,
    ADHOC_BLINN_PHONG,
    SHADING_MODEL_COUNT
};

static const std::string ShadingModelName[] =
{
    "AdHocPhong",
    "AdHocBlinnPhong"
};

#define BUFFER_COUNT 2

class ShadingEvaluationWindow: public QMainWindow
{
    Q_OBJECT

    Ui::ShadingEvalWindow                                   m_UI;

    BackendType                                             m_Backend;
    ShaderCompilerType                                      m_ShaderCompiler;

    struct SceneParams
    {
        Tempest::Vector4 CameraPosition;
        Tempest::Vector4 SunDirection;
    } m_SceneParams;

    BackendType::BufferType                                *m_ConstantBuffer = nullptr;
    BackendType::CommandBufferType                         *m_RasterCommandBuffer = nullptr;
    BackendType::CommandBufferType                         *m_RayTraceCommandBuffer = nullptr;
    BackendType::CommandBufferType                         *m_FpsCommandBuffer = nullptr;
    ShaderCompilerType::ShaderProgramType                  *m_BackgroundShaderProgram = nullptr;
    ShaderCompilerType::ShaderProgramType                  *m_RayTraceBlit = nullptr;
    BackendType::StateObjectType                           *m_BackgroundStateObject = nullptr;
    BackendType::StateObjectType                           *m_RayTraceBlitStateObject = nullptr;
    
    Tempest::FpsCounter                                    *m_FpsCounter = nullptr;

    BackendType::IOCommandBufferType                       *m_RTIOCommandBuffers[BUFFER_COUNT];
    BackendType::StorageType                               *m_RTBackbufferStorages[BUFFER_COUNT];

    ShaderCompilerType::ShaderProgramType                  *m_MeshShading[SHADING_MODEL_COUNT];
    BackendType::StateObjectType                           *m_PipelineState[SHADING_MODEL_COUNT];
    ResourceTableType                                      *m_MeshResTable[SHADING_MODEL_COUNT];

                
    std::unique_ptr<Tempest::TextureTable<BackendType>>     m_TextureTable;
    std::unique_ptr<Tempest::MixedMeshBlob<BackendType>>    m_Mesh;
    std::unique_ptr<ResourceTableType>                      m_BackgroundResTable;
    
    float                                                   m_Roll = 0.0f,
                                                            m_Yaw = 0.0f;

    Tempest::ResourceIndex                                  m_WorldTransIdx,
                                                            m_RotTransIdx,
                                                            m_InvProjTransIdx;

    size_t                                                  m_CurrentModel = 0;
    size_t                                                  m_BufferIndex = 0;

    BackendType::TextureType                               *m_RTGPUTexture = nullptr;
    std::unique_ptr<Tempest::RayTracerScene>                m_RTScene;
    std::unique_ptr<Tempest::CubeMap>                       m_GlobalCubeMap;

    Tempest::Vector3                                        m_TeapotPosition = Tempest::Vector3{0.0f, -50.0f, 0.0f};

    Tempest::RTMicrofacetMaterial                           m_GlobalMaterial;

    Tempest::BakedResourceTable                             m_Textures;
    
public:
    ShadingEvaluationWindow()
    {
        std::fill(std::begin(m_RTBackbufferStorages), std::end(m_RTBackbufferStorages), nullptr);
        std::fill(std::begin(m_RTIOCommandBuffers), std::end(m_RTIOCommandBuffers), nullptr);
        std::fill(std::begin(m_MeshShading), std::end(m_MeshShading), nullptr);

        m_UI.setupUi(this);

        connect(m_UI.DiffuseIntensity, SIGNAL(colorChanged(const QColor&)), this, SLOT(shadingChanged()));
        connect(m_UI.SpecularIntensity, SIGNAL(colorChanged(const QColor&)), this, SLOT(shadingChanged()));
        connect(m_UI.SpecularPower, SIGNAL(valueChanged(int)), this, SLOT(shadingChanged()));
        connect(m_UI.EnvironmentIntensity, SIGNAL(colorChanged(const QColor&)), this, SLOT(shadingChanged()));
        connect(m_UI.Fresnel, SIGNAL(valueChanged(int)), this, SLOT(shadingChanged()));
    }

    ~ShadingEvaluationWindow()
    {
        FREE_MEMBER_RESOURCE(m_Backend, m_ConstantBuffer);
        FREE_MEMBER_RESOURCE(m_Backend, m_RasterCommandBuffer);
        FREE_MEMBER_RESOURCE(m_Backend, m_RayTraceCommandBuffer);
        FREE_MEMBER_RESOURCE(m_Backend, m_FpsCommandBuffer);
        FREE_MEMBER_RESOURCE(m_Backend, m_BackgroundStateObject);
        FREE_MEMBER_RESOURCE(m_Backend, m_RayTraceBlitStateObject);
        FREE_MEMBER_RESOURCE(m_Backend, m_RTGPUTexture);
        for(auto& tex : m_RTBackbufferStorages)
        {
            FREE_MEMBER_RESOURCE(m_Backend, tex);
        }
        for(auto& io_cmd : m_RTIOCommandBuffers)
        {
            FREE_MEMBER_RESOURCE(m_Backend, io_cmd);
        }
        FREE_MEMBER_RESOURCE(m_ShaderCompiler, m_BackgroundShaderProgram);
        FREE_MEMBER_RESOURCE(m_ShaderCompiler, m_RayTraceBlit);

        for(auto* pipeline : m_PipelineState)
        {
            FREE_MEMBER_RESOURCE(m_Backend, pipeline);
        }
        
        for(auto* shader : m_MeshShading)
        {
            FREE_MEMBER_RESOURCE(m_ShaderCompiler, shader);
        }

        for(auto* res_table : m_MeshResTable)
        {
            delete res_table;
        }

        delete m_FpsCounter;
    }

    bool init(LibraryType& library)
    {
        m_UI.MainView->attach(&m_Backend);

        auto status = library.initGraphicsLibrary();
        if(!status)
            return false;
        m_Backend.init();

        typedef BackendType::ShaderProgramType ShaderProgramType;

        Tempest::TextureTableDescription tex_table_desc;
        m_TextureTable = std::unique_ptr<Tempest::TextureTable<BackendType>>(new Tempest::TextureTable<BackendType>(&m_Backend, tex_table_desc));

        m_Mesh = Tempest::LoadObjFileStaticGeometryMixedBlob(TEST_ASSETS_DIR "/teapot/teapot.obj", nullptr, nullptr, m_TextureTable.get(), &m_Backend);
        TGE_ASSERT(m_Mesh, "Failed to load test assets");

        auto rt_fmt = Tempest::DataFormat::RGBA8UNorm;
        Tempest::DepthStencilStates ds_state;
        ds_state.DepthTestEnable = true;
        ds_state.DepthWriteEnable = true;

        static_assert(TGE_FIXED_ARRAY_SIZE(ShadingModelName) == SHADING_MODEL_COUNT &&
                      TGE_FIXED_ARRAY_SIZE(m_MeshResTable) == SHADING_MODEL_COUNT &&
                      TGE_FIXED_ARRAY_SIZE(m_PipelineState) == SHADING_MODEL_COUNT &&
                      TGE_FIXED_ARRAY_SIZE(m_MeshShading) == SHADING_MODEL_COUNT, "Bad size");

        for(size_t model_idx = 0, model_idx_end = SHADING_MODEL_COUNT; model_idx < model_idx_end; ++model_idx)
        {
            auto& shader = m_MeshShading[model_idx];
            shader = Tempest::CreateShader(&m_ShaderCompiler, CURRENT_SOURCE_DIR "/shading.tfx", &ShadingModelName[model_idx], 1).release();
            TGE_ASSERT(shader, "Failed to compile mesh shader");

            m_PipelineState[model_idx] = m_Backend.createStateObject(&rt_fmt, 1, Tempest::DataFormat::D24S8, shader, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);
            m_MeshResTable[model_idx] = shader->createResourceTable("Globals", 1);
        }

        m_WorldTransIdx = m_MeshResTable[0]->getResourceIndex("Globals.WorldViewProjectionTransform");
        m_RotTransIdx = m_MeshResTable[0]->getResourceIndex("Globals.RotateTransform");

        m_SceneParams.CameraPosition = Tempest::Vector4{InitialOffset.x, InitialOffset.y, InitialOffset.z, 1.0};
        m_SceneParams.SunDirection = Tempest::Vector4{0.0f, 1.0f, 1.0f, 1.0f};
        NormalizePartialSelf(&m_SceneParams.SunDirection);

        m_ConstantBuffer = m_Backend.createBuffer(sizeof(SceneParams), Tempest::ResourceBufferType::ConstantBuffer, Tempest::RESOURCE_DYNAMIC_DRAW, nullptr);

        Tempest::CommandBufferDescription cmd_buffer_desc;
        cmd_buffer_desc.CommandCount = m_Mesh->DrawBatchCount + 1;
        cmd_buffer_desc.ConstantsBufferSize = 16 * 1024 * 1024;
        m_RasterCommandBuffer = m_Backend.createCommandBuffer(cmd_buffer_desc);

        for(uint32_t i = 0, iend = m_Mesh->DrawBatchCount; i < iend; ++i)
        {
            auto& draw_batch = m_Mesh->DrawBatches[i];
            draw_batch.PipelineState = m_PipelineState[m_CurrentModel];
            draw_batch.ResourceTable = m_MeshResTable[m_CurrentModel]->getBakedTable();
            m_RasterCommandBuffer->enqueueBatch(draw_batch);
        }

        cmd_buffer_desc.CommandCount = 1;
        cmd_buffer_desc.ConstantsBufferSize = 1024;
        m_RayTraceCommandBuffer = m_Backend.createCommandBuffer(cmd_buffer_desc);

        cmd_buffer_desc.CommandCount = 16;
        cmd_buffer_desc.ConstantsBufferSize = 1024;
        m_FpsCommandBuffer = m_Backend.createCommandBuffer(cmd_buffer_desc);

        m_BackgroundShaderProgram = Tempest::CreateShader(&m_ShaderCompiler, CURRENT_SOURCE_DIR "/background.tfx").release();
        TGE_ASSERT(m_BackgroundShaderProgram, "Background shader can't be loaded");
        m_BackgroundResTable = Tempest::CreateResourceTable(m_BackgroundShaderProgram, "Globals", 1);

        m_RayTraceBlit = Tempest::CreateShader(&m_ShaderCompiler, SOURCE_SHADING_DIR "/ray-trace-blit.tfx").release();
        TGE_ASSERT(m_RayTraceBlit, "Failed to load ray trace backbuffer blit shader");

        uint32_t tex_table_slots = TGE_FIXED_ARRAY_SIZE(tex_table_desc.Slots);

        Tempest::Texture* cube_tex[] =
        {
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posx-256.png")),
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negx-256.png")),
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posy-256.png")),
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negy-256.png")),
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posz-256.png")),
            LoadImage(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negz-256.png"))
        };
        auto at_exit = Tempest::CreateAtScopeExit([&cube_tex](){
            for(size_t i = 0; i < 6; ++i)
            {
                delete cube_tex[i];
            }
        });

        m_GlobalCubeMap = std::unique_ptr<Tempest::CubeMap>(new Tempest::CubeMap(cube_tex));

        auto cube_idx = m_TextureTable->loadCube(cube_tex);
        
        m_BackgroundResTable->setResource("Globals.CubeID", cube_idx);
        m_InvProjTransIdx = m_BackgroundResTable->getResourceIndex("Globals.ViewProjectionInverseTransform");

        for(size_t model_idx = 0, model_idx_end = SHADING_MODEL_COUNT; model_idx < model_idx_end; ++model_idx)
        {
            m_MeshResTable[model_idx]->setResource("Globals.CubeID", cube_idx);
        }

        m_BackgroundStateObject = m_Backend.createStateObject(&rt_fmt, 1, Tempest::DataFormat::Unknown, m_BackgroundShaderProgram, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);
        m_RayTraceBlitStateObject = m_Backend.createStateObject(&rt_fmt, 1, Tempest::DataFormat::Unknown, m_RayTraceBlit, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);
        
        m_TextureTable->executeIOOperations();

        uint32_t aux_slots = 1;
        uint32_t total_slots = tex_table_slots + aux_slots;

        m_Textures.realloc(total_slots*4*sizeof(float));
        m_Backend.setActiveTextures(total_slots);

        Tempest::IOCommandBufferDescription io_cmd_buf_desc;
        io_cmd_buf_desc.CommandCount = 1;
        for(auto& io_cmd_buf : m_RTIOCommandBuffers)
        {
            io_cmd_buf = m_Backend.createIOCommandBuffer(io_cmd_buf_desc);
        }

        auto* tex_table_baked = m_TextureTable->getBakedTable();
        TGE_ASSERT(tex_table_baked->getSize() == tex_table_slots*4*sizeof(float), "Invalid texture table");
        memcpy(m_Textures.get(), tex_table_baked->get(), tex_table_baked->getSize());

        auto window_width = m_UI.MainView->width(),
             window_height = m_UI.MainView->height();
        m_RTScene = std::unique_ptr<Tempest::RayTracerScene>(new Tempest::RayTracerScene(window_width, window_height, Tempest::Matrix4::identityMatrix()));

        auto* submeshes = TGE_TYPED_ALLOCA(Tempest::RTSubmesh, m_Mesh->DrawBatchCount);
        for(uint32_t i = 0, iend = m_Mesh->DrawBatchCount; i < iend; ++i)
        {
            auto& draw_batch = m_Mesh->DrawBatches[i];
            auto& vb = draw_batch.VertexBuffers[0];
            // TODO: Base vertex?
            Tempest::Matrix4 world;
            world.identity();
            world.translate(m_TeapotPosition);

            auto& submesh = submeshes[i];
            submesh.BaseIndex = draw_batch.BaseIndex;
            submesh.Material = &m_GlobalMaterial;
            submesh.VertexCount = draw_batch.VertexCount;
            submesh.VertexOffset = vb.Offset;
			submesh.Stride = m_Mesh->DrawBatches[i].VertexBuffers[0].Stride;
        }

        Tempest::SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);
        m_FpsCounter = new Tempest::FpsCounter(&m_Backend, &m_ShaderCompiler, &subdir_loader, 50.0f, window_width, window_height);

        Tempest::Matrix4 world;
        world.identity();
        world.translate(m_TeapotPosition);
        m_RTScene->addTriangleMesh(world, m_Mesh->DrawBatchCount, submeshes, m_Mesh->Indices.size(), &m_Mesh->Indices[0], m_Mesh->Data.size(), &m_Mesh->Data[0]);
		
        on_ShadingModel_currentIndexChanged(0);

        m_RTScene->setGlobalCubeMap(m_GlobalCubeMap.get());
        m_RTScene->commitScene();
        m_RTScene->initWorkers();
		m_RTScene->setSamplesGlobalIllumination(16);
        updateRayTraceResources(window_width, window_height);
        
        return true;
    }
private slots:
    void shadingChanged()
    {
        Tempest::Vector4 diffuse(QColorToVector4(m_UI.DiffuseIntensity->getColor()));
        Tempest::Vector4 specular(QColorToVector4(m_UI.SpecularIntensity->getColor()));
        Tempest::Vector4 env(QColorToVector4(m_UI.EnvironmentIntensity->getColor()));

        specular.w = 5.0f*m_UI.SpecularPower->value() + 1;
        env.w = m_UI.Fresnel->value() / 100.0f;

        auto* res_table = m_MeshResTable[m_CurrentModel];
        res_table->setResource("Globals.Diffuse", diffuse);
        res_table->setResource("Globals.Specular", specular);
        res_table->setResource("Globals.Environment", env);

        m_GlobalMaterial.Specular = Tempest::SRGBToSpectrum(Tempest::Vector3{specular.x, specular.y, specular.z});
        m_GlobalMaterial.SpecularPower.x = specular.w;
        m_GlobalMaterial.Fresnel.x = 1.0f - env.w;
        m_GlobalMaterial.Diffuse = Tempest::SRGBToSpectrum(Tempest::Vector3{diffuse.x, diffuse.y, diffuse.z});
        m_GlobalMaterial.setup();
    }

    void on_ShadingModel_currentIndexChanged(int index)
    {
        m_RasterCommandBuffer->clear();

        m_CurrentModel = index;
        for(uint32_t i = 0, iend = m_Mesh->DrawBatchCount; i < iend; ++i)
        {
            auto& draw_batch = m_Mesh->DrawBatches[i];
            draw_batch.PipelineState = m_PipelineState[m_CurrentModel];
            draw_batch.ResourceTable = m_MeshResTable[m_CurrentModel]->getBakedTable();
            m_RasterCommandBuffer->enqueueBatch(draw_batch);
        }

        BackendType::CommandBufferType::DrawBatchType background_batch;
        background_batch.VertexCount = 3;
        background_batch.PipelineState = m_BackgroundStateObject;
        background_batch.ResourceTable = m_BackgroundResTable->getBakedTable();

        m_RasterCommandBuffer->enqueueBatch(background_batch);
        m_RasterCommandBuffer->prepareCommandBuffer();

        shadingChanged();
    }

    void on_MainView_rendering()
    {
        const float mouse_speed = 0.01f;

        m_Yaw += mouse_speed*m_UI.MainView->getTempestWindow().getMouseDeltaX();
        m_Roll += mouse_speed*m_UI.MainView->getTempestWindow().getMouseDeltaY();

        m_Roll = std::max(0.0f, std::min(Tempest::MathPi*0.5f, m_Roll));

        auto window_width = m_UI.MainView->width();
        auto window_height = m_UI.MainView->height();
        Tempest::Matrix4 mat = Tempest::PerspectiveMatrix(90.0f, (float)window_width / window_height, 0.1f, 1000.0f);

        Tempest::Matrix4 view;
        view.identity();
        view.translate(-InitialOffset);
        view.rotateX(m_Roll);
        view.rotateY(m_Yaw);

        mat *= view;

        Tempest::Matrix4 view_inv;
        view_inv = view.inverse();

        float azimuth = Tempest::ToRadians(m_UI.SunAzimuth->value());
        float altitude = Tempest::ToRadians(m_UI.SunAltitude->value());

        Tempest::Vector3 trans(view_inv.translation());
        m_SceneParams.CameraPosition = Tempest::Vector4{trans.x, trans.y, trans.z, 1.0f};
        m_SceneParams.SunDirection = Tempest::Vector4{cosf(azimuth)*cosf(altitude), sinf(altitude), sinf(azimuth)*cosf(altitude), 1.0f};

        // Yeah, only the sky should shine in this scene
        #if 0
        m_RTScene->clearDirectionalLightSource();
        m_RTScene->addDirectionalLightSource(Tempest::DirectionalLight{ ToVector3(m_SceneParams.SunDirection), Tempest::Vector3(1.0f, 1.0f, 1.0f) });
        #endif

        m_Backend.clearColorBuffer(0, Tempest::Vector4{1.0f, 0.0f, 1.0f, 0.0f});
        m_Backend.clearDepthStencilBuffer();

        m_Backend.setViewportRect(0, 0, window_width, window_height);

        if(m_RTScene && m_UI.RenderMode->currentIndex() == 1)
        {
            auto* frame_data = m_RTScene->draw(window_width, window_height, mat.inverse());

            auto& desc = m_RTGPUTexture->getDescription();
            auto& hdr = frame_data->Backbuffer->getHeader();
            if(desc.Width != hdr.Width ||
               desc.Height != hdr.Height)
            {
                updateRayTraceResources(hdr.Width, hdr.Height);
            }

            m_Backend.setTextures(&m_Textures);

            Tempest::TextureDescription tex_desc;
            tex_desc.Width = hdr.Width;
            tex_desc.Height = hdr.Height;
            tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;
            TGE_ASSERT(frame_data->Backbuffer, "Invalid backbuffer");
            m_RTBackbufferStorages[m_BufferIndex]->storeTexture(0, hdr, frame_data->Backbuffer->getData());

            m_Backend.submitCommandBuffer(m_RTIOCommandBuffers[m_BufferIndex]);

            m_Backend.submitCommandBuffer(m_RayTraceCommandBuffer);
        }
        else
        {
            Tempest::UploadConstantBuffer(m_ConstantBuffer, m_SceneParams);

            m_TextureTable->setTextures(&m_Backend);
            m_Backend.setConstantBuffer(0, m_ConstantBuffer);

            // Yeah, I am aware of faster alternatives.
            Tempest::Matrix4 inverse_mat = mat.inverse();

            Tempest::Matrix4 rot_mat;
            rot_mat.identity();

            mat *= rot_mat;
            mat.translate(m_TeapotPosition);

            auto* res_table = m_MeshResTable[m_CurrentModel];
            res_table->setResource(m_WorldTransIdx, mat);
            res_table->setResource(m_RotTransIdx, rot_mat);

            m_BackgroundResTable->setResource(m_InvProjTransIdx, inverse_mat);

            m_Backend.submitCommandBuffer(m_RasterCommandBuffer);
        }

        if(m_FpsCounter->update(window_width, window_height))
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

        //m_Backend.submitCommandBuffer(m_FpsCommandBuffer);

        m_BufferIndex = (m_BufferIndex + 1) % BUFFER_COUNT;
    }
private:
     void updateRayTraceResources(uint32_t width, uint32_t height)
     {
        Tempest::TextureDescription desc;
        desc.Width = width;
        desc.Height = height;
        desc.Format = Tempest::DataFormat::RGBA8UNorm;

        m_Backend.destroyRenderResource(m_RTGPUTexture);
        m_RTGPUTexture = m_Backend.createTexture(desc, Tempest::RESOURCE_DYNAMIC_DRAW);

        for(auto& tex : m_RTBackbufferStorages)
        {
            m_Backend.destroyRenderResource(tex);
            tex = m_Backend.createStorageBuffer(Tempest::StorageMode::PixelUnpack, width*height*sizeof(uint32_t));
        }
        size_t cnt = 0;
        for(auto& io_cmd_buf : m_RTIOCommandBuffers)
        {
            io_cmd_buf->clear();
            BackendType::IOCommandBufferType::IOCommandType io_command;
            io_command.CommandType = Tempest::IOCommandMode::CopyStorageToTexture;
            io_command.Source.Storage = m_RTBackbufferStorages[cnt++];
            io_command.Destination.Texture = m_RTGPUTexture;
            io_command.SourceOffset = 0;
            io_command.DestinationCoordinate.X = io_command.DestinationCoordinate.Y = 0;
            io_command.Width = width;
            io_command.Height = height;
            io_cmd_buf->enqueueCommand(io_command);
        }
        BackendType::CommandBufferType::DrawBatchType blit_batch;
        blit_batch.VertexCount = 3;
        blit_batch.PipelineState = m_RayTraceBlitStateObject;
        blit_batch.ResourceTable = nullptr;

        m_RayTraceCommandBuffer->clear();
        m_RayTraceCommandBuffer->enqueueBatch(blit_batch);
        m_RayTraceCommandBuffer->prepareCommandBuffer();

        *reinterpret_cast<uint64_t*>(m_Textures.get() + m_TextureTable->getBakedTable()->getSize()) = m_RTGPUTexture->getHandle();
     }
};

#include "shading-evaluation.moc"

int TempestMain(int argc, char** argv)
{
    QApplication app(argc, argv);

    LibraryType library;
    auto status = library.initDeviceContextLibrary();
    if(!status)
        return false;

    ShadingEvaluationWindow wnd;
    wnd.init(library);

    wnd.show();

    return app.exec();
}
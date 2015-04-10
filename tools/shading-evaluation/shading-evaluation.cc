#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector3.hh"
#include "tempest/graphics/preferred-backend.hh"
#include "ui_shading-evaluation.h"

#include <QApplication>
#include <QMainWindow>

#include <chrono>

typedef decltype(Tempest::PreferredSystem().Backend) BackendType;
typedef decltype(Tempest::PreferredSystem().Library) LibraryType;
typedef decltype(Tempest::PreferredSystem().ShaderCompiler) ShaderCompilerType;
typedef decltype(Tempest::PreferredSystem().Window) WindowType;
typedef BackendType::ShaderProgramType::ResourceTableType ResourceTableType;

const Tempest::Vector3 InitialOffset(0.0f, 0.0f, 100.0f);
const size_t BaseLayout = 2;
const size_t BaseModels = 3;

#define FREE_MEMBER_RESOURCE(parent, name) if(name) { parent.destroyRenderResource(name); name = nullptr; }

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
    BackendType::CommandBufferType                         *m_CommandBuffer = nullptr;
    ShaderCompilerType::ShaderProgramType                  *m_BackgroundShaderProgram = nullptr;

    std::unique_ptr<Tempest::TextureTable<BackendType>>     m_TextureTable;
    std::unique_ptr<Tempest::MeshBlob<BackendType>>         m_Mesh;
    std::unique_ptr<ResourceTableType>                      m_ResTable;

    Tempest::ResourceIndex                                  m_WorldTransIdx;
    Tempest::ResourceIndex                                  m_ProjTransIdx;
    Tempest::ResourceIndex                                  m_InvProjTransIdx;

    std::unique_ptr<ShaderCompilerType::ShaderProgramType*[]> m_MaterialShaderPrograms;

    float                                                   m_Roll = 0.0f,
                                                            m_Yaw = 0.0f;

public:
    ShadingEvaluationWindow()
        :   m_MaterialShaderPrograms(new ShaderCompilerType::ShaderProgramType*[BaseLayout*BaseModels])
    {
        m_UI.setupUi(this);
    }

    ~ShadingEvaluationWindow()
    {
        FREE_MEMBER_RESOURCE(m_Backend, m_ConstantBuffer);
        FREE_MEMBER_RESOURCE(m_Backend, m_CommandBuffer);
        FREE_MEMBER_RESOURCE(m_ShaderCompiler, m_BackgroundShaderProgram);

        for(Tempest::uint32 i = 0, iend = BaseLayout*BaseModels; i < iend; ++i)
        {
            FREE_MEMBER_RESOURCE(m_ShaderCompiler, m_MaterialShaderPrograms[i]);
        }
    }

    bool init(LibraryType& library)
    {
        m_UI.MainView->attach(&m_Backend);

        auto status = library.initGraphicsLibrary();
        if(!status)
            return false;
        m_Backend.init();

        typedef BackendType::ShaderProgramType ShaderProgramType;

        Tempest::string models[BaseModels - 1 + BaseLayout - 1] = { "MaterialTexture", "MaterialAmbient", "MaterialSpecular" };

        for(Tempest::uint32 i = 0; i < BaseLayout; ++i)
        {
            for(Tempest::uint32 j = 0; j < BaseModels; ++j)
            {
                Tempest::uint32 idx = i * BaseModels + j;
                auto shader = Tempest::CreateShader(&m_ShaderCompiler, SOURCE_SHADING_DIR "/blinn-phong.tfx", models + 1 - i, i + j);
                m_MaterialShaderPrograms[idx] = shader.release();
            }
        }

        Tempest::TextureTableDescription tex_table_desc;
        m_TextureTable = std::unique_ptr<Tempest::TextureTable<BackendType>>(new Tempest::TextureTable<BackendType>(&m_Backend, tex_table_desc));

        m_Mesh.swap(Tempest::LoadObjFileStaticGeometryBlob(TEST_ASSETS_DIR "/teapot/teapot.obj", nullptr, m_MaterialShaderPrograms.get(), m_TextureTable.get(), &m_Backend));
        TGE_ASSERT(m_Mesh, "Failed to load test assets");

        // Transform is always first
        m_WorldTransIdx = m_Mesh->ResourceTables[0]->getResourceIndex("Globals.WorldViewProjectionTransform");
        m_ProjTransIdx = m_Mesh->ResourceTables[0]->getResourceIndex("Globals.RotateTransform");

        m_SceneParams.CameraPosition = Tempest::Vector4(InitialOffset.x(), InitialOffset.y(), InitialOffset.z(), 1.0);
        m_SceneParams.SunDirection = Tempest::Vector4(0.0f, 1.0f, 1.0f, 1.0f);
        m_SceneParams.SunDirection.normalizePartial();

        m_ConstantBuffer = m_Backend.createBuffer(sizeof(SceneParams), Tempest::ResourceBufferType::ConstantBuffer, Tempest::RESOURCE_DYNAMIC_DRAW, nullptr);

        Tempest::CommandBufferDescription cmd_buffer_desc;
        cmd_buffer_desc.CommandCount = m_Mesh->DrawBatchCount + 1;
        cmd_buffer_desc.ConstantsBufferSize = 16 * 1024 * 1024;
        m_CommandBuffer = m_Backend.createCommandBuffer(cmd_buffer_desc);

        for(size_t i = 0; i < m_Mesh->DrawBatchCount; ++i)
        {
            m_CommandBuffer->enqueueBatch(m_Mesh->DrawBatches[i]);

            TGE_ASSERT(m_Mesh->ResourceTables[i]->getResourceIndex("Globals.WorldViewProjectionTransform") == m_WorldTransIdx, "WorldViewProjectionTransform is not first!");
            TGE_ASSERT(m_Mesh->ResourceTables[i]->getResourceIndex("Globals.RotateTransform") == m_ProjTransIdx, "RotateTransform is not second!");
        }

        m_BackgroundShaderProgram = Tempest::CreateShader(&m_ShaderCompiler, CURRENT_SOURCE_DIR "/background.tfx").release();
        TGE_ASSERT(m_BackgroundShaderProgram, "Background shader can't be loaded");
        m_ResTable = Tempest::CreateResourceTable(m_BackgroundShaderProgram, "Globals", 1);
        m_InvProjTransIdx = m_ResTable->getResourceIndex("Globals.ViewProjectionInverseTransform");

        Tempest::uint32 total_slots = TGE_FIXED_ARRAY_SIZE(tex_table_desc.Slots);

        auto cube_idx = m_TextureTable->loadCube(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posx-256.png"),
                                                 Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negx-256.png"),
                                                 Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posy-256.png"),
                                                 Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negy-256.png"),
                                                 Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posz-256.png"),
                                                 Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negz-256.png"));

        m_ResTable->setResource("Globals.CubeID", cube_idx);

        Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
        Tempest::DepthStencilStates ds_state;
        ds_state.DepthTestEnable = true;
        ds_state.DepthWriteEnable = true;
        auto bg_state_obj = Tempest::CreateStateObject(&m_Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, m_BackgroundShaderProgram, Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);

        BackendType::CommandBufferType::DrawBatchType background_batch;
        background_batch.VertexCount = 3;
        background_batch.PipelineState = bg_state_obj.get();
        background_batch.ResourceTable = m_ResTable->getBakedTable();

        m_CommandBuffer->enqueueBatch(background_batch);

        m_TextureTable->executeIOOperations();
        m_CommandBuffer->prepareCommandBuffer();

        m_Backend.setActiveTextures(total_slots);
        return true;
    }
private slots:
    void on_MainView_rendering()
    {
        const float mouse_speed = 0.01f;

        m_Yaw += mouse_speed*m_UI.MainView->getTempestWindow().getMouseDeltaX();
        m_Roll += mouse_speed*m_UI.MainView->getTempestWindow().getMouseDeltaY();

        m_Roll = std::max(0.0f, std::min(Tempest::math_pi*0.5f, m_Roll));

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

        Tempest::Vector3 trans(view_inv.translation());
        m_SceneParams.CameraPosition = Tempest::Vector4(trans.x(), trans.y(), trans.z(), 1.0f);

        Tempest::UploadConstantBuffer(m_ConstantBuffer, m_SceneParams);

        m_TextureTable->setTextures(&m_Backend);
        m_Backend.setConstantBuffer(0, m_ConstantBuffer);

        m_Backend.clearColorBuffer(0, Tempest::Vector4(0, 0, 0, 0));
        m_Backend.clearDepthStencilBuffer();

        m_Backend.setViewportRect(0, 0, window_width, window_height);

        // Yeah, I am aware of faster alternatives.
        Tempest::Matrix4 inverse_mat = mat.inverse();

        Tempest::Matrix4 rot_mat;
        rot_mat.identity();

        mat *= rot_mat;
        mat.translate(Tempest::Vector2(0.0f, -50.0f));

        for(size_t j = 0; j < m_Mesh->DrawBatchCount; ++j)
        {
            auto* res_table = m_Mesh->ResourceTables[j];

            res_table->setResource(m_WorldTransIdx, mat);
            res_table->setResource(m_ProjTransIdx, rot_mat);
        }

        m_ResTable->setResource(m_InvProjTransIdx, inverse_mat);

        m_Backend.submitCommandBuffer(m_CommandBuffer);
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
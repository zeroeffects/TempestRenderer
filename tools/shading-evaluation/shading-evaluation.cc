#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/graphics/state-object.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector3.hh"

#include <chrono>

int TempestMain(int argc, char** argv)
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::GLSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    typedef decltype(sys_obj->Backend) BackendType;
    typedef BackendType::ShaderProgramType ShaderProgramType;
    
    const size_t base_layout = 2;
    const size_t base_models = 3;
    Tempest::string models[base_models - 1 + base_layout - 1] = { "MaterialTexture", "MaterialAmbient", "MaterialSpecular" };
    std::vector<Tempest::UniqueResource<decltype(sys_obj->ShaderCompiler), ShaderProgramType>> shader_table;
    shader_table.reserve(base_layout*base_models);
    ShaderProgramType* shader_ptr_table[base_layout*base_models];

    for(Tempest::uint32 i = 0; i < base_layout; ++i)
    {
        for(Tempest::uint32 j = 0; j < base_models; ++j)
        {
            Tempest::uint32 idx = i * base_models + j;
            auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, SOURCE_SHADING_DIR "/blinn-phong.tfx", models + 1 - i, i + j);
            shader_ptr_table[idx] = shader.get();
            shader_table.push_back(std::move(shader));
        }
    }

    Tempest::TextureTableDescription tex_table_desc;
    Tempest::TextureTable<BackendType> texture_table(&sys_obj->Backend, tex_table_desc);

    auto mesh_blob = Tempest::LoadObjFileStaticGeometryBlob(TEST_ASSETS_DIR "/teapot/teapot.obj", nullptr, shader_ptr_table, &texture_table, &sys_obj->Backend);
    TGE_ASSERT(mesh_blob, "Failed to load test assets");

    // Transform is always first
    auto world_trans_idx = mesh_blob->ResourceTables[0]->getResourceIndex("Globals.WorldViewProjectionTransform");
    auto proj_trans_idx = mesh_blob->ResourceTables[0]->getResourceIndex("Globals.RotateTransform");

    struct SceneParams
    {
        Tempest::Vector4 CameraPosition;
        Tempest::Vector4 SunDirection;
    } scene_params;

    Tempest::Vector3 initial_offset(0.0f, 0.0f, 100.0f);

    scene_params.CameraPosition = Tempest::Vector4(initial_offset.x(), initial_offset.y(), initial_offset.z(), 1.0);
    scene_params.SunDirection = Tempest::Vector4(0.0f, 1.0f, 1.0f, 1.0f);
    scene_params.SunDirection.normalizePartial();

    auto const_buf = Tempest::CreateBuffer(&sys_obj->Backend, &scene_params, sizeof(SceneParams), Tempest::ResourceBufferType::ConstantBuffer, Tempest::RESOURCE_DYNAMIC_DRAW);

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = mesh_blob->DrawBatchCount + 1;
    cmd_buffer_desc.ConstantsBufferSize = 16*1024*1024;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);

    for(size_t i = 0; i < mesh_blob->DrawBatchCount; ++i)
    {
        command_buf->enqueueBatch(mesh_blob->DrawBatches[i]);

        TGE_ASSERT(mesh_blob->ResourceTables[i]->getResourceIndex("Globals.WorldViewProjectionTransform") == world_trans_idx, "WorldViewProjectionTransform is not first!");
        TGE_ASSERT(mesh_blob->ResourceTables[i]->getResourceIndex("Globals.RotateTransform") == proj_trans_idx, "RotateTransform is not second!");
    }

    auto background_shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/background.tfx");
    TGE_ASSERT(background_shader, "Background shader can't be loaded");
    auto background_res_table = Tempest::CreateResourceTable(background_shader.get(), "Globals", 1);
    auto inv_proj_trans_idx = background_res_table->getResourceIndex("Globals.ViewProjectionInverseTransform");

    Tempest::uint32 total_slots = TGE_FIXED_ARRAY_SIZE(tex_table_desc.Slots);

    auto cube_idx = texture_table.loadCube(Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posx-256.png"),
                                           Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negx-256.png"),
                                           Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posy-256.png"),
                                           Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negy-256.png"),
                                           Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/posz-256.png"),
                                           Tempest::Path(TEST_ASSETS_DIR "/Storforsen4/negz-256.png"));

    background_res_table->setResource("Globals.CubeID", cube_idx);

    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    Tempest::DepthStencilStates ds_state;
    ds_state.DepthTestEnable = true;
    ds_state.DepthWriteEnable = true;
    auto bg_state_obj = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, background_shader.get(), Tempest::DrawModes::TriangleList, nullptr, nullptr, &ds_state);

    BackendType::CommandBufferType::DrawBatchType background_batch;
    background_batch.VertexCount = 3;
    background_batch.PipelineState = bg_state_obj.get();
    background_batch.ResourceTable = background_res_table->getBakedTable();

    command_buf->enqueueBatch(background_batch);

    texture_table.executeIOOperations();
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    sys_obj->Backend.setActiveTextures(total_slots);

    float roll = 0.0f, yaw = 0.0f;

    const float mouse_speed = 0.01f;

    while(!sys_obj->Window.isDead())
    {
        yaw += mouse_speed*sys_obj->Window.getMouseDeltaX();
        roll += mouse_speed*sys_obj->Window.getMouseDeltaY();

        roll = std::max(0.0f, std::min(Tempest::math_pi*0.5f, roll));

        auto window_width = sys_obj->Window.getWidth();
        auto window_height = sys_obj->Window.getHeight();
        Tempest::Matrix4 mat = Tempest::PerspectiveMatrix(90.0f, (float)window_width / window_height, 0.1f, 1000.0f);

        Tempest::Matrix4 view;
        view.identity();
        view.translate(-initial_offset);
        view.rotateX(roll);
        view.rotateY(yaw);

        mat *= view;

        Tempest::Matrix4 view_inv;
        view_inv = view.inverse();

        Tempest::Vector3 trans(view_inv.translation());
        scene_params.CameraPosition = Tempest::Vector4(trans.x(), trans.y(), trans.z(), 1.0f);

        Tempest::UploadConstantBuffer(const_buf.get(), scene_params);

        texture_table.setTextures(&sys_obj->Backend);
        sys_obj->Backend.setConstantBuffer(0, const_buf.get());

        sys_obj->Backend.clearColorBuffer(0, Tempest::Vector4(0, 0, 0, 0));
        sys_obj->Backend.clearDepthStencilBuffer();

        sys_obj->Backend.setViewportRect(0, 0, window_width, window_height);

        // Yeah, I am aware of faster alternatives.
        Tempest::Matrix4 inverse_mat = mat.inverse();

        Tempest::Matrix4 rot_mat;
        rot_mat.identity();

        mat *= rot_mat;
        mat.translate(Tempest::Vector2(0.0f, -50.0f));

        for(size_t j = 0; j < mesh_blob->DrawBatchCount; ++j)
        {
            auto* res_table = mesh_blob->ResourceTables[j];

            res_table->setResource(world_trans_idx, mat);
            res_table->setResource(proj_trans_idx, rot_mat);
        }

        background_res_table->setResource(inv_proj_trans_idx, inverse_mat);

        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }

    return 0;
}
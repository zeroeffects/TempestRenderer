#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector3.hh"

#include <chrono>

TGE_TEST("Testing loading object files directly into the engine for testing purposes")
{
    const Tempest::uint32 amp_up_geometry = 8;

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

    auto mesh_blob = Tempest::LoadObjFileStaticGeometryBlob(TEST_ASSETS_DIR "/cube/cube.obj", nullptr, shader_ptr_table, &texture_table, &sys_obj->Backend);
    TGE_ASSERT(mesh_blob, "Failed to load test assets");

    // Transform is always first
    auto world_trans_idx = mesh_blob->ResourceTables[0]->getResourceIndex("Globals.WorldViewProjectionTransform");
    auto proj_trans_idx = mesh_blob->ResourceTables[0]->getResourceIndex("Globals.RotateTransform");

    struct SceneParams
    {
        Tempest::Vector4 CameraPosition;
        Tempest::Vector4 SunDirection;
    } scene_params;

    scene_params.CameraPosition = Tempest::Vector4(0.0f, 0.0f, 4.0f, 1.0f);
    scene_params.SunDirection = Tempest::Vector4(1.0f, 1.0f, 1.0f, 1.0f);
    scene_params.SunDirection.normalizePartial();

    auto const_buf = Tempest::CreateBuffer(&sys_obj->Backend, &scene_params, sizeof(SceneParams), Tempest::ResourceBufferType::ConstantBuffer);

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = amp_up_geometry*mesh_blob->DrawBatchCount;
    cmd_buffer_desc.ConstantsBufferSize = 16*1024*1024;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);

    size_t batch_count = amp_up_geometry*mesh_blob->DrawBatchCount;
    std::unique_ptr<Tempest::BakedResourceTable[]> baked_tables(new Tempest::BakedResourceTable[batch_count]);

    BackendType::CommandBufferType::DrawBatchType draw_batch;

    for(size_t i = 0; i < mesh_blob->DrawBatchCount; ++i)
    {
        draw_batch = mesh_blob->DrawBatches[i];
        auto* orig_table = draw_batch.ResourceTable;
        for(size_t j = 0; j < amp_up_geometry; ++j)
        {
            auto size = orig_table->getSize();
            auto& inst_table = baked_tables[j*mesh_blob->DrawBatchCount + i]; // Because we are doing it this way inside the loop
            inst_table.realloc(size);
            memcpy(inst_table.get(), orig_table->get(), size);
            draw_batch.ResourceTable = &inst_table;

            command_buf->enqueueBatch(draw_batch);

            TGE_ASSERT(mesh_blob->ResourceTables[i]->getResourceIndex("Globals.WorldViewProjectionTransform") == world_trans_idx, "WorldViewProjectionTransform is not first!");
            TGE_ASSERT(mesh_blob->ResourceTables[i]->getResourceIndex("Globals.RotateTransform") == proj_trans_idx, "RotateTransform is not second!");
        }
    }

    texture_table.executeIOOperations();
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    size_t period = 5;

    Tempest::uint32 total_slots = TGE_FIXED_ARRAY_SIZE(tex_table_desc.Slots);

    sys_obj->Backend.setActiveTextures(total_slots);

    while(!sys_obj->Window.isDead())
    {
        texture_table.setTextures(&sys_obj->Backend);
        sys_obj->Backend.setConstantBuffer(0, const_buf.get());

        sys_obj->Backend.clearColorBuffer(0, Tempest::Vector4(0, 0, 0, 0));
        sys_obj->Backend.clearDepthStencilBuffer();

        auto now = std::chrono::system_clock::now();

        for(size_t i = 0; i < amp_up_geometry; ++i)
        {
            Tempest::Matrix4 mat = Tempest::PerspectiveMatrix(90.0f, (float)sys_obj->Window.getWidth() / sys_obj->Window.getHeight(), 0.1f, 10.0f);
            mat.translate(-Tempest::ToVector3(scene_params.CameraPosition));

            Tempest::Matrix4 rot_mat;
            rot_mat.identity();
            rot_mat.rotateY(2.0f * Tempest::math_pi * (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 5000) / 5000.0f);

            mat.translate(Tempest::Vector2((amp_up_geometry*0.5f - i - 0.5f)*1.0f, 0.0f));
            //mat.translate(-Tempest::Vector2(((i % 4) / 4.0f - 0.5f), ((i / 4) / 4.0f - 0.5f)));

            mat *= rot_mat;

            for(size_t j = 0; j < mesh_blob->DrawBatchCount; ++j)
            {
                auto* res_table = mesh_blob->ResourceTables[j];

                auto& cur_baked_table = baked_tables[i*mesh_blob->DrawBatchCount + j];
                res_table->swapBakedTable(cur_baked_table);

                res_table->setResource(world_trans_idx, mat);
                res_table->setResource(proj_trans_idx, rot_mat);

                res_table->swapBakedTable(cur_baked_table);
            }
        }

        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}
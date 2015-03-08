#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"

#include <chrono>

TGE_TEST("Testing loading object files directly into the engine for testing purposes")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::GLSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

    typedef decltype(sys_obj->Backend) BackendType;
    typedef BackendType::ShaderProgramType ShaderProgramType;
    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    
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

    auto mesh_blob = Tempest::LoadObjFileStaticGeometryBlob(TEST_ASSETS_DIR "/cube/cube.obj", nullptr, shader_ptr_table, &sys_obj->Backend);
    TGE_ASSERT(mesh_blob, "Failed to load test assets");

    // Transform is always first
    auto trans_idx = mesh_blob->ResourceTables[0]->getResourceIndex("Globals.Transform");
    
    for(size_t i = 0; i < mesh_blob->DrawBatchCount; ++i)
    {
        command_buf->enqueueBatch(mesh_blob->DrawBatches[i]);
        TGE_ASSERT(mesh_blob->ResourceTables[i]->getResourceIndex("Globals.Transform") == trans_idx, "Transform is not first!");
    }

    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    size_t period = 5;

    for(;;)
    {
        sys_obj->Backend.clearColorBuffer(0, Tempest::Vector4(0, 0, 0, 0));
        sys_obj->Backend.clearDepthStencilBuffer();

        for(size_t i = 0; i < mesh_blob->DrawBatchCount; ++i)
        {
            auto now = std::chrono::system_clock::now();

            Tempest::Matrix4 mat = Tempest::PerspectiveMatrix(90.0f, (float)sys_obj->Window.getWidth() / sys_obj->Window.getHeight(), 0.1f, 10.0f);
            mat.translate(Tempest::Vector3(0.0f, 0.0f, -2.0f));
            mat.rotateY(2.0f * Tempest::math_pi * (std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 5000) / 5000.0f);

            mesh_blob->ResourceTables[i]->setResource(trans_idx, mat);
        }

        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}
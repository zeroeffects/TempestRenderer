#include "tempest/utils/testing.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/debug/fps-counter.hh"

TGE_TEST("Testing the rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_CHECK(sys_obj, "GL initialization failed");

    Tempest::SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);

    Tempest::FpsCounter fps_counter(&sys_obj->Backend, &sys_obj->ShaderCompiler, &subdir_loader, 50.0f, (float)wdesc.Width, (float)wdesc.Height);
    fps_counter.update((float)wdesc.Width, (float)wdesc.Height);
    auto fps_counter_batch_count = fps_counter.getDrawBatchCount();
    auto fps_counter_batches = fps_counter.getDrawBatches();

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = fps_counter_batch_count;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);

    for(size_t i = 0; i < fps_counter_batch_count; ++i)
    {
        command_buf->enqueueBatch(fps_counter_batches[i]);
    }

    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    while(!sys_obj->Window.isDead())
    {
        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }
}
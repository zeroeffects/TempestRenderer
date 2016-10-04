#include "tempest/graphics/api-all.hh"
#include "tempest/debug/fps-counter.hh"

int TempestMain(int argc, char** argv)
{
	Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

	Tempest::SubdirectoryFileLoader subdir_loader(SOURCE_SHADING_DIR);
    std::unique_ptr<Tempest::FpsCounter> fps_counter(new Tempest::FpsCounter(&sys_obj->Backend, &sys_obj->ShaderCompiler, &subdir_loader, 150.0f, static_cast<float>(wdesc.Width), static_cast<float>(wdesc.Height)));

    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;
    auto fps_cmd_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);

	sys_obj->Window.show();
    
    while(!sys_obj->Window.isDead())
    {
		if(fps_counter->update(static_cast<float>(wdesc.Width), static_cast<float>(wdesc.Height)))
        {
            fps_cmd_buf->clear();
            auto draw_batches = fps_counter->getDrawBatches();
            auto batch_count = fps_counter->getDrawBatchCount();
            for(decltype(batch_count) i = 0; i < batch_count; ++i)
            {
                fps_cmd_buf->enqueueBatch(draw_batches[i]);
            }

            fps_cmd_buf->prepareCommandBuffer();
        }

        sys_obj->Backend.submitCommandBuffer(fps_cmd_buf.get());

        sys_obj->Window.swapBuffers();
    }

	return EXIT_SUCCESS;
}

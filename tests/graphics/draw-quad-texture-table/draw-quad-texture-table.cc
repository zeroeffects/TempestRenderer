#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/math/matrix4.hh"

#include <numeric>

#define TEMPEST_RENDERING_SYSTEM Tempest::GLSystem

TGE_TEST("Testing texture tables")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test";
    
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEMPEST_RENDERING_SYSTEM>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    typedef decltype(sys_obj->Backend) BackendType;

    Tempest::CommandBufferDescription cmd_buf_desc;
    cmd_buf_desc.CommandCount = 16;
    cmd_buf_desc.ConstantsBufferSize = 1024;
    auto cmd_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buf_desc);

    struct BatchDescription
    {
        Tempest::string Name;
        Tempest::Vector4 Transform;
        BackendType::CommandBufferType::DrawBatchType OutBatch;
    };
    
    BatchDescription batch_descriptions[] =
    {
        { CURRENT_SOURCE_DIR "/barbara.png", Tempest::Vector4(-1.0f, -1.0f, 1.0f, 1.0f) },
        { CURRENT_SOURCE_DIR "/brick-house.png", Tempest::Vector4(0.0f, 0.5f, 0.25f, 0.25f) },
        { CURRENT_SOURCE_DIR "/peppers.png", Tempest::Vector4(0.25f, -0.75f, 0.5f, 0.5f) }
    };

    struct VertexFormat
    {
        Tempest::Vector2         Vertex;
        Tempest::Vector2         TexCoord;
    };

    std::vector<Tempest::VertexAttributeDescription> attr
    {
        { 0, "Position", Tempest::DataFormat::RG32F, 0 },
        { 0, "TexCoord", Tempest::DataFormat::RG32F, sizeof(Tempest::Vector2) }
    };

    auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/test.tfx");
    
    auto rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    auto state = Tempest::CreateStateObject(&sys_obj->Backend, attr, &rt_fmt, 1, shader.get());

    Tempest::TextureTableDescription tex_desc;
    Tempest::TextureTable<BackendType> texture_table(&sys_obj->Backend, tex_desc);

    const auto batch_count = TGE_FIXED_ARRAY_SIZE(batch_descriptions);

    std::array<Tempest::uint16, 6*batch_count> indices;
    std::array<VertexFormat, 4*batch_count> data;

    typedef BackendType::ShaderProgramType::ResourceTableType::BakedResourceTableType BakedResourceTableType;

    std::vector<std::unique_ptr<BakedResourceTableType>> tables;
    
    size_t ind = 0;
    Tempest::uint16 vert = 0;
    Tempest::Vector2 remap(0.0f, 0.0f);
    auto res_table = Tempest::CreateResourceTable(shader.get(), "Globals", 1);
    auto translate_res_idx = res_table->getResourceIndex("Globals.Transform");
    TGE_ASSERT(translate_res_idx.ResourceTableIndex != Tempest::InvalidResourceIndex, "Unknown resource");
    auto texture_res_idx = res_table->getResourceIndex("Globals.TextureID");
    TGE_ASSERT(texture_res_idx.ResourceTableIndex != Tempest::InvalidResourceIndex, "Unknown resource");
    for(auto& batch_desc : batch_descriptions)
    {
        res_table->resetBakedTable();
        auto idx = texture_table.loadTexture(Tempest::Path(batch_desc.Name), &remap);
        res_table->setResource(translate_res_idx, batch_desc.Transform);
        res_table->setResource(texture_res_idx, idx);
        auto baked_table = Tempest::ExtractBakedResourceTable(res_table.get());
        auto& batch = batch_desc.OutBatch;
        batch.BaseIndex = 0;
        batch.BaseVertex = vert;
        batch.VertexCount = 6;
        batch.PipelineState = state.get();
        batch.ResourceTable = baked_table.get();

        indices[ind++] = 0;
        indices[ind++] = 1;
        indices[ind++] = 2;
        indices[ind++] = 0;
        indices[ind++] = 2;
        indices[ind++] = 3;
        data[vert++] = VertexFormat{ Tempest::Vector2(0.0f, 0.0f), Tempest::Vector2(0.0f, 0.0f) };
        data[vert++] = VertexFormat{ Tempest::Vector2(0.0f, 1.0f), Tempest::Vector2(0.0f, remap.y()) };
        data[vert++] = VertexFormat{ Tempest::Vector2(1.0f, 1.0f), Tempest::Vector2(remap.x(), remap.y()) };
        data[vert++] = VertexFormat{ Tempest::Vector2(1.0f, 0.0f), Tempest::Vector2(remap.x(), 0.0f) };
        tables.push_back(std::move(baked_table));
    }

    auto vert_buf = Tempest::CreateBuffer(&sys_obj->Backend, data, Tempest::VBType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&sys_obj->Backend, indices, Tempest::VBType::IndexBuffer);

    for(auto& batch_desc : batch_descriptions)
    {
        auto& batch = batch_desc.OutBatch;
        batch.IndexBuffer = index_buf.get();
        batch.VertexBuffers[0].VertexBuffer = vert_buf.get();
        batch.VertexBuffers[0].Stride = 2*sizeof(Tempest::Vector2);
        batch.VertexBuffers[0].Offset = 0;
        cmd_buf->enqueueBatch(batch);
    }

    texture_table.executeIOOperations();
    cmd_buf->prepareCommandBuffer();

    sys_obj->Window.show();

    Tempest::uint32 total_slots = std::accumulate(std::begin(tex_desc.Slots), std::end(tex_desc.Slots), 0);
    
    sys_obj->Backend.setActiveTextures(total_slots);

    for(;;)
    {
        texture_table.setTextures(&sys_obj->Backend);

        sys_obj->Backend.submitCommandBuffer(cmd_buf.get());
        sys_obj->Window.swapBuffers();
    }
}
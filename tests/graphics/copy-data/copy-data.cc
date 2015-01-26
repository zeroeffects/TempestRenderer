#include "tempest/utils/testing.hh"
#include "tempest/graphics/rendering-convenience.hh"
#include "tempest/graphics/api-all.hh"

#define TEMPEST_RENDERING_SYSTEM Tempest::GLSystem

TGE_TEST("Testing the rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<TEMPEST_RENDERING_SYSTEM>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    typedef decltype(sys_obj->Backend) BackendType;
    
    const Tempest::uint32 edge = 2;
    const auto edge2 = edge*edge;

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = tex_desc.Height = edge;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    Tempest::uint32 in_tex_data[edge2]
    {
        0xABBA, 0x15,
        0x01D, 0x570FF
    };

    auto texture = Tempest::CreateTexture(&sys_obj->Backend, tex_desc);
    auto storage_write = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::PixelUnpack, sizeof(in_tex_data));
    auto storage_read = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::PixelPack, sizeof(in_tex_data));

    storage_write->storeTexture(0, tex_desc, in_tex_data);

    Tempest::IOCommandBufferDescription io_cmd_buf_desc;
    io_cmd_buf_desc.CommandCount = 128;

    auto io_command_buffer = Tempest::CreateIOCommandBuffer(&sys_obj->Backend, io_cmd_buf_desc);
    
    typedef BackendType::IOCommandBufferType::IOCommandType IOCommandType;
    IOCommandType io_cmd;
    io_cmd.CommandType = Tempest::IOCommandMode::CopyStorageToTexture;
    io_cmd.Width = io_cmd.Height = edge;
    io_cmd.Source.Storage = storage_write.get();
    io_cmd.Destination.Texture = texture.get();
    io_command_buffer->enqueueCommand(io_cmd);

    io_cmd.CommandType = Tempest::IOCommandMode::CopyTextureToStorage;
    io_cmd.Source.Texture = texture.get();
    io_cmd.Destination.Storage = storage_read.get();
    io_command_buffer->enqueueCommand(io_cmd);

    sys_obj->Backend.submitCommandBuffer(io_command_buffer.get());

    Tempest::uint32 out_tex_data[edge2];
    storage_read->extractTexture(0, tex_desc, out_tex_data);

    auto res = std::equal(std::begin(in_tex_data), std::end(in_tex_data), std::begin(out_tex_data));
    TGE_ASSERT(res, "Texture data should be the same after transferring it back and forth");
}
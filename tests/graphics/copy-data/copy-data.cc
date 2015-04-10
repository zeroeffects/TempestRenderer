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

    Tempest::IOCommandBufferDescription io_cmd_buf_desc;
    io_cmd_buf_desc.CommandCount = 128;

    auto io_command_buffer = Tempest::CreateIOCommandBuffer(&sys_obj->Backend, io_cmd_buf_desc);

    typedef decltype(sys_obj->Backend) BackendType;
    typedef BackendType::IOCommandBufferType::IOCommandType IOCommandType;

    {
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

    io_command_buffer->clear();

    {
        std::unique_ptr<Tempest::Texture> tex(Tempest::LoadImage(Tempest::Path(CURRENT_SOURCE_DIR "/Mandrill.tga")));
        TGE_ASSERT(tex, "Failed to load texture");
        
        auto& hdr = tex->getHeader();
        Tempest::uint32 size = hdr.Width*hdr.Height*DataFormatElementSize(hdr.Format);

        auto tex_storage_write = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::PixelUnpack, size);

        tex_storage_write->storeTexture(0, hdr, tex->getData());

        {
            auto texture2d = Tempest::CreateTexture(&sys_obj->Backend, hdr);
            auto tex_storage_read = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::PixelPack, size);

            IOCommandType io_cmd;
            io_cmd.CommandType = Tempest::IOCommandMode::CopyStorageToTexture;
            io_cmd.Width = hdr.Width;
            io_cmd.Height = hdr.Height;
            io_cmd.Source.Storage = tex_storage_write.get();
            io_cmd.Destination.Texture = texture2d.get();
            io_command_buffer->enqueueCommand(io_cmd);

            io_cmd.CommandType = Tempest::IOCommandMode::CopyTextureToStorage;
            io_cmd.Source.Texture = texture2d.get();
            io_cmd.Destination.Storage = tex_storage_read.get();
            io_command_buffer->enqueueCommand(io_cmd);

            sys_obj->Backend.submitCommandBuffer(io_command_buffer.get());

            std::unique_ptr<Tempest::uint8> result_tex_data(new Tempest::uint8[size]);
            tex_storage_read->extractTexture(0, hdr, result_tex_data.get());

            auto res = std::equal(tex->getData(), tex->getData() + size, result_tex_data.get());
            TGE_ASSERT(res, "Texture data should be the same after transferring it back and forth");
        }

        io_command_buffer->clear();

        {
            auto tex_storage_read = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::PixelPack, size);

            Tempest::TextureDescription array_desc = hdr;
            array_desc.Tiling = Tempest::TextureTiling::Array;
            array_desc.Depth = 16;
            auto texture_array = Tempest::CreateTexture(&sys_obj->Backend, array_desc);

            IOCommandType io_cmd;
            io_cmd.CommandType = Tempest::IOCommandMode::CopyStorageToTexture;
            io_cmd.Width = hdr.Width;
            io_cmd.Height = hdr.Height;
            io_cmd.DestinationSlice = 2;
            io_cmd.SourceOffset = 0;
            io_cmd.Source.Storage = tex_storage_write.get();
            io_cmd.DestinationCoordinate.X = io_cmd.DestinationCoordinate.Y = 0;
            io_cmd.Destination.Texture = texture_array.get();
            io_command_buffer->enqueueCommand(io_cmd);

            io_cmd.CommandType = Tempest::IOCommandMode::CopyTextureToStorage;
            io_cmd.SourceSlice = io_cmd.DestinationSlice;
            io_cmd.DestinationSlice = 0;
            io_cmd.Source.Texture = texture_array.get();
            io_cmd.SourceCoordinate.X = io_cmd.SourceCoordinate.Y = 0;
            io_cmd.Destination.Storage = tex_storage_read.get();
            io_cmd.DestinationOffset = 0;
            io_command_buffer->enqueueCommand(io_cmd);

            sys_obj->Backend.submitCommandBuffer(io_command_buffer.get());

            std::unique_ptr<Tempest::uint8> result_tex_data(new Tempest::uint8[size]);
            tex_storage_read->extractTexture(0, hdr, result_tex_data.get());

            auto res = std::equal(tex->getData(), tex->getData() + size, result_tex_data.get());
            TGE_ASSERT(res, "Texture data should be the same after transferring it back and forth");
        }
    }
}
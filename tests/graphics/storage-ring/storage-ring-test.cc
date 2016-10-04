#include "tempest/utils/testing.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/graphics/storage-ring.hh"

TGE_TEST("Testing the rendering context")
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_CHECK(sys_obj, "GL initialization failed");

    char data[2*1024];

    Tempest::StorageRing<Tempest::PreferredBackend> ring(&sys_obj->Backend, 1024);

    auto index = ring.pushData(data, 2*1024);
    TGE_CHECK(index == Tempest::StorageRing<Tempest::PreferredBackend>::InvalidIndex, "Broken insertion procedure");

    index = ring.pushData(data, 1024);
    index = ring.pushData(data, 1024);
    TGE_CHECK(index == 0, "Invalid index");

    index = ring.pushData(data, 1);
    index = ring.pushData(data, 15);
    TGE_CHECK(index == 1, "Invalid index");

    index = ring.pushData(data, 1024);
    TGE_CHECK(index == 0, "Invalid index");
}
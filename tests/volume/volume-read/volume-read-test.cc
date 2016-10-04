#include "tempest/utils/testing.hh"
#include "tempest/volume/volume.hh"

#include <memory>

TGE_TEST("Testing volume reading capabilities")
{
    std::unique_ptr<Tempest::VolumeRoot> volume(Tempest::ParseVolumeHierarchy(TEST_ASSETS_DIR "/scarf/data/volume"));
    TGE_CHECK(volume, "Failed to parse volume");
}
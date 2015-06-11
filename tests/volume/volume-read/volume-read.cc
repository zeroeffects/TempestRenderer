#include "tempest/utils/testing.hh"
#include "tempest/volume/volume.hh"

#include <memory>

TGE_TEST("Testing volume reading capabilities")
{
    std::unique_ptr<Tempest::VolumeRoot> volume(Tempest::ParseVolume(TEST_ASSETS_DIR "/scarf/data/volume"));
    TGE_ASSERT(volume, "Failed to parse volume");
}
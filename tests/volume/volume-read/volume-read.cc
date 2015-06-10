#include "tempest/utils/testing.hh"
#include "tempest/volume/volume.hh"

TGE_TEST("Testing volume reading capabilities")
{
    auto* volume = Tempest::ParseVolume(TEST_ASSETS_DIR "/scarf/data/volume");
    TGE_ASSERT(volume, "Failed to parse volume");
}
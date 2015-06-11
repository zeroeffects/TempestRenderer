/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in
*   all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*   THE SOFTWARE.
*/

#include "tempest/volume/volume.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/file-system.hh"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>

namespace Tempest
{
Volume::~Volume()
{
    delete[] GridData;
}

VolumeRoot::~VolumeRoot()
{
    delete[] Volumes;
}

VolumeRoot* ParseVolume(const string& name)
{
    string vol_descr = name + "_description.vol";

    std::fstream root_fs(vol_descr.c_str(), std::ios::binary | std::ios::in);
    if(!root_fs)
    {
        Log(LogLevel::Error, "Failed to load volume description file for reading: ", name);
        return nullptr;
    }        
    
    std::unique_ptr<VolumeRoot> volume_root(new VolumeRoot);

    root_fs.read(reinterpret_cast<char*>(&volume_root->MinCorner), sizeof(volume_root->MinCorner));
    root_fs.read(reinterpret_cast<char*>(&volume_root->MaxCorner), sizeof(volume_root->MaxCorner));
    root_fs.read(reinterpret_cast<char*>(&volume_root->Dimensions), sizeof(volume_root->Dimensions));

    size_t grid_dims = volume_root->Dimensions.X*volume_root->Dimensions.Y*volume_root->Dimensions.Z;
    volume_root->FileFormat = GridDataType::Invalid;

    auto* volumes = volume_root->Volumes = new Volume[grid_dims];
    memset(volumes, 0, grid_dims*sizeof(Volume));

    while(root_fs.good())
    {
        Box block;
        root_fs.read(reinterpret_cast<char*>(&block), sizeof(block));

        std::stringstream ss; // yeah, i know
        ss << name << "_" << std::setw(3) << std::setfill('0') << block.X << "_"
                          << std::setw(3) << std::setfill('0') << block.Y << "_"
                          << std::setw(3) << std::setfill('0') << block.Z << "-density.vol";
        string vg_name = ss.str();

        std::fstream grid_fs(vg_name.c_str(), std::ios::binary | std::ios::in);

        if(!grid_fs)
        {
            Log(LogLevel::Error, "Failed to load volume grid file for reading: ", name);
            return nullptr;
        }

        uint32 magic;
        grid_fs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
        if(!grid_fs || magic != TEMPEST_MAKE_FOURCC('V', 'O', 'L', '\x3'))
        {
            Log(LogLevel::Error, "Invalid volume grid file format: ", vg_name);
            return nullptr;
        }

        auto& volume = volumes[(block.Z * volume_root->Dimensions.Y + block.Y) * volume_root->Dimensions.X + block.X];

        int32 channels = 0;
        Vector3 ab1, ab2;
        GridDataType fmt = GridDataType::Invalid;
        static_assert(sizeof(fmt) + sizeof(volume.Dimensions) == 4*4, "Invalid format");
        grid_fs.read(reinterpret_cast<char*>(&fmt), sizeof(fmt));
        grid_fs.read(reinterpret_cast<char*>(&volume.Dimensions), sizeof(volume.Dimensions));
        grid_fs.read(reinterpret_cast<char*>(&channels), sizeof(channels));
        
        size_t grid_chunk_size = volume.Dimensions.X * volume.Dimensions.Y * volume.Dimensions.Z;
        size_t elem_size = 0;

        if(volume_root->FileFormat == GridDataType::Invalid)
        {
            switch(fmt)
            {
            case GridDataType::Float32:
            {
                elem_size = sizeof(float);
            } break;
            case GridDataType::Float16:
            {
                elem_size = sizeof(float)/2;
            } break;
            case GridDataType::UInt8:
            {
                elem_size = sizeof(uint8);
            } break;
            default:
            {
                Log(LogLevel::Error, "Unsupported file format");
                return nullptr;
            }
            }

            volume_root->Channels = channels;
            volume_root->FileFormat = fmt;
        }

        size_t data_size = channels*elem_size*grid_chunk_size;
        volume.GridData = new uint8[data_size];

        grid_fs.read(reinterpret_cast<char*>(&ab1), sizeof(ab1));
        grid_fs.read(reinterpret_cast<char*>(&ab2), sizeof(ab2));

        if(!grid_fs)
        {
            Log(LogLevel::Error, "Invalid volume bounding box: ", vg_name);
            return nullptr;
        }

        grid_fs.read(reinterpret_cast<char*>(volume.GridData), data_size);
        if(!grid_fs)
        {
            Log(LogLevel::Error, "Incomplete volume data: ", vg_name);
            return nullptr;
        }
    }

    return volume_root.release();
}
}
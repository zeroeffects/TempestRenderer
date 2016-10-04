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
#include <cstring>

namespace Tempest
{
extern float s_cosPhi[256],
			 s_sinPhi[256],
			 s_cosTheta[256],
			 s_sinTheta[256];

struct GridParameters
{
    GridDataType Type;
    size_t Channels;
    const char*  DefaultSuffix;
};
    
static const GridParameters GridParams[2] =
{
    { GridDataType::UInt8, 1, "-density.vol" },
    { GridDataType::QuantizedDirections, 1, "-orientation.vol" }
};

Volume::~Volume()
{
    free(Data);
}

VolumeRoot::~VolumeRoot()
{
    delete[] Volumes;
}

size_t GetGridDataTypeSize(GridDataType grid_type)
{
    switch(grid_type)
    {
    case GridDataType::Float32:
    {
        return sizeof(float);
    } break;
    case GridDataType::Float16:
    {
        return sizeof(float)/2;
    } break;
    case GridDataType::UInt8:
    {
        return sizeof(uint8_t);
    } break;
    case GridDataType::QuantizedDirections:
    {
        return sizeof(uint16_t);
    } break;
    default:
    {
        Log(LogLevel::Error, "Unsupported file format");
    }
    }
    return 0;
}

static bool ParseVolumeElement(const char* filename, GridDataType param_type, size_t param_channels, size_t prealloc_size, Volume* volume, size_t data_offset, size_t* data_size, Vector3* aabb_min, Vector3* aabb_max)
{
    std::fstream grid_fs(filename, std::ios::binary | std::ios::in);
    if(!grid_fs)
    {
        Log(LogLevel::Error, "Failed to load volume grid file for reading: ", filename);
        return nullptr;
    }

    uint32_t magic;
    grid_fs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    if(!grid_fs || magic != TEMPEST_MAKE_FOURCC('V', 'O', 'L', '\x3'))
    {
        Log(LogLevel::Error, "Invalid volume grid file format: ", filename);
        return false;
    }

    int32_t channels = 0;
    GridDataType fmt = GridDataType::Invalid;
    Box dims;
    static_assert(sizeof(fmt) + sizeof(dims) == 4*4, "Invalid format");
    grid_fs.read(reinterpret_cast<char*>(&fmt), sizeof(fmt));
    grid_fs.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    grid_fs.read(reinterpret_cast<char*>(&channels), sizeof(channels));
        
    size_t grid_chunk_size = dims.X * dims.Y * dims.Z;

    size_t elem_size = GetGridDataTypeSize(param_type);
    if(elem_size == 0)
        return false;    

    TGE_ASSERT(channels == param_channels || (param_type == GridDataType::QuantizedDirections && channels >= 2), "channel count");

    *data_size = elem_size*param_channels*grid_chunk_size;

    if(volume->Data == nullptr)
    {
        TGE_ASSERT(data_offset == 0, "Invalid parameter");
        volume->Dimensions = dims;
        volume->Data = malloc(grid_chunk_size*prealloc_size);
        memset(volume->Data, 0, grid_chunk_size*prealloc_size);
    }
    else
    {
        TGE_ASSERT(volume->Dimensions == dims, "Invalid dimensions");
    }

    grid_fs.read(reinterpret_cast<char*>(aabb_min), sizeof(*aabb_min));
    grid_fs.read(reinterpret_cast<char*>(aabb_max), sizeof(*aabb_max));

    if(!grid_fs)
    {
        Log(LogLevel::Error, "Invalid volume bounding box: ", filename);
        return false;
    }

    if(fmt == param_type)
    {
        grid_fs.read(reinterpret_cast<char*>(volume->Data) + data_offset, *data_size);
        TGE_ASSERT(grid_fs.good(), "broken reader function");
        if(!grid_fs)
        {
            Log(LogLevel::Error, "Incomplete volume data: ", filename);
            return false;
        }
    }
    else
    {
        size_t actual_size = (fmt != GridDataType::QuantizedDirections ? channels : 1)*GetGridDataTypeSize(fmt)*grid_chunk_size;

        std::unique_ptr<char[]> interm_data(new char[actual_size]);
        grid_fs.read(interm_data.get(), actual_size);
        TGE_ASSERT(grid_fs.good(), "broken reader function");

        switch(param_type)
        {
        case GridDataType::Float32:
        {
            switch(fmt)
            {
            /*
            case GridDataType::Float16:
            {
                
            } break;
            */
            case GridDataType::UInt8:
            {
                auto* out_type = reinterpret_cast<float*>(reinterpret_cast<char*>(volume->Data) + data_offset);
                auto* in_type = reinterpret_cast<uint8_t*>(interm_data.get());
                for(size_t i = 0; i < grid_chunk_size; ++i)
                {
                    for(size_t ch = 0; ch < channels; ++ch)
                    {
                        out_type[i*channels + ch] = (1.0f/255.0f)*in_type[i*channels + ch];
                    }
                }
            } break;
            /*
            case GridDataType::QuantizedDirections:
            {
            
            } break;
            */
            default:
            {
                Log(LogLevel::Error, "Unsupported file format");
                return false;
            }
            }
        } break;
        /*
        case GridDataType::Float16:
        {
            switch(fmt)
            {
            case GridDataType::Float32:
            {
                
            } break;
            case GridDataType::UInt8:
            {
            
            } break;
            case GridDataType::QuantizedDirections:
            {
            
            } break;
            default:
            {
                Log(LogLevel::Error, "Unsupported file format");
            }
            }
        } break;
        */
        case GridDataType::UInt8:
        {
            switch(fmt)
            {
            case GridDataType::Float32:
            {
                auto* out_values = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(volume->Data) + data_offset);
                auto* in_values = reinterpret_cast<float*>(interm_data.get());
                for(size_t i = 0; i < grid_chunk_size; ++i)
                {
                    for(size_t ch = 0; ch < channels; ++ch)
                    {
                        float in_value = in_values[i*channels + ch];
                        TGE_ASSERT(in_value <= 1.0f, "Invalid density");
                        out_values[i*channels + ch] = static_cast<uint8_t>(255.0f*in_value);
                    }
                }
            } break;
            /*
            case GridDataType::Float16:
            {
            
            } break;
            case GridDataType::QuantizedDirections:
            {
            
            } break;
            */
            default:
            {
                Log(LogLevel::Error, "Unsupported file format");
                return false;
            }
            }
        } break;
        case GridDataType::QuantizedDirections:
        {
            switch(fmt)
            {
            case GridDataType::Float32:
            {
                if(channels < 2)
                {
                    Log(LogLevel::Error, "Unsupported file format");
                    return false;
                }

                auto* out_values = reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(volume->Data) + data_offset);
                auto* in_values = reinterpret_cast<float*>(interm_data.get());

                for(size_t i = 0; i < grid_chunk_size; ++i)
                {
                    float x = in_values[i*channels + 0];
                    float y = in_values[i*channels + 1];
                    float sin_theta2 = x*x + y*y;

                    float cos_theta, sin_theta;
                    
                    if(channels < 3)
                    {
                        sin_theta = sqrtf(sin_theta2);
                        cos_theta = sqrtf(1.0f - sin_theta2);
                    }
                    else
                    {
                        cos_theta = in_values[i*channels + 2];
                        sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
                    }

                    float cos_phi, sin_phi;
                    if(sin_theta < 1e-3f)
                    {
                        cos_phi = 1.0f;
                        sin_phi = 0.0f;
                    }
                    else
                    {
                        cos_phi = Clamp(x/sin_theta, -1.0f, 1.0f);
                        sin_phi = copysign(sqrtf(1.0f - cos_phi*cos_phi), y);
                    }

                    float theta = atan2(sin_theta, cos_theta);
                    float phi = atan2(sin_phi, cos_phi);

                    if(phi < 0.0f)
                    {
                        phi += 2.0f*MathPi;
                    }
                    
                    TGE_ASSERT(-1e-3f <= theta && theta <= MathPi + 1e-3f, "Invalid angle");
                    TGE_ASSERT(-1e-3f <= phi && phi <= 2.0f*MathPi, "Invalid angle");

                    auto theta_quant = (uint8_t)Clamp(theta*255.0f/MathPi + 0.5f, 0.0f, 255.0f);
                    auto phi_quant = (uint8_t)Clamp(phi*255.0f/(2.0f*MathPi) + 0.5f, 0.0f, 255.0f);

                #ifndef NDEBUG
                    {
                    float cp = s_cosPhi[phi_quant];
                    float sp = s_sinPhi[phi_quant];
                    float ct = s_cosTheta[theta_quant];
                    float st = s_sinTheta[theta_quant];

                    Vector3 orig{ x, y, cos_theta };
                    if(orig.x || orig.y || orig.z)
                    {
                        Vector3 recons{ cp*st, sp*st, ct };
                        TGE_ASSERT(fabsf(Dot(orig, recons)) > 0.9, "Invalid length");
                    }
                    }
                #endif

                    out_values[i*sizeof(uint16_t) + 0] = theta_quant;
                    out_values[i*sizeof(uint16_t) + 1] = phi_quant;
                }
            } break;
            /*
            case GridDataType::Float16:
            {
            
            } break;
            */
            case GridDataType::UInt8:
            {
                
            } break;
            default:
            {
                Log(LogLevel::Error, "Unsupported file format");
                return false;
            }
            }
        } break;
        default:
        {
            Log(LogLevel::Error, "Unsupported file format");
            return false;
        }
        }
    }

    return true;
}


VolumeRoot* ParseVolumeHierarchy(const std::string& name)
{
    std::string vol_descr = name + "_description.vol";

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

    auto* volumes = volume_root->Volumes = new Volume[grid_dims];
    memset(volumes, 0, grid_dims*sizeof(Volume));

    size_t total_voxel_size = 0;
    for(auto& param : GridParams)
    {
        total_voxel_size += GetGridDataTypeSize(param.Type)*param.Channels;
    }

    while(root_fs.good())
    {
        Box block;
        root_fs.read(reinterpret_cast<char*>(&block), sizeof(block));

        size_t data_offset = 0;

        TGE_ASSERT(block.X < volume_root->Dimensions.X &&
                   block.Y < volume_root->Dimensions.Y &&
                   block.Z < volume_root->Dimensions.Z, "Out of bounds");
        auto& volume = volumes[(block.Z * volume_root->Dimensions.Y + block.Y) * volume_root->Dimensions.X + block.X];

        for(auto& params : GridParams)
        {
            std::stringstream ss; // yeah, i know
            ss << name << "_" << std::setw(3) << std::setfill('0') << block.X << "_"
                              << std::setw(3) << std::setfill('0') << block.Y << "_"
                              << std::setw(3) << std::setfill('0') << block.Z;
            std::string filename = ss.str() + params.DefaultSuffix;

            size_t data_size;
            Vector3 aabb1, aabb2;
            ParseVolumeElement(filename.c_str(), params.Type, params.Channels, total_voxel_size, &volume, data_offset, &data_size, &aabb1, &aabb2);

            data_offset += data_size;
        }
    }

    return volume_root.release();
}

VolumeRoot* ParseVolume(const std::string& density, const std::string& orientation)
{
    size_t total_voxel_size = 0;
    for(auto& param : GridParams)
    {
        total_voxel_size += GetGridDataTypeSize(param.Type)*param.Channels;
    }

    std::unique_ptr<VolumeRoot> volume_root(new VolumeRoot);
    volume_root->Volumes = new Volume[1];
    volume_root->Dimensions.X = volume_root->Dimensions.Y = volume_root->Dimensions.Z = 1;

    auto& volume = volume_root->Volumes[0];

    size_t data_size;
    ParseVolumeElement(density.c_str(), GridParams[0].Type, GridParams[0].Channels, total_voxel_size, &volume, 0, &data_size, &volume_root->MinCorner, &volume_root->MaxCorner);

    Vector3 aabb1, aabb2;
    ParseVolumeElement(orientation.c_str(), GridParams[1].Type, GridParams[1].Channels, total_voxel_size, &volume, data_size, &data_size, &aabb1, &aabb2);
    TGE_ASSERT(volume_root->MinCorner == aabb1 && volume_root->MaxCorner == aabb2, "Invalid bounding boxes");

    return volume_root.release();
}
}

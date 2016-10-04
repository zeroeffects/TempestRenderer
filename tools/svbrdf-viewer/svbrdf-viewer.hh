/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2016 Zdravko Velinov
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

#include "tempest/utils/config.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/threads.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"

#include <cstdint>

#ifndef NDEBUG
//#   define CPU_DEBUG
#endif

#ifndef DISABLE_CUDA
#   define CUDA_ACCELERATED 1
#endif

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#   define EXECUTE_PARALLEL_FOR_LOOP_2D Tempest::ExecuteParallelForLoop2DGPU
#   define EXECUTE_PARALLEL_FOR_LOOP Tempest::ExecuteParallelForLoopGPU

const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#   define EXECUTE_PARALLEL_FOR_LOOP_2D Tempest::ExecuteParallelForLoop2DCPU
#   define EXECUTE_PARALLEL_FOR_LOOP Tempest::ExecuteParallelForLoopCPU

const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;
#endif

struct SGGXMapCPU
{
    Tempest::TexturePtr DiffuseMap,
                        SpecularMap,
                        BasisMap,
                        ScaleMap;
};

struct SGGXMapBoundConst
{
    const void* DiffuseMap,
              * SpecularMap,
              * BasisMap,
              * ScaleMap;
};

struct SGGXMapBound
{
    void* DiffuseMap,
        * SpecularMap,
        * BasisMap,
        * ScaleMap;
};

struct DebugNormalsMaterial: public Tempest::RTSpatiallyVaryingEmitter
{
    const void* DiffuseMap;
    uint32_t    Width,
                Height;
};

enum class DebugMode
{
    DebugTangent,
    DebugBinormal,
    DebugNormal
};

DebugNormalsMaterial* CreateDebugMaterial(DebugMode mode);

class ThreadPool;

void MixTextures(uint32_t id, Tempest::ThreadPool& pool,
                 SGGXMapBoundConst* sggx_maps, uint32_t index0, uint32_t index1, SGGXMapBound& sggx_mix_surfaces,
                 Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                 Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                 Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                 Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height,
                 float t_mix);

void CopyTextures(uint32_t id, Tempest::ThreadPool& pool,
                  SGGXMapBoundConst* sggx_maps, uint32_t index0, SGGXMapBound& sggx_mix_surfaces,
                  Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                  Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                  Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                  Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height);
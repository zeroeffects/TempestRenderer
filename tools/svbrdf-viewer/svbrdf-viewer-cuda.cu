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

#include "svbrdf-viewer.hh"
#include "tempest/compute/compute-convenience.hh"
#include "tempest/graphics/custom-samplers.hh"

#include <cuda_runtime_api.h>

struct LinearMixEvaluatorRGBA
{
    unsigned            Width;
    unsigned            ElementSize;
    Tempest::DataFormat Format;
    void              * MixSurface;
    const void        * Data0, 
                      * Data1;
    float               MixCoefficient;
    
    EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto sample0 = Tempest::RGBAExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data0) + (x + y*Width)*ElementSize);
        auto sample1 = Tempest::RGBAExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data1) + (x + y*Width)*ElementSize);

        auto sample = sample0*(1.0f - MixCoefficient) + sample1*MixCoefficient;

        Tempest::Surface2DWrite(sample, MixSurface, x*sizeof(Tempest::Vector4), y);
    }
};

struct LinearMixEvaluatorRG
{
    unsigned            Width;
    unsigned            ElementSize;
    Tempest::DataFormat Format;
    void              * MixSurface;
    const void        * Data0, 
                      * Data1;
    float               MixCoefficient;
    
    EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto sample0 = Tempest::RGExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data0) + (x + y*Width)*ElementSize);
        auto sample1 = Tempest::RGExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data1) + (x + y*Width)*ElementSize);

        auto sample = sample0*(1.0f - MixCoefficient) + sample1*MixCoefficient;

        Tempest::Surface2DWrite(sample, MixSurface, x*sizeof(Tempest::Vector2), y);
    }
};


struct CopyEvaluatorRGBA
{
    unsigned            Width;
    unsigned            ElementSize;
    Tempest::DataFormat Format;
    void              * MixSurface;
    const void        * Data;
    
    EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto sample = Tempest::RGBAExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data) + (x + y*Width)*ElementSize);
        Tempest::Surface2DWrite(sample, MixSurface, x*sizeof(Tempest::Vector4), y);
    }
};

struct CopyEvaluatorRG
{
    unsigned            Width;
    unsigned            ElementSize;
    Tempest::DataFormat Format;
    void              * MixSurface;
    const void        * Data;
    
    EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        auto sample = Tempest::RGExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data) + (x + y*Width)*ElementSize);
        Tempest::Surface2DWrite(sample, MixSurface, x*sizeof(Tempest::Vector2), y);
    }
};

struct SlerpMixEvaluator
{
    unsigned            Width;
    unsigned            ElementSize;
    Tempest::DataFormat Format;
    void              * MixSurface;
    const void        * Data0,
                      * Data1;
    float               MixCoefficient;

    EXPORT_CUDA void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        Tempest::Quaternion sample0, sample1;

        sample0.V4 = Tempest::RGBAExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data0) + (x + y*Width)*ElementSize);
        sample1.V4 = Tempest::RGBAExtractor::extract(Format, reinterpret_cast<const uint8_t*>(Data1) + (x + y*Width)*ElementSize);

        auto sample = Tempest::Slerp(sample0, sample1, MixCoefficient);

        Tempest::Surface2DWrite(sample, MixSurface, x*sizeof(Tempest::Quaternion), y);
    }
};

EXPORT_CUDA Tempest::Spectrum DebugTangents(const Tempest::SampleData& sample_data)
{
    auto material = static_cast<const DebugNormalsMaterial*>(sample_data.Material);
    Tempest::Quaternion quat;
    #ifdef LINEAR_SAMPLING
        quat.V4 = Tempest::SampleRGBA(material->DiffuseMap, sample_data.TexCoord);
    #else
        quat = Tempest::SampleQuaternionSlerp(material->DiffuseMap, material->Width, material->Height, sample_data.TexCoord);
    #endif
    auto norm = Tempest::ToTangent(quat);
    return Tempest::RGBToSpectrum(Tempest::ConvertSRGBToLinear(norm*0.5f + 0.5f));
}

EXPORT_CUDA Tempest::Spectrum DebugBinormals(const Tempest::SampleData& sample_data)
{
    auto material = static_cast<const DebugNormalsMaterial*>(sample_data.Material);
    Tempest::Quaternion quat;
    #ifdef LINEAR_SAMPLING
        quat.V4 = Tempest::SampleRGBA(material->DiffuseMap, sample_data.TexCoord);
    #else
        quat = Tempest::SampleQuaternionSlerp(material->DiffuseMap, material->Width, material->Height, sample_data.TexCoord);
    #endif
    auto norm = Tempest::ToBinormal(quat);
    return Tempest::RGBToSpectrum(Tempest::ConvertSRGBToLinear(norm*0.5f + 0.5f));
}

EXPORT_CUDA Tempest::Spectrum DebugNormals(const Tempest::SampleData& sample_data)
{
    auto material = static_cast<const DebugNormalsMaterial*>(sample_data.Material);
    Tempest::Quaternion quat;
    static_assert(sizeof(quat) == sizeof(Tempest::Vector4), "Invalid quaternion size");
    #ifdef LINEAR_SAMPLING
        quat.V4 = Tempest::SampleRGBA(material->DiffuseMap, sample_data.TexCoord);
    #else
        quat = Tempest::SampleQuaternionSlerp(material->DiffuseMap, material->Width, material->Height, sample_data.TexCoord);
    #endif
    auto norm = Tempest::ToNormal(quat);
    return Tempest::RGBToSpectrum(Tempest::ConvertSRGBToLinear(norm*0.5f + 0.5f));
}

DebugNormalsMaterial* CreateDebugMaterial(DebugMode mode)
{
    auto debug_material = new DebugNormalsMaterial;
    debug_material->Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    debug_material->MaterialSize = sizeof(DebugNormalsMaterial);
    switch(mode)
    {
    case DebugMode::DebugTangent: debug_material->EmitFunction = DebugTangents; break;
    case DebugMode::DebugBinormal: debug_material->EmitFunction = DebugBinormals; break;
    default:
    case DebugMode::DebugNormal: debug_material->EmitFunction = DebugNormals; break;
    }

    return debug_material;
}

#if defined(CUDA_ACCELERATED) && !defined(CPU_DEBUG)
void MixTextures(uint32_t id, Tempest::ThreadPool& pool,
                 SGGXMapBoundConst* sggx_maps, uint32_t index0, uint32_t index1, SGGXMapBound& sggx_mix_surfaces,
                 Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                 Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                 Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                 Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height,
                 float t_mix)
{
    auto& sggx_slice0 = sggx_maps[index0],
        & sggx_slice1 = sggx_maps[index1];

    auto diffuse0 = sggx_slice0.DiffuseMap;
    auto diffuse1 = sggx_slice1.DiffuseMap;

    LinearMixEvaluatorRGBA diffuse_evaluator{ mix_diffuse_width, Tempest::DataFormatElementSize(diffuse_fmt), diffuse_fmt, sggx_mix_surfaces.DiffuseMap, diffuse0, diffuse1, t_mix };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_diffuse_width, mix_diffuse_height, diffuse_evaluator);

    auto specular0 = sggx_slice0.SpecularMap;
    auto specular1 = sggx_slice1.SpecularMap;
                
    LinearMixEvaluatorRGBA specular_evaluator{ mix_specular_width, Tempest::DataFormatElementSize(specular_fmt), specular_fmt, sggx_mix_surfaces.SpecularMap, specular0, specular1, t_mix };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_specular_width, mix_specular_height, specular_evaluator);

    auto basis0 = sggx_slice0.BasisMap;
    auto basis1 = sggx_slice1.BasisMap;

    SlerpMixEvaluator basis_evaluator{ mix_basis_width, Tempest::DataFormatElementSize(basis_fmt), basis_fmt, sggx_mix_surfaces.BasisMap, basis0, basis1, t_mix };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_basis_width, mix_basis_height, basis_evaluator);

    auto scale0 = sggx_slice0.ScaleMap;
    auto scale1 = sggx_slice1.ScaleMap;

    LinearMixEvaluatorRG scale_evaluator{ mix_scale_width, Tempest::DataFormatElementSize(scale_fmt), scale_fmt, sggx_mix_surfaces.ScaleMap, scale0, scale1, t_mix };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_scale_width, mix_scale_height, scale_evaluator);
}

void CopyTextures(uint32_t id, Tempest::ThreadPool& pool,
                  SGGXMapBoundConst* sggx_maps, uint32_t index0, SGGXMapBound& sggx_mix_surfaces,
                  Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                  Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                  Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                  Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height)
{
    auto diffuse0 = sggx_maps[index0].DiffuseMap;

    CopyEvaluatorRGBA diffuse_evaluator{ mix_diffuse_width, Tempest::DataFormatElementSize(diffuse_fmt), diffuse_fmt, sggx_mix_surfaces.DiffuseMap, diffuse0 };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_diffuse_width, mix_diffuse_height, diffuse_evaluator);

    auto specular0 = sggx_maps[index0].SpecularMap;
    CopyEvaluatorRGBA specular_evaluator{ mix_specular_width, Tempest::DataFormatElementSize(specular_fmt), specular_fmt, sggx_mix_surfaces.SpecularMap, specular0 };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_specular_width, mix_specular_height, specular_evaluator);

    auto basis0 = sggx_maps[index0].BasisMap;
    CopyEvaluatorRGBA basis_evaluator{  mix_basis_width, Tempest::DataFormatElementSize(basis_fmt), basis_fmt, sggx_mix_surfaces.BasisMap, basis0 };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_basis_width, mix_basis_height, basis_evaluator);

    auto scale0 = sggx_maps[index0].ScaleMap;
    CopyEvaluatorRG scale_evaluator{ mix_scale_width, Tempest::DataFormatElementSize(scale_fmt), scale_fmt, sggx_mix_surfaces.ScaleMap, scale0 };
    EXECUTE_PARALLEL_FOR_LOOP_2D(id, pool, mix_scale_width, mix_scale_height, scale_evaluator);
}
#else
void MixTextures(uint32_t id, Tempest::ThreadPool& pool,
                 SGGXMapBoundConst* sggx_maps, uint32_t index0, uint32_t index1, SGGXMapBound& sggx_mix_surfaces,
                 Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                 Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                 Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                 Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height,
                 float t_mix)
{
    auto& sggx_slice0 = sggx_maps[index0],
        & sggx_slice1 = sggx_maps[index1];

    auto diffuse0 = sggx_slice0.DiffuseMap;
    auto diffuse1 = sggx_slice1.DiffuseMap;

    LinearMixEvaluatorRGBA diffuse_evaluator{ mix_diffuse_width, Tempest::DataFormatElementSize(diffuse_fmt), diffuse_fmt, sggx_mix_surfaces.DiffuseMap, diffuse0, diffuse1, t_mix };
    auto diffuse_task = Tempest::CreateParallelForLoop2D(mix_diffuse_width, mix_diffuse_height, 64, diffuse_evaluator);
    pool.enqueueTask(&diffuse_task);

    auto specular0 = sggx_slice0.SpecularMap;
    auto specular1 = sggx_slice1.SpecularMap;
                
    LinearMixEvaluatorRGBA specular_evaluator{ mix_specular_width, Tempest::DataFormatElementSize(specular_fmt), specular_fmt, sggx_mix_surfaces.SpecularMap, specular0, specular1, t_mix };
    auto specular_task = Tempest::CreateParallelForLoop2D(mix_specular_width, mix_specular_height, 64, specular_evaluator);
    pool.enqueueTask(&specular_task);

    auto basis0 = sggx_slice0.BasisMap;
    auto basis1 = sggx_slice1.BasisMap;

    SlerpMixEvaluator basis_evaluator{ mix_basis_width, Tempest::DataFormatElementSize(basis_fmt), basis_fmt, sggx_mix_surfaces.BasisMap, basis0, basis1, t_mix };
    auto basis_task = Tempest::CreateParallelForLoop2D(mix_basis_width, mix_basis_height, 64, basis_evaluator);
    pool.enqueueTask(&basis_task);

    auto scale0 = sggx_slice0.ScaleMap;
    auto scale1 = sggx_slice1.ScaleMap;

    LinearMixEvaluatorRG scale_evaluator{ mix_scale_width, Tempest::DataFormatElementSize(scale_fmt), scale_fmt, sggx_mix_surfaces.ScaleMap, scale0, scale1, t_mix };
    auto scale_task = Tempest::CreateParallelForLoop2D(mix_scale_width, mix_scale_height, 64, scale_evaluator);
    pool.enqueueTask(&scale_task);
   
    pool.waitAndHelp(id, &scale_task);
    pool.waitAndHelp(id, &basis_task);
    pool.waitAndHelp(id, &specular_task);
    pool.waitAndHelp(id, &diffuse_task);
}

void CopyTextures(uint32_t id, Tempest::ThreadPool& pool,
                  SGGXMapBoundConst* sggx_maps, uint32_t index0, SGGXMapBound& sggx_mix_surfaces,
                  Tempest::DataFormat diffuse_fmt, uint32_t mix_diffuse_width, uint32_t mix_diffuse_height,
                  Tempest::DataFormat specular_fmt, uint32_t mix_specular_width, uint32_t mix_specular_height,
                  Tempest::DataFormat basis_fmt, uint32_t mix_basis_width, uint32_t mix_basis_height,
                  Tempest::DataFormat scale_fmt, uint32_t mix_scale_width, uint32_t mix_scale_height)
{
    auto diffuse0 = sggx_maps[index0].DiffuseMap;

    CopyEvaluatorRGBA diffuse_evaluator{ mix_diffuse_width, Tempest::DataFormatElementSize(diffuse_fmt), diffuse_fmt, sggx_mix_surfaces.DiffuseMap, diffuse0 };
    auto diffuse_task = Tempest::CreateParallelForLoop2D(mix_diffuse_width, mix_diffuse_height, 64, diffuse_evaluator);
    pool.enqueueTask(&diffuse_task);

    auto specular0 = sggx_maps[index0].SpecularMap;
    CopyEvaluatorRGBA specular_evaluator{ mix_specular_width, Tempest::DataFormatElementSize(specular_fmt), specular_fmt, sggx_mix_surfaces.SpecularMap, specular0 };
    auto specular_task = Tempest::CreateParallelForLoop2D(mix_specular_width, mix_specular_height, 64, specular_evaluator);
    pool.enqueueTask(&specular_task);

    auto basis0 = sggx_maps[index0].BasisMap;
    CopyEvaluatorRGBA basis_evaluator{ mix_basis_width, Tempest::DataFormatElementSize(basis_fmt), basis_fmt, sggx_mix_surfaces.BasisMap, basis0 };
    auto basis_task = Tempest::CreateParallelForLoop2D(mix_basis_width, mix_basis_height, 64, basis_evaluator);
    pool.enqueueTask(&basis_task);

    auto scale0 = sggx_maps[index0].ScaleMap;
    CopyEvaluatorRG scale_evaluator{ mix_scale_width, Tempest::DataFormatElementSize(scale_fmt), scale_fmt, sggx_mix_surfaces.ScaleMap, scale0 };
    auto scale_task = Tempest::CreateParallelForLoop2D(mix_scale_width, mix_scale_height, 64, scale_evaluator);
    pool.enqueueTask(&scale_task);

    pool.waitAndHelp(id, &scale_task);
    pool.waitAndHelp(id, &basis_task);
    pool.waitAndHelp(id, &specular_task);
    pool.waitAndHelp(id, &diffuse_task);
}
#endif
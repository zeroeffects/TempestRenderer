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

#ifndef _TEMPEST_SGGX_FITTING_HH_
#define _TEMPEST_SGGX_FITTING_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/quaternion.hh"

#include <memory>

namespace Tempest
{
union Vector4;
union Point2;
struct BTF;
class ThreadPool;
class TimeQuery;
class Texture;
}

const float SphereGap = 0.1f;
const float SphereRadius = 4.0f;

const uint32_t CommonSampleCount = 512;
const uint32_t EigenVectorSphereSamples = CommonSampleCount*CommonSampleCount;
const uint32_t EigenVectorPlaneSamples = CommonSampleCount;
const uint32_t NDRReduceSphereSamples = 64*64;
const uint32_t ConvertSamples = 1024;
const uint32_t CurveFitSteps = 1024;

const uint32_t NDFTextureResolution = 64;

union OptimizationParameters
{
    struct ParametersStruct
    {
        Tempest::Vector2 StandardDeviation;
        float            Specular;
        float            Diffuse;
        Tempest::Vector3 Euler;
    } Parameters;
    float ParametersArray[7];
};

union OptimizationParametersSpecular
{
    struct ParametersStruct
    {
        Tempest::Vector2 StandardDeviation;
        float            Specular;
        Tempest::Vector3 Euler;
    } Parameters;
    float ParametersArray[6];
};
#undef None
enum class FilteringTechnique
{
    None,
    Gaussian
};

enum class BasisExtractStrategy
{
    PCAHemisphere,
    PCAPlaneProject,
    PhotometricNormals
};

enum
{
    LSF_OPTION_CUDA = 1 << 0,
    LSF_OPTON_DIFFUSE = 1 << 1,
    LSF_OPTION_DISABLE_DIRECTION_FITTING = 1 << 2,
    LSF_OPTION_FILTER_DOWNSAMPLING = 1 << 3,
    LSF_OPTION_FIT_TOP = 1 << 4
};

struct LeastSquaresFitOptions
{
    uint32_t           MultiScatteringBounceCount;
    float              Fresnel;

    uint32_t           DownSampleLightView;
    uint32_t           Flags = 0;
};

enum
{
    PIPELINE_OPTION_CUDA = 1 << 0,
    PIPELINE_OPTION_MAXIMIZE_NORMAL_PROJECTION = 1 << 1,
    PIPELINE_OPTION_REFIT = 1 << 2,
    PIPELINE_OPTION_NDF_TOP = 1 << 3,
    PIPELINE_OPTION_PRINT_PER_PIXEL = 1 << 4,
};

struct FitPipelineOptions
{
    FilteringTechnique   Filter;
    BasisExtractStrategy BasisExtract;
    uint32_t             KernelRadius,
                         Flags = 0;
};

void LeastSquaresFitSGGX(uint32_t id, Tempest::ThreadPool& pool, const Tempest::BTF* btf, float* lv_lum_slice, const LeastSquaresFitOptions& opts,
                         const OptimizationParameters& parameters, OptimizationParameters* opt_parameters, float* rmse);

void DisplayDistributions(uint32_t image_width, uint32_t image_height,
                          const Tempest::Vector2& sggx_stddev, const Tempest::Vector3& albedo, const Tempest::Vector3& specular, const Tempest::Vector4& sggx_basis,
                          const Tempest::BTF* btf_ptr, const float* lv_lum_slice, uint32_t btf_start_x, uint32_t btf_start_y,
                          const Tempest::Texture& ndf_texture);

void DisplayDistributionMap(uint32_t image_width, uint32_t image_height, const Tempest::Point2* points,
                            const Tempest::Vector2* sggx_stddev_map, const Tempest::Vector3* specular_map, const Tempest::Quaternion* sggx_basis_map, uint32_t sample_count,
                            const Tempest::BTF* btf_ptr);

struct SGGXParameters
{
    Tempest::Vector2    StandardDeviation;
    Tempest::Vector3    Specular;
    Tempest::Vector3    Diffuse;
    Tempest::Quaternion Orientation;
    float               RMSE;
};

struct CudaResourceDeleter
{
    void operator()(void* data);
};

void NullTestFillBTF(const LeastSquaresFitOptions& opts, const OptimizationParameters& sggx_parameters, Tempest::BTF* btf);

bool SymmetryTest(const Tempest::BTF* btf, const OptimizationParameters& sggx_parameters);

class SVBRDFFitPipeline
{
    std::unique_ptr<Tempest::Texture>               m_NDFTexture;
    std::unique_ptr<float[]>                        m_LuminanceSlice;
    std::unique_ptr<float[]>                        m_AveragedProbabilities;
    std::unique_ptr<Tempest::Vector3[]>             m_Lights;

    const Tempest::BTF*               m_BTFCPU;

#ifndef DISABLE_CUDA
    std::unique_ptr<float, CudaResourceDeleter>     m_GPULuminanceSlice;
    std::unique_ptr<float, CudaResourceDeleter>     m_GPUNDFData;

    const Tempest::BTF*               m_BTFGPU;
#endif

    Tempest::Spectrum                               m_AverageSpectrum;
public:
#ifdef DISABLE_CUDA
    SVBRDFFitPipeline(const Tempest::BTF* btf_cpu);
#else
    SVBRDFFitPipeline(const Tempest::BTF* btf_cpu, const Tempest::BTF* btf_gpu);
#endif

    const Tempest::Texture* getLastNDFTexture() const { return m_NDFTexture.get(); }
    const float*            getLastLuminanceSlice() const { return m_LuminanceSlice.get(); }

    void cache(uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, FitPipelineOptions& pipeline_opts, Tempest::TimeQuery* timer);

    void fit(uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, 
             LeastSquaresFitOptions* opts, FitPipelineOptions& pipeline_opts, Tempest::TimeQuery* timer, SGGXParameters* out_parameters, bool reoptimize = false);
};

#endif // _TEMPEST_SGGX_FITTING_HH_

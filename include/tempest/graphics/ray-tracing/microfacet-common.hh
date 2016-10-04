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

#ifndef _MICROFACET_COMMON_HH_
#define _MICROFACET_COMMON_HH_

#include "tempest/math/vector3.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-common.hh"

#if defined(__CUDACC__) && defined(ILLUMINATION_MODEL_STATIC_IMPLEMENTATION)
#	define SPLIT_EXPORT_CUDA static __device__
#	define SPLIT_TABLE_EXPORT_CUDA __device__
#   define MODEL_NAMESPACE Cuda
#elif defined(__CUDACC__) && (defined(EXPORT_TABLE) || defined(ILLUMINATION_MODEL_IMPLEMENTATION)) && !defined(CPU_DEBUG)
#	define SPLIT_EXPORT_CUDA __device__
#	define SPLIT_TABLE_EXPORT_CUDA __device__
#   define MODEL_NAMESPACE Cuda
#else
#	define SPLIT_EXPORT_CUDA
#	define SPLIT_TABLE_EXPORT_CUDA
#   define MODEL_NAMESPACE Cpp
#endif

namespace Tempest
{
struct MicrofacetAngles
{
    float CosIncidentNorm;
    float CosOutgoingNorm;
    union
    {
        float CosIncidentMicroNorm;
        float CosOutgoingMicroNorm; // Because it is half-way angle
    };
    float CosMicroNormNorm;
};

struct MicrofacetAnglesAnisotropic
{
    float CosIncidentNorm;
    float CosOutgoingNorm;
    union
    {
        float CosIncidentMicroNorm;
        float CosOutgoingMicroNorm; // Because it is half-way angle
    };
    float CosMicroNormNorm;
	float CosMicroNormTangent;
	float CosMicroNormBinorm;
};


inline EXPORT_CUDA void ComputeMicrofacetAngles(const SampleData& sample_data, MicrofacetAngles* angles)
{
    auto half_vec = sample_data.IncidentLight + sample_data.OutgoingLight; // or microsurface normal
    NormalizeSelf(&half_vec);

    angles->CosIncidentNorm = Dot(sample_data.IncidentLight, sample_data.Normal);
    angles->CosOutgoingNorm = Dot(sample_data.OutgoingLight, sample_data.Normal);
    angles->CosIncidentMicroNorm = Dot(sample_data.IncidentLight, half_vec);
    angles->CosMicroNormNorm = Dot(sample_data.Normal, half_vec);
}

inline EXPORT_CUDA void ComputeMicrofacetAnglesAnisotropic(const SampleData& sample_data, MicrofacetAnglesAnisotropic* angles)
{
    auto half_vec = sample_data.IncidentLight + sample_data.OutgoingLight; // or microsurface normal
    NormalizeSelf(&half_vec);

    angles->CosIncidentNorm = Dot(sample_data.IncidentLight, sample_data.Normal);
    angles->CosOutgoingNorm = Dot(sample_data.OutgoingLight, sample_data.Normal);
    angles->CosIncidentMicroNorm = Dot(sample_data.IncidentLight, half_vec);
    angles->CosMicroNormNorm = Dot(sample_data.Normal, half_vec);

    Vector2 tangent_plane_angle{ Dot(sample_data.Tangent, half_vec), Dot(sample_data.Binormal, half_vec) };
    NormalizeSelf(&tangent_plane_angle);

	angles->CosMicroNormTangent = tangent_plane_angle.x;
	angles->CosMicroNormBinorm = tangent_plane_angle.y;
}

inline EXPORT_CUDA float ComputeRoughnessProjectedOnDirectionSq(float geom_factor_x, float geom_factor_y, float cos_tan_out, float sin_tan_out)
{
    float tan_term = geom_factor_x*cos_tan_out;
    float binorm_term = geom_factor_y*sin_tan_out;
    return fabsf(tan_term*tan_term + binorm_term*binorm_term);
}

inline EXPORT_CUDA float SpecularPowerToGeometryFactorSq(float specular_power)
{
    return 2.0f/(specular_power + 2.0f);
}
}

#endif
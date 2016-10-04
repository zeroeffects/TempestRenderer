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

#ifndef _TEMPEST_BTF_HH_
#define _TEMPEST_BTF_HH_

#include "tempest/utils/logging.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/matrix3.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/math/triangle.hh"
#include "tempest/math/intersect.hh"
#include "tempest/utils/memory.hh"
#include "tempest/utils/logging.hh"
#include "tempest/mesh/lbvh2.hh"
#include "tempest/mesh/sslbvh2.hh"

#include <memory>

namespace Tempest
{
class Path;
class ThreadPool;

#define BTF_MAX_CHANNEL_COUNT 3

#ifndef __CUDA_ARCH__
struct BTFCommonHeader
{
    uint32_t Size;
    uint32_t Version;
    char     MeasurementSetup[80];
    char     ImageSensor[80];
    char     LightSource[80];
    float    PPMM;
    Vector3  RGB;
};

struct BTFExtra
{
	BTFCommonHeader             Header;
    std::string                 XMLString;
	std::vector<std::string>    Channels;
    std::vector<Matrix3>        Rotations;
};
#else
struct BTFExtra;
#endif

struct Edge
{
    uint32_t Index0,
             Index1;
    uint32_t Triangle;
    float    Angle0;
    float    Angle1;
};

struct BTF
{
    uint32_t                    ConsineFlag = false;
    uint32_t				    ChannelCount = 3;
    uint32_t                    Width = 0,
                                Height = 0;
    uint32_t                    DynamicRangeReduction = false;
    struct
    {
        uint32_t                Width = 0;
        uint32_t                Height = 0;
    }                           HeightMapSize;
    uint16_t*					HeightMap = nullptr;

    uint32_t                    ColorModel = 0;
    Vector3                     ColorMean;
    Matrix3                     ColorTransform;

    uint32_t                    RowCount = 0,
                                ColumnCount = 0,
                                DataSize = 0;

    uint8_t*					LeftSingularU = nullptr,
           *                    RightSingularSxV = nullptr;

	uint64_t					LeftSingularUSize = 0,
								RightSingularSxVSize = 0;

    Vector2*					LightsParabolic = nullptr;

    uint32_t                    LightCount = 0;

    uint32_t*                   LightIndices = nullptr;
    uint32_t                    LightTriangleCount = 0;

    uint32_t                    UElementStride = 0,
                                SxVElementStride = 0;

	uint32_t*					Offsets = nullptr;
	uint32_t*					ComponentCounts = nullptr;

    SimpleStacklessLBVH2Node<AABB2>* LightBVH = nullptr;

    Edge*                       Edges = nullptr;
    uint32_t                    EdgeCount = 0;

    /*
    std::unique_ptr<Vector2[]>  m_ViewsParabolic;

    uint32_t                    m_ViewCount = 0;

    uint32_t*                   m_ViewIndices = nullptr;
    uint32_t                    m_ViewTriangleCount = 0;
    
    std::unique_ptr<SimpleStacklessLBVH2Node[]> m_ViewBVH;
    */
};

BTF* LoadBTF(const Path& file_path, BTFExtra* extra = nullptr);

BTF* CutBTF(const BTF* source_btf, uint32_t x, uint32_t y, uint32_t width, uint32_t height);

BTF* CreateDummyBTF(uint32_t light_count);

void DestroyCPUBTF(BTF* btf);
void DestroyGPUBTF(BTF* btf);

#ifndef DISABLE_CUDA
BTF* CreateGPUBTF(BTF* cpu_btf);
#endif

inline EXPORT_CUDA bool BTFFetchLightViewDirection(const BTF* btf, const Vector3& light, const Vector3& view,
												   uint32_t* light_prim_id, Vector3* light_barycentric,
												   uint32_t* view_prim_id, Vector3* view_barycentric);

inline EXPORT_CUDA Spectrum BTFSampleSpectrum(const BTF* btf, const Vector3& light, const Vector3& view, const Vector2& tc);
inline EXPORT_CUDA Spectrum BTFFetchPixelSampleLightViewSpectrum(const BTF* btf, const Vector3& light, const Vector3& view, uint32_t x, uint32_t y);
inline EXPORT_CUDA Spectrum BTFFetchPixelSampleLightViewSpectrum(const BTF* btf, uint32_t light_prim_id, const Vector3& light_barycentric, uint32_t view_prim_id, const Vector3& view_barycentric, uint32_t x, uint32_t y);
inline EXPORT_CUDA Spectrum BTFFetchSpectrum(const BTF* btf, uint32_t light_vert, uint32_t view_vert, uint32_t x, uint32_t y);
inline EXPORT_CUDA Spectrum BTFFetchSpectrum(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx);
inline EXPORT_CUDA void BTFFetchChannelsSingleFloat(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** multi_chan_result);
inline void BTFFetchChannelsSIMD(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** multi_chan_result);
inline void BTFEvaluateMatrixHalfFloatToFloat(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** result);

#if defined(__CUDA_ARCH__) || defined(LINUX)
#	define BTFFetchChannels BTFFetchChannelsSingleFloat
#else
#	define BTFFetchChannels BTFFetchChannelsSIMD
#endif

template<class T, class TOp>
inline EXPORT_CUDA void BTFEvaluateMatrix(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** result);

struct BTFDeleter
{
	inline void operator()(BTF* btf) { DestroyCPUBTF(btf); }
};

typedef std::unique_ptr<BTF, BTFDeleter> BTFPtr;

struct GPUBTFDeleter
{
	inline void operator()(BTF* btf) { DestroyGPUBTF(btf); }
};

typedef std::unique_ptr<BTF, GPUBTFDeleter> BTFGPUPtr;

inline float* ComputeLightViewSampleWeights(BTF* btf_ptr)
{
	auto light_count = btf_ptr->LightCount;

	std::unique_ptr<float[]> direction_contrib(new float[light_count]);
    memset(direction_contrib.get(), 0, light_count*sizeof(float));

	float* weights = new float[light_count*light_count];

    float total_area = 0.0f;

    for(uint32_t light_idx = 0, light_idx_end = btf_ptr->LightTriangleCount; light_idx < light_idx_end;)
    {
        auto i0 = btf_ptr->LightIndices[light_idx++];
        auto i1 = btf_ptr->LightIndices[light_idx++];
        auto i2 = btf_ptr->LightIndices[light_idx++];

        Tempest::Vector3 v0 = Tempest::ParabolicToCartesianCoordinates(btf_ptr->LightsParabolic[i0]);
        Tempest::Vector3 v1 = Tempest::ParabolicToCartesianCoordinates(btf_ptr->LightsParabolic[i1]);
        Tempest::Vector3 v2 = Tempest::ParabolicToCartesianCoordinates(btf_ptr->LightsParabolic[i2]);

        auto area = TriangleArea(v0, v1, v2);

        float vert_contrib = area/3.0f;

        direction_contrib[i0] += vert_contrib;
        direction_contrib[i1] += vert_contrib;
        direction_contrib[i2] += vert_contrib;

        total_area += area;
    }

    for(uint32_t view_idx = 0; view_idx < light_count; ++view_idx)
        for(uint32_t light_idx = 0; light_idx < light_count; ++light_idx)
        {
            float contrib = direction_contrib[light_idx]*direction_contrib[view_idx]/(total_area*total_area);
            weights[view_idx*light_count + light_idx] = contrib;
        }

	return weights;
}

template<class T>
inline EXPORT_CUDA T BTFSampleLuminanceSlice(const BTF* btf, const Vector3& light, const Vector3& view, const T* lum_slice)
{
    uint32_t light_prim_id, view_prim_id;
    Tempest::Vector3 light_barycentric, view_barycentric;
    auto intersect = BTFFetchLightViewDirection(btf, light, view, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
    if(!intersect)
        return {};

    T final_result{};
    for(uint32_t view_idx = 0; view_idx < 3; ++view_idx)
    {
        auto view_vert_idx = btf->LightIndices[view_prim_id*3 + view_idx];
        float weight_view = Array(view_barycentric)[(view_idx + 2) % 3];

        for(uint32_t light_idx = 0; light_idx < 3; ++light_idx)
        {
            auto light_vert_idx = btf->LightIndices[light_prim_id*3 + light_idx];
            float weight_light = Array(light_barycentric)[(light_idx + 2) % 3];

            uint32_t lv_idx = view_vert_idx*btf->LightCount + light_vert_idx;

            auto spec_value = lum_slice[lv_idx];
            final_result += weight_light*weight_view*spec_value;
        }
    }

    return final_result;
}

void BTFParallelExtractLuminanceSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, float** out_luminance_slice);
void BTFParallelExtractLuminanceSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, float** out_luminance_slice, Vector3* out_avg_spec);
void BTFParallelExtractRGBSlice(const BTF* btf_cpu, uint32_t id, ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, Vector3** out_spectrum_slice);

struct EdgeComparison
{
    inline EXPORT_CUDA bool operator()(const Edge& lhs, float value)
    {
        return lhs.Angle1 < value;
    };
};

inline EXPORT_CUDA void IntersectEdge(const BTF* btf, const Vector3& dir, IntersectTriangleQuery2D& intersect_tri)
{
    float angle = atan2f(dir.y, dir.x);

    size_t count = btf->EdgeCount, step;
    auto iter_begin = btf->Edges,
         iter_first = iter_begin;
    while(count > 0)
    {
        step = count/2;
        auto iter = iter_first + step;
        if (iter->Angle1 < angle)
        {                 
          iter_first = ++iter;
          count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    if(iter_first == btf->Edges + btf->EdgeCount)
    {
        iter_first = btf->Edges;
    }


    auto tri_idx = intersect_tri.PrimitiveID = iter_first->Triangle;
    auto tri = btf->LightIndices + tri_idx*3;

    float ratio = (angle - iter_first->Angle0)/(iter_first->Angle1 - iter_first->Angle0);
    auto lights = btf->LightsParabolic;
    Vector2 partial_barycentric;

    auto i0 = tri[0];
    auto i1 = tri[1];
    auto i2 = tri[2];

    auto& v0 = lights[i0];
    auto& v1 = lights[i1];
    auto& v2 = lights[i2];

    auto point = GenericLinearInterpolate(lights[iter_first->Index0], lights[iter_first->Index1], ratio);

    TriangleBarycentricCoordinates(point, v0, v1, v2, &partial_barycentric);
    
    partial_barycentric = Vector2Clamp(partial_barycentric, 0.0f, 1.0f);
    float w = 1.0f - partial_barycentric.x - partial_barycentric.y;

    intersect_tri.BarycentricCoordinates = { partial_barycentric.x, partial_barycentric.y, w };
}

inline EXPORT_CUDA bool BTFFetchLightViewDirection(const BTF* btf, const Vector3& light, const Vector3& view,
												   uint32_t* light_prim_id, Vector3* light_barycentric,
												   uint32_t* view_prim_id, Vector3* view_barycentric)
{
    auto par_view = CartesianToParabolicCoordinates(view),
         par_light = CartesianToParabolicCoordinates(light);

    if(view.z < 0.0f || light.z < 0.0f)
        return false;

    IntersectTriangleQuery2D intersect_light_tri{ btf->LightsParabolic, btf->LightIndices };
    IntersectTriangleQuery2D intersect_view_tri;

    /*
    bool intersect = Tempest::IntersectTriangleList2D(par_light, btf->LightsParabolic, btf->LightCount, btf->LightIndices, btf->LightTriangleCount, &intersect_light_tri.PrimitiveID, &intersect_light_tri.BarycentricCoordinates);
    /*/

    bool intersect = Tempest::IntersectSSLBVHNodeSingle(btf->LightBVH, par_light, intersect_light_tri);
    //*/
        
    if(!intersect)
    {
        IntersectEdge(btf, light, intersect_light_tri);
    }

    *light_barycentric = intersect_light_tri.BarycentricCoordinates;
    *light_prim_id = intersect_light_tri.PrimitiveID;

    /*
    IntersectTriangle intersect_view_tri{ btf->ViewsParabolic ? btf->ViewsParabolic : btf->m_LightsParabolic, btf->ViewIndices ? btf->ViewIndices : btf->LightIndices };
    intersect = Tempest::IntersectTriangleList2D(par_view, btf->ViewsParabolic, btf->ViewCount, btf->ViewIndices, btf->ViewTriangleCount, &intersect_view_tri.PrimitiveID, &intersect_view_tri.BarycentricCoordinates);
    /*/
    #if 0
    if(m_ViewBVH)
    {
        intersect_view_tri = { btf->ViewsParabolic, btf->ViewIndices };
        intersect = Tempest::IntersectSSLBVHNodeSingle(btf->ViewBVH, par_view, intersect_view_tri);
    }
    else
    #endif
    {
        intersect_view_tri = { btf->LightsParabolic, btf->LightIndices };
        intersect = Tempest::IntersectSSLBVHNodeSingle(btf->LightBVH, par_view, intersect_view_tri);
    }
    //*/
    if(!intersect)
    {
        IntersectEdge(btf, view, intersect_view_tri);
    }

    *view_barycentric = intersect_view_tri.BarycentricCoordinates;
    *view_prim_id = intersect_view_tri.PrimitiveID;
    return true;
}

inline EXPORT_CUDA Spectrum BTFFetchPixelSampleLightViewSpectrum(const BTF* btf, const Vector3& light, const Vector3& view, uint32_t x, uint32_t y)
{
    Vector3 light_barycentric, view_barycentric;
    uint32_t light_prim_id, view_prim_id;
    auto intersect = BTFFetchLightViewDirection(btf, light, view, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
        
    if(!intersect)
        return {};

    return BTFFetchPixelSampleLightViewSpectrum(btf, light_prim_id, light_barycentric, view_prim_id, view_barycentric, x, y);
}

inline EXPORT_CUDA Spectrum BTFFetchPixelSampleLightViewSpectrum(const BTF* btf, uint32_t light_prim_id, const Vector3& light_barycentric, uint32_t view_prim_id, const Vector3& view_barycentric, uint32_t x, uint32_t y)
{
    #if 0
    auto view_indices = btf->ViewIndices;
    #else
    auto view_indices = btf->LightIndices;
    #endif
    Spectrum final_result{};
    for(uint32_t view_idx = 0; view_idx < 3; ++view_idx)
    {
        auto view_vert_idx = view_indices[view_prim_id*3 + view_idx];
        float weight_view = Array(view_barycentric)[(view_idx + 2) % 3];

        for(uint32_t light_idx = 0; light_idx < 3; ++light_idx)
        {
            auto light_vert_idx = btf->LightIndices[light_prim_id*3 + light_idx];
            float weight_light = Array(light_barycentric)[(light_idx + 2) % 3];

            auto spec_value = BTFFetchSpectrum(btf, light_vert_idx, view_vert_idx, x, y);
            final_result += weight_light*weight_view*spec_value;
        }
    }

    return final_result;
}

inline EXPORT_CUDA Spectrum BTFSampleSpectrum(const BTF* btf, const Vector3& light, const Vector3& view, const Vector2& tc)
{
    Vector3 light_barycentric, view_barycentric;
    uint32_t light_prim_id, view_prim_id;
    auto intersect = BTFFetchLightViewDirection(btf, light, view, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
        
    if(!intersect)
        return {};

    #if 0
    auto view_indices = btf->ViewIndices;
    #else
    auto view_indices = btf->LightIndices;
    #endif

    Vector2 tc_unorm{ btf->Width*tc.x, btf->Height*tc.y };

    uint32_t x0 = (uint32_t)(tc_unorm.x) % btf->Width;
    uint32_t y0 = (uint32_t)(tc_unorm.y) % btf->Height;
    uint32_t x1 = (x0 + 1) % btf->Width;
    uint32_t y1 = (y0 + 1) % btf->Height;

    Spectrum final_result{};
    for(uint32_t view_idx = 0; view_idx < 3; ++view_idx)
    {
        auto view_vert_idx = view_indices[view_prim_id*3 + view_idx];
        float weight_view = Array(view_barycentric)[(view_idx + 2) % 3];

        for(uint32_t light_idx = 0; light_idx < 3; ++light_idx)
        {
            auto light_vert_idx = btf->LightIndices[light_prim_id*3 + light_idx];
            float weight_light = Array(light_barycentric)[(light_idx + 2) % 3];

            auto c00 = BTFFetchSpectrum(btf, light_vert_idx, view_vert_idx, x0, y0);
            auto c01 = BTFFetchSpectrum(btf, light_vert_idx, view_vert_idx, x1, y0);
            auto c10 = BTFFetchSpectrum(btf, light_vert_idx, view_vert_idx, x0, y1);
            auto c11 = BTFFetchSpectrum(btf, light_vert_idx, view_vert_idx, x1, y1);

            float fx1 = tc_unorm.x - FastFloor(tc_unorm.x),
                  fx0 = 1.0f - fx1,
                  fy1 = tc_unorm.y - FastFloor(tc_unorm.y),
                  fy0 = 1.0f - fy1;

            auto spec_value = (fx0 * c00 + fx1 * c01) * fy0 +
                              (fx0 * c10 + fx1 * c11) * fy1;

            final_result += weight_light*weight_view*spec_value;
        }
    }

    return final_result;

}

inline EXPORT_CUDA uint32_t BTFNearestAngle(const BTF* btf, const Vector3& dir)
{
    uint32_t nearest_idx = 0;
    float cos_vec = fabsf(Tempest::Dot(Tempest::Normalize(Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[0])), dir));

    for(uint32_t idx = 1; idx < btf->LightCount; ++idx)
    {
        float cur_cos_vec = fabsf(Tempest::Dot(Tempest::Normalize(Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[idx])), dir));
        if(cur_cos_vec > cos_vec)
        {
            nearest_idx = idx;
            cos_vec = cur_cos_vec;
        }
    }

    return nearest_idx;
}

inline EXPORT_CUDA Spectrum BTFFetchSpectrum(const BTF* btf, uint32_t light_vert, uint32_t view_vert, uint32_t x, uint32_t y)
{
    TGE_ASSERT(light_vert < btf->LightCount && view_vert < btf->LightCount &&
                x < btf->Width && y < btf->Height, "Invalid index");
    uint32_t xy_idx = y*btf->Width + x;
    uint32_t lv_idx = view_vert*btf->LightCount + light_vert;

    return BTFFetchSpectrum(btf, lv_idx, xy_idx);
}

inline EXPORT_CUDA Spectrum BTFFetchSpectrum(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx)
{
    uint32_t chan_count = btf->ChannelCount;

    if(chan_count > BTF_MAX_CHANNEL_COUNT)
        return {};
    
    float multi_chan_result[BTF_MAX_CHANNEL_COUNT];
    memset(multi_chan_result, 0, BTF_MAX_CHANNEL_COUNT*sizeof(float));

    float* res_ptr = multi_chan_result;
    BTFFetchChannels(btf, lv_idx, xy_idx, &res_ptr);
        
    Spectrum result{};
    float drr_eps = 1e-5f;

    switch(chan_count)
    {
    case 3:
    {
        Vector3 interm_color = *reinterpret_cast<Vector3*>(multi_chan_result);

        switch(btf->ColorModel)
        {
        case 0: break;
        case 11:
        {
            YUV yuv;
            yuv.Color.x = expf(interm_color.x) - drr_eps;
            yuv.Color.y = interm_color.y*yuv.Color.x + drr_eps;
            yuv.Color.z = interm_color.z*yuv.Color.x + drr_eps;
            interm_color = Tempest::YUVToRGB(yuv);
        } break;
        default:
        {
            TGE_ASSERT(false, "Unsupported color model");
        }
        }

        if(btf->DynamicRangeReduction)
        {
            interm_color = Vector3Exp(interm_color) - drr_eps;
        }
        result = RGBToSpectrum(interm_color); 
    } break;
    default:
    {
        TGE_ASSERT(false, "Conversion is unsupported");
    } break;
    }

    if(btf->ConsineFlag)
    {
        TGE_ASSERT(false, "Stub");
        // TODO
    }

    return result;
}

inline EXPORT_CUDA void BTFFetchChannelsSingleFloat(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** multi_chan_result)
{
    switch(btf->DataSize)
    {
    case 2:
    {
        static_assert(sizeof(half) == sizeof(uint16_t), "Invalid half-float implementation");
        BTFEvaluateMatrix<half, float>(btf, lv_idx, xy_idx, multi_chan_result);
    } break;
    case 4:
    {
        BTFEvaluateMatrix<float, float>(btf, lv_idx, xy_idx, multi_chan_result);
    } break;
    case 8:
    {
        BTFEvaluateMatrix<double, double>(btf, lv_idx, xy_idx, multi_chan_result);
    } break;
    default:
    {
        TGE_ASSERT(false, "Unsupported");
    }
    }
}

#ifndef LINUX
inline void BTFFetchChannelsSIMD(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** multi_chan_result)
{
    switch(btf->DataSize)
    {
    case 2:
    {
        static_assert(sizeof(half) == sizeof(uint16_t), "Invalid half-float implementation");
        BTFEvaluateMatrixHalfFloatToFloat(btf, lv_idx, xy_idx, multi_chan_result);
    } break;
    case 4:
    {
        BTFEvaluateMatrix<float, float>(btf, lv_idx, xy_idx, multi_chan_result); // TODO: SIMD
    } break;
    case 8:
    {
        BTFEvaluateMatrix<double, double>(btf, lv_idx, xy_idx, multi_chan_result); // TODO: SIMD
    } break;
    default:
    {
        TGE_ASSERT(false, "Unsupported");
    }
    }
}
#endif

template<class T, class TOp>
inline EXPORT_CUDA void BTFEvaluateMatrix(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** result)
{
    uint32_t chan_count = btf->ChannelCount;
	uint32_t* u_offsets = btf->Offsets;
	uint32_t* sxv_offsets = btf->Offsets + btf->ChannelCount;
    for(uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
    {
		auto component_count = btf->ComponentCounts[chan_idx];
		uint32_t SxV_elem_offset = xy_idx*component_count*btf->SxVElementStride;
		uint32_t SxV_offset = sxv_offsets[chan_idx];
		T* SxVslice = reinterpret_cast<T*>(btf->RightSingularSxV + SxV_offset + SxV_elem_offset);

        uint32_t u_elem_offset = lv_idx*component_count*btf->UElementStride;
		uint32_t u_offset = u_offsets[chan_idx];
		T* Uslice = reinterpret_cast<T*>(btf->LeftSingularU + u_offset + u_elem_offset);
        TOp sum{};
        for(uint32_t comp_idx = 0; comp_idx < component_count; ++comp_idx)
        {
            auto u = static_cast<TOp>(Uslice[comp_idx]);
            auto v = static_cast<TOp>(SxVslice[comp_idx]); 
            TGE_ASSERT(std::isfinite(u) && std::isfinite(v), "invalid component");
            sum += u*v;
        }
        (*result)[chan_idx] = static_cast<float>(sum);
    }
}

#ifndef LINUX
inline void BTFEvaluateMatrixHalfFloatToFloat(const BTF* btf, uint32_t lv_idx, uint32_t xy_idx, float** result)
{
    uint32_t chan_count = btf->ChannelCount;
	uint32_t* u_offsets = btf->Offsets;
	uint32_t* sxv_offsets = btf->Offsets + btf->ChannelCount;
    for(uint32_t chan_idx = 0; chan_idx < chan_count; ++chan_idx)
    {
		auto component_count = btf->ComponentCounts[chan_idx];
		uint32_t SxV_idx = xy_idx*component_count*btf->SxVElementStride;
		uint32_t SxV_offset = sxv_offsets[chan_idx];
		auto SxVslice = btf->RightSingularSxV + SxV_idx + SxV_offset;

		uint32_t u_idx = lv_idx*component_count*btf->UElementStride;
		uint32_t u_offset = u_offsets[chan_idx];
		auto Uslice = btf->LeftSingularU + u_idx + u_offset;

        __m256 sum = _mm256_setzero_ps();
        const uint32_t m128_fp16_count = sizeof(__m128)/sizeof(uint16_t);
        uint32_t steps = component_count/m128_fp16_count;
        for(uint32_t step_idx = 0; step_idx < steps; ++step_idx)
        {
            uint32_t offset = step_idx*sizeof(__m128);
            __m128i U_vec_half = _mm_loadu_si128(reinterpret_cast<__m128i*>(Uslice + offset));
            __m128i SxV_vec_half = _mm_loadu_si128(reinterpret_cast<__m128i*>(SxVslice + offset));

            __m256 U_vec = _mm256_cvtph_ps(U_vec_half);
            __m256 V_vec = _mm256_cvtph_ps(SxV_vec_half);

            __m256 USV = _mm256_mul_ps(U_vec, V_vec);

            sum = _mm256_add_ps(sum, USV);
        }

        union partial_m128
        {
            uint16_t value[m128_fp16_count];
            __m128i pack;
        } partial_U_half, partial_SxV_half;

        partial_U_half.pack = _mm_setzero_si128();
        partial_SxV_half.pack = _mm_setzero_si128();

        for(uint32_t comp_idx = steps*m128_fp16_count, part_idx = 0; comp_idx < component_count; ++comp_idx, ++part_idx)
        {
            partial_U_half.value[part_idx] = reinterpret_cast<uint16_t*>(Uslice)[comp_idx];
            partial_SxV_half.value[part_idx] = reinterpret_cast<uint16_t*>(SxVslice)[comp_idx];
        }

        __m256 partial_U = _mm256_cvtph_ps(partial_U_half.pack);
        __m256 partial_SxV = _mm256_cvtph_ps(partial_SxV_half.pack);

        __m256 partial_USV = _mm256_mul_ps(partial_U, partial_SxV);

        sum = _mm256_add_ps(sum, partial_USV);

        __m128 sum_lo = _mm256_extractf128_ps(sum, 0);
        __m128 sum_hi = _mm256_extractf128_ps(sum, 1);

        __m128 sum_reduce = _mm_add_ps(sum_lo, sum_hi);
        sum_reduce = _mm_hadd_ps(sum_reduce, sum_reduce);
        sum_reduce = _mm_hadd_ps(sum_reduce, sum_reduce);

        float final_sum = _mm_cvtss_f32(sum_reduce);

        (*result)[chan_idx] = final_sum;
    }
}
#endif
}

#endif // _TEMPEST_BTF_HH_

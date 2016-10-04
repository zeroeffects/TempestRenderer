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

#define ILLUMINATION_MODEL_IMPLEMENTATION
#define EXPORT_TABLE
#include "embree2/rtcore.h"
#include "embree2/rtcore_ray.h"

#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/graphics/cube-map.hh"
#include "tempest/math/vector4.hh"
#include "tempest/utils/logging.hh"
#include "tempest/utils/system.hh"
#include "tempest/volume/volume.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/math/intersect.hh"
#include "tempest/math/hdr.hh"

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

#include <algorithm>
#include <thread>
#include <cmath>
#include <cstring>

namespace Tempest
{
#define ENABLE_MIS 1
//#define ENABLE_MIS_POWER_HEURISTIC 1
static const float SurfaceHitCriteria = 175;
static const float VolumeHitCriteria = 5;
static const __m128 SurfaceHitCriteriaV4 = _mm_set1_ps(SurfaceHitCriteria);
static const __m128 VolumeHitCriteriaV4 = _mm_set1_ps(VolumeHitCriteria);
static const float TransmittanceThreshold = 1e-3f;
static const float RcpMaxExtinction = (1.0f/255.0f); // well should be something per voxel grid
static const float DensityScale = 100.0f;
static const float Epsilon = 1e-3f;

// Some precomputed tables
float s_cosPhi[256],
 	  s_sinPhi[256],
      s_cosTheta[256],
	  s_sinTheta[256];

static void UnalignedToAlignedBounds(const AABBUnaligned& aabb, RTCBounds* bounds)
{
    bounds->lower_x = aabb.MinCorner.x;
    bounds->lower_y = aabb.MinCorner.y;
    bounds->lower_z = aabb.MinCorner.z;
    
    bounds->upper_x = aabb.MaxCorner.x;
    bounds->upper_y = aabb.MaxCorner.y;
    bounds->upper_z = aabb.MaxCorner.z;
}

void RTMaterial::setup()
{
    auto setup_func = MaterialSetupLookup[(size_t)Model];
    if(setup_func)
    {
        setup_func(this);
    }
}

inline static __m128 TrilinearInterpolation(__m128 fact4, __m128 ifact4, 
                                            __m128 p0, __m128 p1)
{
	__m128 mix_fact4_x = _mm_shuffle_ps(ifact4, fact4, _MM_SHUFFLE(0, 0, 0, 0));
	mix_fact4_x = _mm_permute_ps(mix_fact4_x, _MM_SHUFFLE(3, 1, 2, 0));
	__m128 mix_fact4_y = _mm_shuffle_ps(ifact4, fact4, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 mul_fact4_xy = _mm_mul_ps(mix_fact4_x, mix_fact4_y);
	__m128 ifact4_z = _mm_permute_ps(ifact4, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 mul_fact4_xyz_0 = _mm_mul_ps(mul_fact4_xy, ifact4_z);
	__m128 fact4_z = _mm_permute_ps(fact4, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 mul_fact4_xyz_1 = _mm_mul_ps(mul_fact4_xy, fact4_z);
	p0 = _mm_mul_ps(p0, mul_fact4_xyz_0);
	p1 = _mm_mul_ps(p1, mul_fact4_xyz_1);

	__m128 result = _mm_add_ps(p0, p1);
	result = _mm_hadd_ps(result, result);
	result = _mm_hadd_ps(result, result);

	return result;
}

static void SampleVolumePoint(const RTVolume& volume, const Vector3& point, const Vector3& lower, const Vector3& upper, Vector3* tangent)
{
    const size_t angle_offset = volume.Dimensions.X*volume.Dimensions.Y*volume.Dimensions.Z;

    float t00 = 0.0f, t01 = 0.0f, t02 = 0.0f,
		  t10 = 0.0f, t11 = 0.0f, t12 = 0.0f,
		  t20 = 0.0f, t21 = 0.0f, t22 = 0.0f;

    Vector3 lfactor(point - lower);
    Vector3 v3factor[2] = { lfactor, Vector3{1.0f, 1.0f, 1.0f} - lfactor };

    uint32_t x[2] = { (uint32_t)lower.x, (uint32_t)upper.x },
           y[2] = { (uint32_t)lower.y, (uint32_t)upper.y },
           z[2] = { (uint32_t)lower.z, (uint32_t)upper.z };

    // TODO: Replace
	for (int k=0; k < 8; ++k)
	{
        uint32_t index = (z[(k & 4) >> 2] * volume.Dimensions.Y + y[(k & 2) >> 1]) * volume.Dimensions.X + x[k & 1];
		float factor = v3factor[(k & 4) >> 2].z * v3factor[(k & 2) >> 1].y * v3factor[k & 1].x;
     
//        TGE_ASSERT(volume.GridData[index], "Zero density cell - would result in incorrect orientation");

		uint8_t* quant = (uint8_t*)volume.GridData + angle_offset + 2*sizeof(uint8_t)*index;
		uint8_t theta = quant[0];
		uint8_t phi = quant[1];

        float cos_phi = s_cosPhi[phi];
        float sin_phi = s_sinPhi[phi];
        float cos_theta = s_cosTheta[theta];
        float sin_theta = s_sinTheta[theta];

		float d[4] = { cos_phi*sin_theta, sin_phi*sin_theta, cos_theta, 0.0f };

		// The outer product gives us the structure tensor.
		// What we find is the smoothed structure tensor at x.
		t00 += factor * d[0] * d[0];
		t01 += factor * d[0] * d[1];
		t02 += factor * d[0] * d[2];
		t11 += factor * d[1] * d[1];
		t12 += factor * d[1] * d[2];
		t22 += factor * d[2] * d[2];
	}

	t10 = t01;
	t20 = t02;
	t21 = t12;

	// To derive the local orientation we must find the eigenvector of
	// the structure tensor. Refer to Coherence-Enhancing Filtering on the GPU
	// (GPU Pro 4 article). Also, Visualization Handbook for reference about this
	// technique without explanation.

    if(t00 != 0.0f || t01 != 0.0f || t02 != 0.0f ||
       t00 != 0.0f || t11 != 0.0f || t12 != 0.0f ||
       t20 != 0.0f || t21 != 0.0f || t22 != 0.0f)
    {
		*tangent = Vector3{RcpSqrt3, RcpSqrt3, RcpSqrt3}; // TODO: Is there better initial guess? 

	    // power iteration
	    for (int i=0; i<8; ++i)
	    {
            *tangent = Vector3{t00*tangent->x + t01*tangent->y + t02*tangent->z,
							   t10*tangent->x + t11*tangent->y + t12*tangent->z,
							   t20*tangent->x + t21*tangent->y + t22*tangent->z};
            NormalizeSelf(&*tangent);
	    }
    }
    else
    {
		*tangent = Vector3{0.0f, 0.0f, 0.0f};
    }
}

inline static void SampleVolume(uint8_t* volume, __m128 cvec4, __m128 min_cell, __m128 max_cell, __m128i dim_orig, __m128* p0, __m128* p1, __m128* factor, __m128* inv_factor)
{
    __m128i idx0 = _mm_cvtps_epi32(min_cell);
    __m128i idx1 = _mm_cvtps_epi32(max_cell);

    __m128i z0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(idx0), _mm_castsi128_ps(idx0), _MM_SHUFFLE(2, 2, 2, 2)));
    __m128i dim_y = _mm_shuffle_epi32(dim_orig, _MM_SHUFFLE(1, 1, 1, 1));

    __m128i p0_idx = _mm_mullo_epi32(z0, dim_y);
    __m128i y_mix_idx = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(idx0), _mm_castsi128_ps(idx1), _MM_SHUFFLE(1, 1, 1, 1)));
    p0_idx = _mm_add_epi32(p0_idx, y_mix_idx);

    __m128i dim_x = _mm_shuffle_epi32(dim_orig, _MM_SHUFFLE(0, 0, 0, 0));
    p0_idx = _mm_mullo_epi32(p0_idx, dim_x);

    __m128i x_mix_idx = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(idx0), _mm_castsi128_ps(idx1), _MM_SHUFFLE(0, 0, 0, 0)));
    x_mix_idx = _mm_shuffle_epi32(x_mix_idx, _MM_SHUFFLE(2, 0, 2, 0));

    p0_idx = _mm_add_epi32(p0_idx, x_mix_idx);

    uint8_t p000 = volume[ElementInt<0>(p0_idx)],
          p001 = volume[ElementInt<1>(p0_idx)],
          p010 = volume[ElementInt<2>(p0_idx)],
          p011 = volume[ElementInt<3>(p0_idx)];

    __m128i z1 = _mm_shuffle_epi32(idx1, _MM_SHUFFLE(2, 2, 2, 2));
    __m128i p1_idx = _mm_mullo_epi32(z1, dim_y);
    p1_idx = _mm_add_epi32(p1_idx, y_mix_idx);
    p1_idx = _mm_mullo_epi32(p1_idx, dim_x);
    p1_idx = _mm_add_epi32(p1_idx, x_mix_idx);
    
    uint8_t p100 = volume[ElementInt<0>(p1_idx)],
          p101 = volume[ElementInt<1>(p1_idx)],
          p110 = volume[ElementInt<2>(p1_idx)],
          p111 = volume[ElementInt<3>(p1_idx)];

	*p0 = _mm_setr_ps(p000, p001, p010, p011);
	*p1 = _mm_setr_ps(p100, p101, p110, p111);

    *factor = _mm_sub_ps(cvec4, min_cell);
    *inv_factor = _mm_sub_ps(_mm_set1_ps(1.0f), *factor);
}

inline static void SampleVolume(uint8_t* grid, __m128 cvec4, __m128i dim_orig, __m128 mask, __m128* p0, __m128* p1, __m128* factor, __m128* inv_factor, __m128* min_cell, __m128* max_cell)
{
    __m128 min_cell_pos = _mm_floor_ps(cvec4);
    __m128 max_cell_pos = _mm_add_ps(min_cell_pos, _mm_set1_ps(1));

    __m128 max_cell_neg = _mm_ceil_ps(cvec4);
    __m128 min_cell_neg = _mm_add_ps(max_cell_neg, _mm_set1_ps(-1));

    *min_cell = _mm_blendv_ps(min_cell_neg, min_cell_pos, mask);
    *max_cell = _mm_blendv_ps(max_cell_neg, max_cell_pos, mask);

    __m128i dimi = _mm_add_epi32(dim_orig, _mm_set1_epi32(-1));
    *min_cell = _mm_max_ps(*min_cell, _mm_setzero_ps());
    *max_cell = _mm_min_ps(*max_cell, _mm_cvtepi32_ps(dimi));

    SampleVolume(grid, cvec4, *min_cell, *max_cell, dim_orig, p0, p1, factor, inv_factor);
}

bool RTVolumeSet::sampleScattering(float threshold, const Vector3& pos, SampleData* data)
{
    // Early out if completely outside
    if(MinCorner.x > pos.x || pos.x > MaxCorner.x ||
       MinCorner.y > pos.y || pos.y > MaxCorner.y ||
       MinCorner.z > pos.z || pos.z > MaxCorner.z)
    {
        return false;
    }

	Vector3 dims = ToVector3(Dimensions);
	Vector3 hi_vol_div = dims/(MaxCorner - MinCorner);

	Vector3 vol_cell = (pos - MinCorner) * hi_vol_div;
	vol_cell = Vector3Clamp(vol_cell, Vector3{0, 0, 0}, dims - Vector3{1, 1, 1});
	
	int32_t idx = (int32_t(vol_cell.z)*Dimensions.Y + int32_t(vol_cell.y))*Dimensions.X + int32_t(vol_cell.x);

	RTVolume* vol_ptr = VolumeGrid[idx];
	if(vol_ptr == nullptr)
	{
		data->Tangent = Vector3{0.0f, 0.0f, 0.0f};
		data->Binormal = Vector3{0.0f, 0.0f, 0.0f};
        data->TotalDensity = 0.0f;
        data->DirectionalDensity = 0.0f;
		return false;
	}

    auto& volume = *vol_ptr;

    __m128 min_corner = _mm_setr_ps(volume.MinCorner.x, volume.MinCorner.y, volume.MinCorner.z, 0.0f);
    __m128 max_corner = _mm_setr_ps(volume.MaxCorner.x, volume.MaxCorner.y, volume.MaxCorner.z, 0.0f);
    __m128i dim_orig = _mm_setr_epi32(volume.Dimensions.X, volume.Dimensions.Y, volume.Dimensions.Z, 0);
	__m128i dimi = _mm_add_epi32(dim_orig, _mm_set1_epi32(-1));
    __m128 dimf = _mm_cvtepi32_ps(dimi);
    __m128 volume_size = _mm_sub_ps(max_corner, min_corner);
    __m128 div = _mm_div_ps(dimf, volume_size);

    __m128 pos_v4 = _mm_setr_ps(pos.x, pos.y, pos.z, 0.0f);
    __m128 cvec4 = _mm_sub_ps(pos_v4, min_corner);
    cvec4 = _mm_mul_ps(cvec4, div);

    cvec4 = _mm_max_ps(cvec4, _mm_setzero_ps());
    cvec4 = _mm_min_ps(cvec4, dimf);

	__m128 factor, inv_factor, p0, p1;

    __m128 min_cell = _mm_floor_ps(cvec4);
    __m128 max_cell = _mm_add_ps(min_cell, _mm_set1_ps(1));
    max_cell = _mm_min_ps(max_cell, dimf);

    SampleVolume(volume.GridData, cvec4, min_cell, max_cell, dim_orig, &p0, &p1, &factor, &inv_factor);

    __m128 density = TrilinearInterpolation(factor, inv_factor, p0, p1);

    // Early out before sampling orientation. It won't get better anyway
	if(ElementFloat<0>(density) == 0.0f)
	{
		data->Tangent = Vector3{0.0f, 0.0f, 0.0f};
		data->Binormal = Vector3{0.0f, 0.0f, 0.0f};
        data->TotalDensity = 0.0f;
        data->DirectionalDensity = 0.0f;
		return false;
	}

	float max_density = ElementFloat<0>(density) / 255.0f;
    if(max_density < threshold)
    {
		data->Tangent = Vector3{0.0f, 1.0f, 0.0f};
		data->Binormal = Vector3{0.0f, 0.0f, 0.0f};
        data->TotalDensity = max_density;
        data->DirectionalDensity = 1.0f;
        return true;
    }

    Vector3 orientation;
	SampleVolumePoint(volume, Vector3{ElementFloat<0>(cvec4), ElementFloat<1>(cvec4), ElementFloat<2>(cvec4)},
							  Vector3{ElementFloat<0>(min_cell), ElementFloat<1>(min_cell), ElementFloat<2>(min_cell)},
							  Vector3{ElementFloat<0>(max_cell), ElementFloat<1>(max_cell), ElementFloat<2>(max_cell)},
                              &orientation);

    auto vol_material = static_cast<const RTVolumeMaterial*>(data->Material);

    Tempest::Matrix3 basis;
    basis.makeBasisTangent(orientation);
    data->Tangent = basis.tangent();
    data->Binormal = basis.binormal();
    data->Normal = basis.normal();

    auto dir_density = DensityLookup[(size_t)vol_material->VolumeModel](*data);

    data->TotalDensity = (DensityScale/255.0f) * ElementFloat<0>(density) * dir_density;
    data->DirectionalDensity = dir_density;

    return true;
}

void RTVolumeSet::sample(RTCRay& ray, SampleData* data) const
{
	size_t vol = -ray.instID - 1;
	auto& volume = Volumes.Values[vol];
	
	//uint8_t density = volume.GridData[voxel_idx];
	//float dn = density*(1.0f/255.0f);

	Vector3 size = volume.MaxCorner - volume.MinCorner;
	Vector3 dims = ToVector3(volume.Dimensions) - 1.0f;

    uint32_t x =  ray.primID %  volume.Dimensions.X;
    uint32_t y = (ray.primID /  volume.Dimensions.X) % volume.Dimensions.Y;
    uint32_t z =  ray.primID / (volume.Dimensions.X  * volume.Dimensions.Y);

    Tempest::Vector3 dir = ToVector3(ray.dir);

    Tempest::Vector3 p = (ToVector3(ray.org) + ray.tnear*dir - volume.MinCorner)*dims/size;
	
	Vector3 point = Vector3Clamp(p, Vector3{0.0f, 0.0f, 0.0f}, dims),
			lower{(float)x, (float)y, (float)z},
			upper{float(x + 1), float(y + 1), float(z + 1)};

    upper = Vector3Min(upper, dims);

    point = Vector3Clamp(point, lower, upper);

    SampleVolumePoint(volume, point, lower, upper, &data->Tangent);

    data->Normal = ToVector3(ray.Ng);
    data->Material = Material;
}

void DefaultMeshSample(void* v0, void* v1, void* v2, uint32_t stride, const Vector3& barycentric, SampleData* data)
{
    size_t normal_offset = stride - 3*sizeof(float);

    auto norm0 = (Vector3*)(reinterpret_cast<uint8_t*>(v0) + normal_offset);
    auto norm1 = (Vector3*)(reinterpret_cast<uint8_t*>(v1) + normal_offset);
    auto norm2 = (Vector3*)(reinterpret_cast<uint8_t*>(v2) + normal_offset);

    float u = barycentric.x, v = barycentric.y, w = barycentric.z;

	Vector3 norm = Normalize(*norm0*w + *norm1*u + *norm2*v);

    if(stride >= sizeof(PTNFormat))
    {
        auto& tc0 = reinterpret_cast<const PTNFormat*>(v0)->TexCoord;
        auto& tc1 = reinterpret_cast<const PTNFormat*>(v1)->TexCoord;
        auto& tc2 = reinterpret_cast<const PTNFormat*>(v2)->TexCoord;

        data->TexCoord = tc0*w + tc1*u + tc2*v;

#if 1
		if(stride == sizeof(PTTcNFormat))
		{
			auto a0 = reinterpret_cast<const PTTcNFormat*>(v0)->InterpolationConstant;
			auto a1 = reinterpret_cast<const PTTcNFormat*>(v1)->InterpolationConstant;
			auto a2 = reinterpret_cast<const PTTcNFormat*>(v2)->InterpolationConstant;

			float a_coef = a0*w + a1*u + a2*v;
			
			norm = ComputeConsistentNormal(-data->IncidentLight, norm, a_coef);

			TGE_ASSERT(std::isfinite(norm.x) && std::isfinite(norm.y) && std::isfinite(norm.z) &&
					   (norm.x || norm.y || norm.z), "invalid normal");
		}
#endif

		if(stride >= sizeof(PTTNFormat))
		{
			auto& tan0 = reinterpret_cast<const PTTNFormat*>(v0)->Tangent;
			auto& tan1 = reinterpret_cast<const PTTNFormat*>(v1)->Tangent;
			auto& tan2 = reinterpret_cast<const PTTNFormat*>(v2)->Tangent;

			auto tangent = tan0*w + tan1*u + tan2*v;

			Tempest::Matrix3 basis;
			basis.makeBasisOrthogonalize(tangent, norm);

			data->Tangent = basis.tangent();
			data->Binormal = basis.binormal();

			TGE_ASSERT(std::isfinite(data->Tangent.x) && std::isfinite(data->Tangent.y) && std::isfinite(data->Tangent.z) &&
					   (data->Tangent.x || data->Tangent.y || data->Tangent.z), "Broken tangent");
		}
    }
    else
    {
        data->TexCoord = Vector2{ 0.0f, 0.0f };
    }

	data->Normal = norm;
}

void RTMesh::sample(RTCRay& ray, SampleData* data) const
{
    auto prim_id = ray.primID;

    uint32_t index0 = IndexBuffer[prim_id*3];
    uint32_t index1 = IndexBuffer[prim_id*3 + 1];
    uint32_t index2 = IndexBuffer[prim_id*3 + 2];

	char* v0 = (char*)VertexBuffer + index0*Stride;
	char* v1 = (char*)VertexBuffer + index1*Stride;
	char* v2 = (char*)VertexBuffer + index2*Stride;

    Tempest::Vector3 interpolation{ ray.u, ray.v, 1.0f-ray.u-ray.v };

    data->Material = Material;

	if(Sampler)
	{
		Sampler(v0, v1, v2, Stride, interpolation, data);
    }
    else
    {
        DefaultMeshSample(v0, v1, v2, Stride, interpolation, data);
    }
}

void RTHair::sample(RTCRay& ray, SampleData* data) const
{
    data->Tangent = ToVector3(ray.Ng);
    NormalizeSelf(&data->Tangent);
    data->Binormal = Cross(data->Tangent, ToVector3(ray.dir));
    NormalizeSelf(&data->Binormal);
    data->Normal = Cross(data->Tangent, data->Binormal);
    data->Material = Material;
    data->TexCoord = Vector2{}; // TODO or not TODO - what a question
}


static void HierarchicalVolumeBoundsFunction(RTVolumeSet* user_geom, size_t i, RTCBounds& bounds)
{
    auto& volume = user_geom->Volumes.Values[i];
    bounds.lower_x = volume.MinCorner.x;
    bounds.lower_y = volume.MinCorner.y;
    bounds.lower_z = volume.MinCorner.z;
    bounds.upper_x = volume.MaxCorner.x;
    bounds.upper_y = volume.MaxCorner.y;
    bounds.upper_z = volume.MaxCorner.z;
}

inline static __m128 GridStep(__m128 rcp_dir, __m128 dir4, __m128 cvec4)
{
    __m128 floor_vec = _mm_floor_ps(cvec4);
    floor_vec = _mm_add_ps(floor_vec, _mm_set1_ps(1));
    __m128 ceil_vec = _mm_ceil_ps(cvec4);
    ceil_vec = _mm_add_ps(ceil_vec, _mm_set1_ps(-1));

    __m128 mask = _mm_cmplt_ps(rcp_dir, _mm_setzero_ps());
    __m128 target = _mm_blendv_ps(floor_vec, ceil_vec, mask);
    __m128 delta = _mm_sub_ps(target, cvec4);
    delta = _mm_mul_ps(delta, rcp_dir);

    __m128 min_delta = _mm_min_ps(delta, _mm_shuffle_ps(delta, delta, _MM_SHUFFLE(3, 0, 0, 1)));
    min_delta = _mm_min_ps(min_delta, _mm_shuffle_ps(delta, delta, _MM_SHUFFLE(3, 1, 2, 2)));
    __m128 step = _mm_mul_ps(min_delta, dir4);
    cvec4 = _mm_add_ps(cvec4, step);

    __m128 prominence_mask = _mm_cmpeq_ps(min_delta, delta);

	cvec4 = _mm_blendv_ps(cvec4, target, prominence_mask);

	return cvec4;
}

static void HierarchicalVolumeIntersectFunction(RTVolumeSet* user_geom, RTCRay& ray, size_t i)
{
    auto& volume = user_geom->Volumes.Values[i];

    __m128 dir_v4 = _mm_load_ps(ray.dir);
    __m128 org = _mm_load_ps(ray.org);
    //__m128 rcp_dir = _mm_rcp_ps(dir_v4); // poor precision
    __m128 rcp_dir = _mm_div_ps(_mm_set1_ps(1), dir_v4);

    __m128 mask = _mm_cmpge_ps(rcp_dir, _mm_setzero_ps()); // Might not be perfect
    __m128 min_corner = _mm_setr_ps(volume.MinCorner.x, volume.MinCorner.y, volume.MinCorner.z, 0.0f);
    __m128 max_corner = _mm_setr_ps(volume.MaxCorner.x, volume.MaxCorner.y, volume.MaxCorner.z, 0.0f);
    __m128 tmin_v4 = _mm_blendv_ps(max_corner, min_corner, mask);
    __m128 tmax_v4 = _mm_blendv_ps(min_corner, max_corner, mask);

    __m128 min_value = _mm_set1_ps(std::numeric_limits<float>::min());
    tmin_v4 = _mm_sub_ps(tmin_v4, org);
    tmin_v4 = _mm_add_ps(tmin_v4, min_value);
    tmin_v4 = _mm_mul_ps(tmin_v4, rcp_dir);
    //tmin_v4 = _mm_div_ps(tmin_v4, dir_v4);
    tmin_v4 = _mm_max_ps(tmin_v4, _mm_permute_ps(tmin_v4, _MM_SHUFFLE(3, 0, 0, 1)));
    tmin_v4 = _mm_max_ps(tmin_v4, _mm_permute_ps(tmin_v4, _MM_SHUFFLE(3, 1, 2, 2)));

    tmax_v4 = _mm_sub_ps(tmax_v4, org);
    tmax_v4 = _mm_add_ps(tmax_v4, min_value);
    tmax_v4 = _mm_mul_ps(tmax_v4, rcp_dir);
    //tmax_v4 = _mm_div_ps(tmax_v4, dir_v4);
    tmax_v4 = _mm_min_ps(tmax_v4, _mm_permute_ps(tmax_v4, _MM_SHUFFLE(3, 0, 0, 1)));
    tmax_v4 = _mm_min_ps(tmax_v4, _mm_permute_ps(tmax_v4, _MM_SHUFFLE(3, 1, 2, 2)));

    float tmin = ElementFloat<0>(tmin_v4);
    float tmax = ElementFloat<0>(tmax_v4);
    if(tmin > tmax || tmax < ray.tnear || tmin > ray.tfar)
        return;

	__m128i inc4 = _mm_castps_si128(mask);
	inc4 = _mm_mullo_epi32(inc4, _mm_set1_epi32(2));
	inc4 = _mm_sub_epi32(_mm_set1_epi32(-1), inc4);
    __m128 inc4f = _mm_cvtepi32_ps(inc4);

    TGE_ASSERT(std::isfinite(tmin) && std::isfinite(tmax), "bad intersection point");

    __m128 cvec4;
    
    __m128i dim_orig = _mm_setr_epi32(volume.Dimensions.X, volume.Dimensions.Y, volume.Dimensions.Z, 0);
	__m128i dimi = _mm_add_epi32(dim_orig, _mm_set1_epi32(-1));
    __m128 dimf = _mm_cvtepi32_ps(dimi);
    __m128 volume_size = _mm_sub_ps(max_corner, min_corner);
    __m128 div = _mm_div_ps(dimf, volume_size);

    if(tmin <= ray.tnear)
    {
		__m128 offset = _mm_mul_ps(dir_v4, _mm_set1_ps(ray.tnear));
		__m128 cur_org = _mm_add_ps(org, offset);

        cvec4 = _mm_sub_ps(cur_org, min_corner);
        cvec4 = _mm_mul_ps(cvec4, div);

        cvec4 = _mm_max_ps(cvec4, _mm_setzero_ps());
        cvec4 = _mm_min_ps(cvec4, dimf);

        cvec4 = GridStep(rcp_dir, dir_v4, cvec4);

        __m128 cmp0 = _mm_cmpgt_ps(_mm_setzero_ps(), cvec4);
        __m128 cmp1 = _mm_cmpgt_ps(cvec4, dimf);
        __m128 cmp = _mm_or_ps(cmp0, cmp1);

        // Is simd better in this case?
        if(ElementInt<0>(cmp) || ElementInt<1>(cmp) || ElementInt<2>(cmp))
            return; // Mind the return
    }
    else
    {
        __m128 step = _mm_mul_ps(dir_v4, tmin_v4);
        cvec4 = _mm_add_ps(org, step);
        cvec4 = _mm_sub_ps(cvec4, min_corner);
        cvec4 = _mm_mul_ps(cvec4, div);

        cvec4 = _mm_max_ps(cvec4, _mm_setzero_ps());
        cvec4 = _mm_min_ps(cvec4, dimf);
    }

    TGE_ASSERT(tmin != -std::numeric_limits<float>::infinity(), "bad minimum intersection point");
    TGE_ASSERT(tmax != std::numeric_limits<float>::infinity(), "bad minimum intersection point");

    __m128 factor, inv_factor, min_cell, max_cell, p0, p1;
    SampleVolume(volume.GridData, cvec4, dim_orig, mask, &p0, &p1, &factor, &inv_factor, &min_cell, &max_cell);

    __m128 cur_density = TrilinearInterpolation(factor, inv_factor, p0, p1);
        
	__m128 hit_criteria = IsVolumeIlluminationModel(user_geom->Material->Model) ? VolumeHitCriteriaV4 : SurfaceHitCriteriaV4;
    for(;;)
    {
        __m128 next_cvec4 = cvec4;
        next_cvec4 = GridStep(rcp_dir, dir_v4, next_cvec4);

        __m128 cmp0 = _mm_cmpgt_ps(_mm_setzero_ps(), next_cvec4);
        __m128 cmp1 = _mm_cmpgt_ps(next_cvec4, dimf);
        __m128 cmp = _mm_or_ps(cmp0, cmp1);

        // Is simd better in this case?
        if(ElementInt<0>(cmp) || ElementInt<1>(cmp) || ElementInt<2>(cmp))
            return; // Mind the return

        __m128 cmp_p0 = _mm_or_ps(p0, p1);

        if(ElementInt<0>(cmp_p0) || ElementInt<1>(cmp_p0) || ElementInt<2>(cmp_p0) || ElementInt<3>(cmp_p0))
        {
            factor = _mm_sub_ps(next_cvec4, min_cell);
            inv_factor = _mm_sub_ps(_mm_set1_ps(1), factor);

            __m128 next_density = TrilinearInterpolation(factor, inv_factor, p0, p1);

            __m128 hit_dist0 = hit_criteria - next_density;
            __m128 hit_dist1 = hit_criteria - cur_density;

            __m128 hit_cmp0 = _mm_cmple_ps(hit_dist0, _mm_setzero_ps());
            __m128 hit_cmp1 = _mm_cmple_ps(hit_dist1, _mm_setzero_ps());
            __m128 hit_cmp_final = _mm_xor_ps(hit_cmp0, hit_cmp1);

            if(ElementInt<0>(hit_cmp_final))
            {
                __m128 delta_density = _mm_sub_ps(next_density, cur_density);

                if(ElementFloat<0>(delta_density) > 1e-6f)
                {
                    __m128 t4 = _mm_div_ps(hit_dist1, delta_density);
                    __m128 delta_vec = _mm_sub_ps(next_cvec4, cvec4);
                    __m128 step = _mm_mul_ps(delta_vec, t4);

					TGE_ASSERT(ElementFloat<0>(t4) > 0.0f, "Stepping backwards is not allowed");

                    cvec4 = _mm_add_ps(cvec4, step);
                }

                factor = _mm_sub_ps(cvec4, min_cell);
                inv_factor = _mm_sub_ps(_mm_set1_ps(1), factor);

                __m128 p0_ = _mm_permute_ps(p0, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 p1_ = _mm_permute_ps(p1, _MM_SHUFFLE(2, 3, 0, 1));

                __m128 mul_x;
                {
                //__m128 dx4 = _mm_permute_ps(inc4f, _MM_SHUFFLE(0, 0, 0, 0));            
                __m128 y_first = _mm_shuffle_ps(inv_factor, factor, _MM_SHUFFLE(1, 1, 1, 1));
                __m128 z_first = _mm_permute_ps(inv_factor, _MM_SHUFFLE(2, 2, 2, 2));
                __m128 mul_first = _mm_mul_ps(y_first, z_first);
                mul_first = _mm_mul_ps(mul_first, p0_);
                //mul_first = _mm_mul_ps(mul_first, dx4);

                __m128 z_second = _mm_permute_ps(factor, _MM_SHUFFLE(2, 2, 2, 2));
                __m128 mul_second = _mm_mul_ps(y_first, z_second);
                mul_second = _mm_mul_ps(mul_second, p1_);
                //mul_second = _mm_mul_ps(mul_second, dx4);

                mul_x = _mm_add_ps(mul_first, mul_second);
                }

			    __m128 xy;
                {
                //__m128 dy4 = _mm_permute_ps(inc4f, _MM_SHUFFLE(1, 1, 1, 1));  
                __m128 shuf_p0 = _mm_shuffle_ps(p0_, p0_, _MM_SHUFFLE(0, 2, 1, 3));
                __m128 x_first = _mm_shuffle_ps(inv_factor, factor, _MM_SHUFFLE(0, 0, 0, 0));
                __m128 z_first = _mm_permute_ps(inv_factor, _MM_SHUFFLE(2, 2, 2, 2));
                __m128 mul_first = _mm_mul_ps(x_first, z_first);
                mul_first = _mm_mul_ps(mul_first, shuf_p0);
                //mul_first = _mm_mul_ps(mul_first, dy4);
                __m128 shuf_p1 = _mm_shuffle_ps(p1_, p1_, _MM_SHUFFLE(0, 2, 1, 3));
                __m128 z_second = _mm_permute_ps(factor, _MM_SHUFFLE(2, 2, 2, 2));
                __m128 mul_second = _mm_mul_ps(x_first, z_second);
                mul_second = _mm_mul_ps(mul_second, shuf_p1);
                //mul_second = _mm_mul_ps(mul_second, dy4);

                __m128 mul_y = _mm_add_ps(mul_first, mul_second);
                xy = _mm_hsub_ps(mul_x, mul_y);
                }

			    __m128 xyz_;
			    {
                //__m128 dz4 = _mm_permute_ps(inc4f, _MM_SHUFFLE(2, 2, 2, 2));
			    __m128 x_first = _mm_shuffle_ps(inv_factor, factor, _MM_SHUFFLE(0, 0, 0, 0));
			    x_first = _mm_permute_ps(x_first, _MM_SHUFFLE(0, 3, 0, 3));
                __m128 y_first = _mm_shuffle_ps(inv_factor, factor, _MM_SHUFFLE(1, 1, 1, 1));
			    __m128 mul_xy = _mm_mul_ps(x_first, y_first);
			    __m128 mul_first = _mm_mul_ps(mul_xy, p0_);
			    //mul_first = _mm_mul_ps(mul_first, dz4);
			    __m128 mul_second = _mm_mul_ps(mul_xy, p1_);
			    //mul_second = _mm_mul_ps(mul_second, dz4);
			    __m128 mul_z = _mm_sub_ps(mul_second, mul_first);
			    mul_z = _mm_hadd_ps(mul_z, _mm_setzero_ps());
			    xyz_ = _mm_hadd_ps(xy, mul_z);
			    }

			    __m128 xyz_len = _mm_mul_ps(xyz_, xyz_);
			    xyz_len = _mm_hadd_ps(xyz_len, xyz_len);
			    xyz_len = _mm_hadd_ps(xyz_len, xyz_len);
			    xyz_len = _mm_rsqrt_ps(xyz_len);
			    
                __m128 norm = _mm_mul_ps(xyz_, xyz_len);
                __m128 face_forward = _mm_mul_ps(norm, dir_v4);
                face_forward = _mm_hadd_ps(face_forward, face_forward);
                face_forward = _mm_hadd_ps(face_forward, face_forward);
                face_forward = _mm_andnot_ps(face_forward, _mm_set1_ps(-0.0f));
                norm = _mm_xor_ps(norm, face_forward);

                *((__m128*)ray.Ng) = norm;

                if(!std::isfinite(ray.Ng[0]))
                {
                    *((__m128*)ray.Ng) = _mm_xor_ps(*((__m128*)ray.dir), _mm_set1_ps(-0.0f));
                }

                TGE_ASSERT(Dot(ToVector3(ray.Ng), ToVector3(ray.dir)) <= 0.0f, "Back collision or invalid normal vector");

                break;
            }
            cur_density = next_density;
        }
        else
        {
            cur_density = _mm_setzero_ps();
        }

        cvec4 = next_cvec4;

		SampleVolume(volume.GridData, cvec4, dim_orig, mask, &p0, &p1, &factor, &inv_factor, &min_cell, &max_cell);
    }

    __m128 dist_vec = _mm_div_ps(cvec4, div);
    dist_vec = _mm_add_ps(dist_vec, min_corner);
    dist_vec = _mm_sub_ps(dist_vec, org);

    __m128 t4 = _mm_mul_ps(dist_vec, dir_v4);
	__m128 mask_w = _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1));
    t4 = _mm_and_ps(t4, mask_w);
    t4 = _mm_hadd_ps(t4, t4);
    t4 = _mm_hadd_ps(t4, t4);

    tmin = ElementFloat<0>(t4);
	if(tmin > ray.tfar) // TODO: should break earlier
		return;

    ray.u = 0.0f;
    ray.v = 0.0f;
    //ray.tnear = tmin;
    TGE_ASSERT(tmin > -1e-7f, "Bad collision");
    ray.tfar = std::max(0.0f, tmin);
    ray.geomID = user_geom->GeometryID;
    ray.instID = -(static_cast<int>(i) + 1);
    ray.primID = (((int)ElementFloat<2>(min_cell)*volume.Dimensions.Y + (int)ElementFloat<1>(min_cell))*volume.Dimensions.X + (int)ElementFloat<0>(min_cell));
}

RayTracerScene::RayTracerScene(uint32_t width, uint32_t height, const Matrix4& view_proj_inv, const RTSettings& settings)
    :   m_DrawLoop(width*height, m_ChunkSize, DrawTask(this)),
        m_DataPool(settings.DataPoolSize)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    m_ScratchMemory = std::unique_ptr<uint8_t[]>(new uint8_t[(m_ThreadPool.getThreadCount() + 1)*ScratchMemoryPerThread]);

    m_ThreadId = m_ThreadPool.allocateThreadNumber();

    InitSpectrum();

    m_Device = rtcNewDevice();
    m_Scene = rtcDeviceNewScene(m_Device, RTC_SCENE_STATIC, RTC_INTERSECT1);

    for (int i=0; i<255; i++)
	{
		float angle = (float) i * ((float) MathPi / 255.0f);
		s_cosPhi[i] = std::cos(2.0f * angle);
		s_sinPhi[i] = std::sin(2.0f * angle);
		s_cosTheta[i] = std::cos(angle);
		s_sinTheta[i] = std::sin(angle);
	}

    s_cosPhi[255] = s_sinPhi[255] = 0;
    s_cosTheta[255] = s_sinTheta[255] = 0;

    TextureDescription tex_desc;
    tex_desc.Width = width;
    tex_desc.Height = height;
    tex_desc.Format = DataFormat::RGBA8UNorm;

    uint32_t tex_size = width*height*DataFormatElementSize(DataFormat::RGBA8UNorm);
    for(auto& frame_data : m_FrameData)
    {
        frame_data.ViewProjectionInverse = view_proj_inv;
        frame_data.Counter.store(0);
        std::unique_ptr<uint8_t[]> surf(new uint8_t[tex_size]);
        memset(surf.get(), 0, tex_size);
        frame_data.Backbuffer = std::unique_ptr<Texture>(new Texture(tex_desc, surf.release()));
    }
}

void RayTracerScene::initWorkers()
{
    m_FrameStart = m_FrameTimer.time();

    auto thread_count = m_ThreadPool.getThreadCount();
    if(m_WorkerThreads != thread_count)
    {
        m_LogAverages = decltype(m_LogAverages)(new Spectrum[thread_count]);
        std::fill(m_LogAverages.get(), m_LogAverages.get() + thread_count, Spectrum{});
        m_WorkerThreads = thread_count;
    }

    if(m_PicturePostProcess == PicturePostProcess::AutoExposureHDR ||
       m_PicturePostProcess == PicturePostProcess::AutoExposureSRGB)
    {
         auto& hdr = m_FrameData[m_FrameIndex].Backbuffer->getHeader();
        auto area = hdr.Width*hdr.Height;
        m_AccumBuffer = new Spectrum[area];
    }
    m_ThreadPool.enqueueTask(&m_DrawLoop);
}

RayTracerScene::~RayTracerScene()
{
    m_ThreadPool.shutdown();

    for(auto& vb : m_VertexCache)
    {
        delete[] vb;
    }

    for(auto* vol : m_Volumes)
    {
        DestroyPackedData(vol);
    }

    for(auto* light_src : m_LightSources)
    {
        delete light_src;
    }

    for(auto* int_geom : m_InternalGeometry)
    {
        delete int_geom;
    }

    rtcDeleteScene(m_Scene);
    rtcDeleteDevice(m_Device);

	delete[] m_AccumBuffer;
}

static void SphereBoundsFunction(const Sphere& sphere, RTCBounds& bounds)
{
    AABBUnaligned aabb;
    SphereBounds(sphere, &aabb);
    UnalignedToAlignedBounds(aabb, &bounds);
}

static void SphereShapeIntersectFunction(uint32_t geom_id, const Sphere& sphere, RTCRay& ray)
{
    if(!IntersectSphere(ToVector3(ray.dir), ToVector3(ray.org), sphere, &ray.tfar, &ray.u, &ray.v, reinterpret_cast<Vector3*>(ray.Ng)))
    {
        return;
    }

    ray.geomID = geom_id;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = 0;
}

static void EllipsoidShapeIntersectFunction(uint32_t geom_id, const Ellipsoid& ellipsoid, RTCRay& ray)
{
    if(!IntersectEllipsoid(ToVector3(ray.dir), ToVector3(ray.org), ellipsoid, &ray.tfar, &ray.u, &ray.v, reinterpret_cast<Vector3*>(ray.Ng)))
    {
        return;    
    }

    ray.geomID = geom_id;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = 0;
}

static void CylinderBoundsFunction(const Cylinder& cylinder, RTCBounds& bounds)
{
    bounds.lower_x = cylinder.Center.x - cylinder.Radius;
    bounds.lower_y = cylinder.Center.y - cylinder.Radius;
    bounds.upper_x = cylinder.Center.x + cylinder.Radius;
    bounds.upper_y = cylinder.Center.y + cylinder.Radius;

    bounds.lower_z = cylinder.Center.z - cylinder.HalfHeight;
    bounds.upper_z = cylinder.Center.z + cylinder.HalfHeight;
}

static void CylinderShapeIntersectFunction(uint32_t geom_id, const Cylinder& cylinder, RTCRay& ray)
{
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector2 dir_v2{ dir.x, dir.y };

    Vector2 dist_to_center{ cylinder.Center.x - org.x, cylinder.Center.y - org.y };
    float b = Dot(dist_to_center, dir_v2);
    float radius = cylinder.Radius;
	float c = Dot(dist_to_center, dist_to_center) - radius*radius;

	float a = Dot(dir_v2, dir_v2);
    float D = b*b - c*a;

    if(D < 0.0f || b < 0.0f)
    {
        return;
    }

	float dist_to_center_z = cylinder.Center.z - org.z;
	float sqrt_D = sqrtf(D);
    float t = (b + (2.0f*(b < sqrt_D) - 1.0f)*sqrt_D)/a;
	if(b > sqrt_D && fabsf(t*dir.z - dist_to_center_z) > cylinder.HalfHeight)
	{
		t = (b + sqrt_D)/a;
	}
    Vector2 norm_v2 = t*dir_v2 - dist_to_center;
    NormalizeSelf(&norm_v2);
    Vector3 norm{ norm_v2.x, norm_v2.y, 0.0f };

	float radius2 = cylinder.Radius*cylinder.Radius;
    float t_top = (dist_to_center_z + cylinder.HalfHeight) / dir.z;
    Vector2 dist_to_top_vec = (t_top*dir_v2 - dist_to_center);
    if(t_top >= 0.0f && t_top < t && Dot(dist_to_top_vec, dist_to_top_vec) < radius2)
    {
        t = t_top;
        norm = Vector3{ 0.0f, 0.0f, 1.0f };
    }

    float t_bottom = (dist_to_center_z - cylinder.HalfHeight) / dir.z;
    Vector2 dist_to_bottom_vec = (t_bottom*dir_v2 - dist_to_center);
    if(t_bottom >= 0.0f && t_bottom < t && Dot(dist_to_bottom_vec, dist_to_bottom_vec) < radius2)
    {
        t = t_bottom;
        norm = Vector3{ 0.0f, 0.0f, -1.0f };
    }

	if(fabsf(t*dir.z - dist_to_center_z) > cylinder.HalfHeight)
	{
		return;
	}

    if(Dot(norm, dir) > 0.0f)
        return;

    ray.u = 0.0f;
    ray.v = 0.0f;
    //ray.tnear = tmin;
    ray.tfar = t;
    ray.geomID = geom_id;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = 0;
    CopyVec3ToFloatArray(norm, ray.Ng);
}

static void ObliqueCylinderBoundsFunction(const ObliqueCylinder& oblique_cylinder, RTCBounds& bounds)
{
    bounds.lower_x = oblique_cylinder.CylinderShape.Center.x - oblique_cylinder.CylinderShape.Radius - oblique_cylinder.Tilt;
    bounds.lower_y = oblique_cylinder.CylinderShape.Center.y - oblique_cylinder.CylinderShape.Radius - oblique_cylinder.Tilt;
    bounds.upper_x = oblique_cylinder.CylinderShape.Center.x + oblique_cylinder.CylinderShape.Radius + oblique_cylinder.Tilt;
    bounds.upper_y = oblique_cylinder.CylinderShape.Center.y + oblique_cylinder.CylinderShape.Radius + oblique_cylinder.Tilt;

    bounds.lower_z = oblique_cylinder.CylinderShape.Center.z - oblique_cylinder.CylinderShape.HalfHeight;
    bounds.upper_z = oblique_cylinder.CylinderShape.Center.z + oblique_cylinder.CylinderShape.HalfHeight;
}

static void ObliqueCylinderIntersectFunction(uint32_t geom_id, const ObliqueCylinder& oblique_cylinder, RTCRay& ray)
{
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector2 dir_v2{ dir.x, dir.y };

    Vector2 bending = oblique_cylinder.Tilt*oblique_cylinder.TiltDirection/oblique_cylinder.CylinderShape.HalfHeight;

    Vector2 dist_to_center{ oblique_cylinder.CylinderShape.Center.x - org.x, oblique_cylinder.CylinderShape.Center.y - org.y };
    dist_to_center += bending*(org.z - oblique_cylinder.CylinderShape.Center.z);
        
    Vector2 a_v2 = dir_v2 - bending*dir.z;

    float b = Dot(dist_to_center, a_v2);
    float radius = oblique_cylinder.CylinderShape.Radius;
	float c = Dot(dist_to_center, dist_to_center) - radius*radius;

	float a = Dot(a_v2, a_v2);
    float D = b*b - c*a;

    if(D < 0.0f || b < 0.0f)
    {
        return;
    }
    
	float sqrt_D = sqrtf(D);
    float t = (b + (2.0f*(b < sqrt_D) - 1.0f)*sqrt_D)/a;
	
	float dist_to_center_z = oblique_cylinder.CylinderShape.Center.z - org.z;
    if(b > sqrt_D && fabsf(t*dir.z - dist_to_center_z) > oblique_cylinder.CylinderShape.HalfHeight)
	{
		t = (b + sqrt_D)/a;
	}
    Vector2 norm_v2 = t*a_v2 - dist_to_center;
    NormalizeSelf(&norm_v2);
    Vector3 norm{ norm_v2.x, norm_v2.y, 0.0f };

	float radius2 = oblique_cylinder.CylinderShape.Radius*oblique_cylinder.CylinderShape.Radius;
    float t_top = (dist_to_center_z + oblique_cylinder.CylinderShape.HalfHeight) / dir.z;
    Vector2 dist_to_top_vec = (t_top*a_v2 - dist_to_center);
    if(t_top >= 0.0f && t_top < t && Dot(dist_to_top_vec, dist_to_top_vec) < radius2)
    {
        t = t_top;
        norm = Vector3{ 0.0f, 0.0f, 1.0f };
    }

    float t_bottom = (dist_to_center_z - oblique_cylinder.CylinderShape.HalfHeight) / dir.z;
    Vector2 dist_to_bottom_vec = (t_bottom*a_v2 - dist_to_center);
    if(t_bottom >= 0.0f && t_bottom < t && Dot(dist_to_bottom_vec, dist_to_bottom_vec) < radius2)
    {
        t = t_bottom;
        norm = Vector3{ 0.0f, 0.0f, -1.0f };
    }

	if(fabsf(t*dir.z - dist_to_center_z) > oblique_cylinder.CylinderShape.HalfHeight)
	{
		return;
	}

    if(Dot(norm, dir) > 0.0f)
        return;

    ray.u = 0.0f;
    ray.v = 0.0f;
    //ray.tnear = tmin;
    ray.tfar = t;
    ray.geomID = geom_id;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = 0;
    CopyVec3ToFloatArray(norm, ray.Ng);
}

static void SphereAreaLightIntersectFunction(SphereAreaLightGeometry* light_geom, RTCRay& ray, size_t)
{
    SphereShapeIntersectFunction(light_geom->GeometryID, light_geom->Light->SphereShape, ray);
}

static void SphereAreaLightBoundsFunction(SphereAreaLightGeometry* light_geom, size_t i, RTCBounds& bounds)
{
    SphereBoundsFunction(light_geom->Light->SphereShape, bounds);
}

static void SphereGeometryIntersectFunction(SphereGeometry* geom, RTCRay& ray, size_t i)
{
    SphereShapeIntersectFunction(geom->GeometryID, geom->SphereShape, ray);
}

static void SphereGeometryBoundsFunction(SphereGeometry* geom, size_t i, RTCBounds& bounds)
{
    SphereBoundsFunction(geom->SphereShape, bounds);
}

static void EllipsoidGeometryIntersectFunction(EllipsoidGeometry* geom, RTCRay& ray, size_t i)
{
    EllipsoidShapeIntersectFunction(geom->GeometryID, geom->EllipsoidShape, ray);
}

static void EllipsoidGeometryBoundsFunction(EllipsoidGeometry* geom, size_t i, RTCBounds& bounds)
{
    UnalignedToAlignedBounds(geom->Bounds, &bounds);
}

static void CylinderGeometryIntersectFunction(CylinderGeometry* geom, RTCRay& ray, size_t i)
{
    CylinderShapeIntersectFunction(geom->GeometryID, geom->CylinderShape, ray);
}

static void CylinderGeometryBoundsFunction(CylinderGeometry* geom, size_t i, RTCBounds& bounds)
{
    CylinderBoundsFunction(geom->CylinderShape, bounds);
}

static void ObliqueCylinderGeometryIntersectFunction(ObliqueCylinderGeometry* geom, RTCRay& ray, size_t i)
{
    ObliqueCylinderIntersectFunction(geom->GeometryID, geom->ObliqueCylinderShape, ray);
}

static void ObliqueCylinderGeometryBoundsFunction(ObliqueCylinderGeometry* geom, size_t i, RTCBounds& bounds)
{
    ObliqueCylinderBoundsFunction(geom->ObliqueCylinderShape, bounds);
}

static void RectGeometryIntersectFunction(BlockerGeometry* geom, RTCRay& ray, size_t i)
{
    auto org = ToVector3(ray.org);
    auto dir = ToVector3(ray.dir);

    if(!IntersectRect3(dir, org, geom->BlockerRect, &ray.tfar, &ray.u, &ray.v, reinterpret_cast<Vector3*>(ray.Ng)))
    {
        return;
    }

    ray.geomID = geom->GeometryID;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
	ray.primID = 0;
}

static void RectGeometryBoundsFunction(BlockerGeometry* geom, size_t i, RTCBounds& bounds)
{
    AABBUnaligned aabb;
    Rect3Bounds(geom->BlockerRect, &aabb);
    UnalignedToAlignedBounds(aabb, &bounds);
}

static void DiskGeometryIntersectFunction(DiskGeometry* geom, RTCRay& ray, size_t i)
{
    auto org = ToVector3(ray.org);
    auto dir = ToVector3(ray.dir);

    if(!IntersectDisk3(dir, org, geom->Disk, &ray.tfar, &ray.u, &ray.v, reinterpret_cast<Vector3*>(ray.Ng)))
    {
        return;
    }

    ray.geomID = geom->GeometryID;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
	ray.primID = 0;
}

static void DiskGeometryBoundsFunction(DiskGeometry* geom, size_t i, RTCBounds& bounds)
{
    AABBUnaligned aabb;
    Disk3Bounds(geom->Disk, &aabb);
    UnalignedToAlignedBounds(aabb, &bounds);
}

void CylinderGeometry::sample(RTCRay& ray, SampleData* data) const
{
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector3 norm = ToVector3(ray.Ng);

    data->TexCoord = Vector2{0.0f, 0.0f};
    data->Material = Material;
    data->Normal = norm;
    if(fabsf(data->Normal.z) == 1.0f)
    {
        Vector2 pos{ org.x + ray.tfar*dir.x, org.y + ray.tfar*dir.y };
        Vector2 tan_v2 = pos - Vector2{ CylinderShape.Center.x, CylinderShape.Center.y };
        NormalizeSelf(&tan_v2);
        data->Tangent = Vector3{ tan_v2.x, tan_v2.y, 0.0f };
        // TODO: Binormal easy to compute
    }
    else
    {
        data->Tangent = Vector3{0.0f, 0.0f, 1.0f};        
    }
    data->Binormal = Cross(norm, data->Tangent);
}

void SphereGeometry::sample(RTCRay& ray, SampleData* data) const
{
    float t = ray.tfar;
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector3 norm = ToVector3(ray.Ng);

    data->TexCoord = Vector2{0.0f, 0.0f};
    data->Material = Material;
    data->Normal = norm;
    data->Binormal = Cross(Vector3{0.0f, 0.0f, 1.0f}, norm);
    NormalizeSelf(&data->Binormal);
    data->Tangent = Cross(data->Binormal, data->Normal);
    //NormalizeSelf(&data->Tangent);
}

void EllipsoidGeometry::sample(RTCRay& ray, SampleData* data) const
{
    float t = ray.tfar;
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector3 norm = ToVector3(ray.Ng);

    data->TexCoord = {0.0f, 0.0f};
    data->Material = Material;
    data->Normal = norm;


    Vector3 quat_norm = ToNormal(EllipsoidShape.Orientation);
    if(fabsf(Dot(quat_norm, norm)) < 1e-6f)
    {
        Tempest::Matrix3 basis;
        basis.makeBasis(norm);

        data->Binormal = basis.binormal();
        data->Tangent = basis.tangent();
    }
    else
    {
        data->Binormal = Normalize(Cross(quat_norm, norm));
        data->Tangent = Cross(data->Binormal, data->Normal);
    }
}

void ObliqueCylinderGeometry::sample(RTCRay& ray, SampleData* data) const
{
    Vector3 org = ToVector3(ray.org);
    Vector3 dir = ToVector3(ray.dir);
    Vector3 norm = ToVector3(ray.Ng);

    data->TexCoord = Vector2{};
    data->Material = Material;
    
    //data->Normal = norm;
    data->Normal = data->Tangent = data->Binormal = Vector3{};
    
    /*
    if(fabsf(data->Normal.z) == 1.0f)
    {
        Vector2 pos{ org.x + ray.tfar*dir.x, org.y + ray.tfar*dir.y };
        Vector2 tan_v2 = pos - Vector2{ CylinderShape.Center.x, CylinderShape.Center.y };
        NormalizeSelf(&tan_v2);
        data->Tangent = Vector3{ tan_v2.x, tan_v2.y, 0.0f };
        // TODO: Binormal easy to compute
    }
    else
    {
        data->Tangent = Vector3{0.0f, 0.0f, 1.0f};        
    }
    data->Binormal = Cross(norm, data->Tangent);
    */
}

void BlockerGeometry::sample(RTCRay& ray, SampleData* data) const
{
    data->TexCoord = Vector2{};
    data->Material = nullptr;
    data->Normal = data->Tangent = data->Binormal = Vector3{};
}

void RectGeometry::sample(RTCRay& ray, SampleData* data) const
{
    data->TexCoord = TexCoordStart + Vector2{ ray.u, ray.v }*TexCoordMultiplier;
    data->Material = Material;

    if(TangentMap)
    {
        Tempest::Matrix3 rel_tangent_space;
        auto tangent = TangentMap->sampleRGB(data->TexCoord);
        NormalizeSelf(&tangent);
        rel_tangent_space.makeBasisTangent(tangent);

        Matrix3 mod_tangent_space = ToMatrix3(Rect.Orientation);
        mod_tangent_space *= rel_tangent_space;

        data->Tangent = mod_tangent_space.tangent();
        data->Binormal = mod_tangent_space.binormal();
        data->Normal = mod_tangent_space.normal();
    }
    else
    {
        Matrix3 tangent_space = ToMatrix3(Rect.Orientation);
        data->Tangent = tangent_space.tangent();
        data->Binormal = tangent_space.binormal();
        data->Normal = tangent_space.normal();
    }
}

void DiskGeometry::sample(RTCRay& ray, SampleData* data) const
{
    auto intersect_pos = ray.tfar*ToVector3(ray.dir) + ToVector3(ray.org);
    data->TexCoord = Vector2{ ray.u, ray.v };
    data->Material = Material;
    data->Binormal = Disk.Center - intersect_pos;
    NormalizeSelf(&data->Binormal);
    data->Tangent = Cross(data->Binormal, Disk.Normal);
    data->Normal = Disk.Normal;
}

unsigned RayTracerScene::addSphereLightSource(SphereAreaLight* light)
{
    m_LightSources.push_back(light);
    
    auto geom_id = light->GeometryID = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new SphereAreaLightGeometry(geom_id);
    geom->Material.Model = IlluminationModel::Emissive;
    geom->Material.Diffuse = light->Radiance;
    geom->Light = light;
    m_InternalGeometry.push_back(geom);

    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&SphereAreaLightBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&SphereAreaLightIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&SphereAreaLightIntersectFunction);
	return geom_id;
}

void BackfaceCulling(void*, RTCRay& ray)
{
    if(Dot(ToVector3(ray.Ng), ToVector3(ray.dir)) <= 0.0f)
        ray.geomID = RTC_INVALID_GEOMETRY_ID;
}

void TwoSidedBias(void*, RTCRay& ray)
{
    if(ray.tfar < 1e-3f)
        ray.geomID = RTC_INVALID_GEOMETRY_ID;
}

uint64_t RayTracerScene::addEllipsoid(const Ellipsoid& ellipsoid, RTMaterial* material, const RTObjectSettings* settings)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new EllipsoidGeometry(geom_id);
    geom->EllipsoidShape = ellipsoid;
    geom->Material = material;

    Tempest::Matrix4 mat(ellipsoid.Orientation, ellipsoid.Center);
    mat.scale(ellipsoid.Scale);

    Tempest::Vector3 extend{ Length(mat.transposeRelativeX()),
                             Length(mat.transposeRelativeY()),
                             Length(mat.transposeRelativeZ()) };

    auto translation = mat.translation();
    geom->Bounds.MinCorner = translation - extend;
    geom->Bounds.MaxCorner = translation + extend;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&EllipsoidGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&EllipsoidGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc)&EllipsoidGeometryIntersectFunction);
    return geom_id;
}

uint64_t RayTracerScene::addSphere(const Sphere& sphere, RTMaterial* material)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new SphereGeometry(geom_id);
    geom->SphereShape = sphere;
    geom->Material = material;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&SphereGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&SphereGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&SphereGeometryIntersectFunction);
	return geom_id;
}

uint64_t RayTracerScene::addCylinder(const Cylinder& cylinder, RTMaterial* material)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new CylinderGeometry(geom_id);
    geom->CylinderShape = cylinder;
    geom->Material = material;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&CylinderGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&CylinderGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&CylinderGeometryIntersectFunction);
	return geom_id;
}

uint64_t RayTracerScene::addObliqueCylinder(const ObliqueCylinder& cylinder, RTMaterial* material)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new ObliqueCylinderGeometry(geom_id);
    geom->ObliqueCylinderShape = cylinder;
    geom->Material = material;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&ObliqueCylinderGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&ObliqueCylinderGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&ObliqueCylinderGeometryIntersectFunction);
	return geom_id;
}

uint64_t RayTracerScene::addBlocker(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new BlockerGeometry(geom_id);
    geom->BlockerRect.Center = pos;
    geom->BlockerRect.Orientation = Tempest::ToQuaternion(Matrix3(tan, Cross(norm, tan), norm));
    geom->BlockerRect.Size = size;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&RectGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&RectGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&RectGeometryIntersectFunction);
	return geom_id;
}

uint64_t RayTracerScene::addRect(const Vector3& pos, const Vector3& tan, const Vector3& norm, const Vector2& size, RTMaterial* material, const AABB2* tc, const void* tangent_map)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new RectGeometry(geom_id);
    geom->Material = material;
    geom->Rect.Center = pos;
    geom->Rect.Orientation = Tempest::ToQuaternion(Matrix3(tan, Cross(norm, tan), norm));
    geom->Rect.Size = size;
    if(tc)
    {
        geom->TexCoordStart = tc->MinCorner;
        geom->TexCoordMultiplier = tc->MaxCorner - tc->MinCorner;
    }
    else
    {
        geom->TexCoordStart = { 0.0f, 0.0f };
        geom->TexCoordMultiplier = { 1.0f, 1.0f };
    }
    geom->TangentMap = reinterpret_cast<const Texture*>(tangent_map);

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&RectGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&RectGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&RectGeometryIntersectFunction);
	return geom_id;
}

uint64_t RayTracerScene::addDisk(const Vector3& pos, const Vector3& norm, float inner_radius, float outer_radius, RTMaterial* material)
{
    auto geom_id = rtcNewUserGeometry(m_Scene, 1);
    auto geom = new DiskGeometry(geom_id);
    geom->Material = material;
    geom->Disk.Center = pos;
    geom->Disk.Normal = norm;
    geom->Disk.InnerRadius = inner_radius;
    geom->Disk.OuterRadius = outer_radius;

    m_InternalGeometry.push_back(geom);
    rtcSetUserData(m_Scene, geom_id, geom);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&DiskGeometryBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&DiskGeometryIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&DiskGeometryIntersectFunction);
	return geom_id;
}

void RayTracerScene::addHierarchicalVolume(VolumeRoot* hi_volume, RTMaterial* material)
{
    size_t volume_count = hi_volume->Dimensions.X*hi_volume->Dimensions.Y*hi_volume->Dimensions.Z;
    size_t non_empty_vol = std::count_if(hi_volume->Volumes, hi_volume->Volumes + volume_count, [](const Volume& volume) { return volume.Data != nullptr; });
    auto geom_id = rtcNewUserGeometry(m_Scene, non_empty_vol);

    auto start_idx = m_Volumes.size();

    Vector3 vol_size = (hi_volume->MaxCorner - hi_volume->MinCorner)/ToVector3(hi_volume->Dimensions);

    auto rt_vol = CreatePackedData<RTVolumeSet>(static_cast<uint32_t>(non_empty_vol), geom_id);
	rt_vol->VolumeGrid = new RTVolume*[volume_count];
    rt_vol->Material = material;

	const size_t density_size = 1;
	const size_t angle_size = 2;
	const size_t data_size = density_size + angle_size;
    const size_t total_data_size = density_size + angle_size;

    rt_vol->RcpMaxExtinction = 1.0f / DensityScale; // TODO TODO TODO divide by tabulated integral

    rt_vol->MinCorner = hi_volume->MinCorner;
    rt_vol->MaxCorner = hi_volume->MaxCorner;
	rt_vol->Dimensions = hi_volume->Dimensions;

    size_t vol_idx = 0;
    for(size_t off = 0, off_end = volume_count; off < off_end; ++off)
    {
        auto& orig_vol = hi_volume->Volumes[off];
        if(orig_vol.Data == nullptr)
		{
			rt_vol->VolumeGrid[off] = nullptr;
            continue;
		}
        size_t x =  off %  hi_volume->Dimensions.X;
        size_t y = (off /  hi_volume->Dimensions.X) % hi_volume->Dimensions.Y;
        size_t z =  off / (hi_volume->Dimensions.X  * hi_volume->Dimensions.Y);

        TGE_ASSERT(vol_idx < non_empty_vol, "Invalid non-empty volume cell count");
		auto vol = rt_vol->VolumeGrid[off] = rt_vol->Volumes.Values + vol_idx;
		++vol_idx;

        size_t volume = orig_vol.Dimensions.X*orig_vol.Dimensions.Y*orig_vol.Dimensions.Z;
        vol->MinCorner = Box{ (int32_t)x, (int32_t)y, (int32_t)z }*vol_size + hi_volume->MinCorner;
        vol->MaxCorner = vol->MinCorner + vol_size;
        vol->Dimensions = orig_vol.Dimensions;
        vol->GridData = new uint8_t[data_size*volume];

		memcpy(vol->GridData, orig_vol.Data, data_size*volume);
    }
    rtcSetUserData(m_Scene, geom_id, rt_vol);
    rtcSetBoundsFunction(m_Scene, geom_id, (RTCBoundsFunc)&HierarchicalVolumeBoundsFunction);
    rtcSetIntersectFunction(m_Scene, geom_id, (RTCIntersectFunc)&HierarchicalVolumeIntersectFunction);
    rtcSetOccludedFunction(m_Scene, geom_id, (RTCOccludedFunc )&HierarchicalVolumeIntersectFunction);

    m_Volumes.push_back(rt_vol);
}

void RayTracerScene::addTriangleMesh(const Matrix4& world,
		                             size_t submesh_count, 
			                         RTSubmesh* submeshes,
				                     size_t index_count, int32_t* tris,
					                 size_t vert_size, void* verts,
                                     MeshOptions* mesh_opts,
									 uint64_t* geom_ids)
{
    uint8_t* vb = vb = new uint8_t[vert_size];
    memcpy(vb, verts, vert_size);

    TGE_ASSERT(vb, "Invalid vertex buffer");
    m_VertexCache.push_back(vb);
	
	auto sub_scene = rtcDeviceNewScene(m_Device, RTC_SCENE_STATIC, RTC_INTERSECT1);

	unsigned inst_id = rtcNewInstance(m_Scene, sub_scene);

    for(size_t submesh_idx = 0; submesh_idx < submesh_count; ++submesh_idx)
    {
        auto& submesh = submeshes[submesh_idx];

        auto tri_count = submesh.VertexCount/3;
        unsigned geom_id = rtcNewTriangleMesh(sub_scene, RTC_GEOMETRY_STATIC, tri_count, (vert_size - submesh.VertexOffset)/submesh.Stride);
		
        uint8_t* vb_data = vb + submesh.VertexOffset;
        auto* ib_data = tris + submesh.BaseIndex;
        rtcSetBuffer(sub_scene, geom_id, RTC_INDEX_BUFFER, ib_data, 0, 3*sizeof(int32_t));
        rtcSetBuffer(sub_scene, geom_id, RTC_VERTEX_BUFFER, vb_data, 0, submesh.Stride);
        auto* geom_desc = new RTMesh(geom_id, ib_data, vb_data, tri_count, (uint32_t)submesh.Stride, submesh.Material);
		if(mesh_opts)
        {
			geom_desc->Sampler = mesh_opts->GeometrySampler;
            geom_desc->UserData = mesh_opts->UserData;
        }
        rtcSetUserData(sub_scene, geom_id, geom_desc);
		if(mesh_opts && mesh_opts->TwoSided)
		{
			rtcSetIntersectionFilterFunction(sub_scene, geom_id, &TwoSidedBias);
			rtcSetOcclusionFilterFunction(sub_scene, geom_id, &TwoSidedBias);
		}
		else
		{
			rtcSetIntersectionFilterFunction(sub_scene, geom_id, &BackfaceCulling);
			rtcSetOcclusionFilterFunction(sub_scene, geom_id, &BackfaceCulling);
		}

        // Extra processing for emissive materials
        if(SampleIncidentLightLookup[(size_t)geom_desc->Material->Model] == nullptr)
        {
            auto mesh_light = new MeshLight;
            mesh_light->IndexBuffer = { reinterpret_cast<uintptr_t>(ib_data) };
            mesh_light->VertexBuffer = { reinterpret_cast<uintptr_t>(vb_data) };
            mesh_light->Stride = submesh.Stride;
            mesh_light->Material = { reinterpret_cast<uintptr_t>(submesh.Material) }; 
            mesh_light->TriangleCount = tri_count;
            mesh_light->GeometryID = geom_id;
            m_LightSources.push_back(mesh_light);
        }

		m_InternalGeometry.push_back(geom_desc);
		if(geom_ids)
		{
			geom_ids[submesh_idx] = ((uint64_t)inst_id << 32ULL) | geom_id;
		}

		auto ptr = rtcGetUserData(sub_scene, geom_id);
		TGE_ASSERT(ptr != nullptr, "Invalid scene");
    }

	rtcCommit(sub_scene);

	Tempest::Vector3 world_v3[4] = { world.relativeX(), world.relativeY(), world.relativeZ(), world.translation() };

	rtcSetUserData(m_Scene, inst_id, sub_scene);
	rtcSetTransform(m_Scene, inst_id, RTC_MATRIX_COLUMN_MAJOR, reinterpret_cast<const float*>(&world_v3));
}

void RayTracerScene::addHair(const Matrix4& world,
		                     size_t submesh_count, 
			                 RTSubmesh* submeshes,
				             size_t curve_count, int32_t* curves,
					         size_t vert_size, void* verts, size_t stride,
							 unsigned* geom_ids)
{
    uint8_t* vb = nullptr;

    size_t vert_count = vert_size/stride;

    TGE_ASSERT(stride == sizeof(HairFormat), "Invalid texture format");
    static_assert(sizeof(HairFormat) == 4*sizeof(float), "Invalid vertex format size");

    vb = new uint8_t[vert_size];
    auto* out_data = reinterpret_cast<HairFormat*>(vb);
    auto* in_data = reinterpret_cast<HairFormat*>(verts);

    for(size_t i = 0; i < vert_count; ++i)
    {
        auto& in_ = in_data[i];
        auto& out_ = out_data[i];

        out_.Position = world*in_.Position;
        auto scaling = world.scaling();
        out_.Radius = in_.Radius*scaling.x; // Oh, well - assume uniform scaling. TODO: Make tests for hairs with eccentricity 
    }

    TGE_ASSERT(vb, "Invalid vertex buffer");
    m_VertexCache.push_back(vb);

    for(size_t submesh_idx = 0; submesh_idx < submesh_count; ++submesh_idx)
    {
        auto& submesh = submeshes[submesh_idx];
        unsigned geom_id = rtcNewHairGeometry(m_Scene, RTC_GEOMETRY_STATIC, submesh.VertexCount/4, (vert_size - submesh.VertexOffset)/stride);
        uint8_t* vb_data = vb + submesh.VertexOffset;
        auto* ib_data = curves + submesh.BaseIndex;
        rtcSetBuffer(m_Scene, geom_id, RTC_INDEX_BUFFER, ib_data, 0, sizeof(int32_t));
        rtcSetBuffer(m_Scene, geom_id, RTC_VERTEX_BUFFER, vb_data, 0, stride);
        auto* geom_desc = new RTHair(geom_id, ib_data, vb_data, (uint32_t)stride, submesh.Material);
        rtcSetUserData(m_Scene, geom_id, geom_desc);
        m_InternalGeometry.push_back(geom_desc);
		if(geom_ids)
		{
			geom_ids[submesh_idx] = geom_id;
		}
    }
}

void RayTracerScene::commitScene()
{
    rtcCommit(m_Scene);
}

void RayTracerScene::postprocess()
{
    if(m_PicturePostProcess == PicturePostProcess::AutoExposureHDR ||
       m_PicturePostProcess == PicturePostProcess::AutoExposureSRGB)
    {
        FrameData& frame_data = m_FrameData[m_FrameIndex];
        auto* backbuffer = frame_data.Backbuffer.get();
        auto& hdr = backbuffer->getHeader();
        uint32_t width = hdr.Width,
                 height = hdr.Height;

        auto log_total_value = m_LogAverages[0];
        for(uint32_t idx = 1; idx < m_WorkerThreads; ++idx)
            log_total_value += m_LogAverages[idx];

        auto area = width*height;
        float mean_value = RGBToLuminance(Vector3Exp(log_total_value / (float)area));

        auto data = reinterpret_cast<uint32_t*>(backbuffer->getData());
        float exp_mod = 0.18f;
        float exp_factor = exp_mod / mean_value;
        auto accum = m_AccumBuffer;
        switch(m_PicturePostProcess)
        {
        case PicturePostProcess::AutoExposureHDR:
        {
            auto color_convert = Tempest::CreateParallelForLoop2D(width, height, 64,
                [data, width, exp_factor, accum](uint32_t worker_id, uint32_t x, uint32_t y)
                {
                    auto value = exp_factor*accum[y*width + x];

                    auto tone_mapped = ReinhardOperator(value);

                    Vector4 v4_color{ tone_mapped.x, tone_mapped.y, tone_mapped.z, 1.0f };
        
                    data[y*width + x] = ToColor(ConvertLinearToSRGB(v4_color));
                });

            m_ThreadPool.enqueueTask(&color_convert);
            m_ThreadPool.waitAndHelp(m_ThreadId, &color_convert);
        } break;
        case PicturePostProcess::AutoExposureSRGB:
        {
            auto color_convert = Tempest::CreateParallelForLoop2D(width, height, 64,
                [data, width, exp_factor, accum](uint32_t worker_id, uint32_t x, uint32_t y)
                {
                    auto value = exp_factor*accum[y*width + x];

                    Vector4 v4_color{ value.x, value.y, value.z, 1.0f };
        
                    data[y*width + x] = ToColor(ConvertLinearToSRGB(v4_color));
                });

            m_ThreadPool.enqueueTask(&color_convert);
            m_ThreadPool.waitAndHelp(m_ThreadId, &color_convert);
        }
        }
    }
}

const FrameData* RayTracerScene::drawOnce()
{
	m_ThreadPool.waitAndHelp(m_ThreadId, &m_DrawLoop);
    postprocess();
    return m_FrameData + m_FrameIndex;
}

const FrameData* RayTracerScene::draw(uint32_t width, uint32_t height, const Matrix4& view_proj_inv)
{
    m_ThreadPool.waitAndHelp(m_ThreadId, &m_DrawLoop);

    postprocess();

    FrameData* prev_frame = m_FrameData + m_FrameIndex;

    // TODO: Triple buffering ?
    auto next_frame = (m_FrameIndex + 1) % Buffering;

    {
		uint64_t frame_end = m_FrameTimer.time();
        m_FrameStart = frame_end;

        auto thread_count = m_ThreadPool.getThreadCount();
        if(m_WorkerThreads != thread_count)
        {
            m_LogAverages = decltype(m_LogAverages)(new Spectrum[thread_count]);
            std::fill(m_LogAverages.get(), m_LogAverages.get() + thread_count, Spectrum{});
            m_WorkerThreads = thread_count;
        }

        FrameData& frame_data = m_FrameData[next_frame];
        TGE_ASSERT(frame_data.Counter.is_lock_free(), "Bad performance for this atomic type");
        frame_data.Counter.store(0, std::memory_order::memory_order_relaxed); // Because mutex
        auto& hdr = frame_data.Backbuffer->getHeader();
        if(width != hdr.Width ||
           height != hdr.Height)
        {
			delete[] m_AccumBuffer;
			uint32_t area = width*height;

            TextureDescription tex_desc;
            tex_desc.Width = width;
            tex_desc.Height = height;
            tex_desc.Format = DataFormat::RGBA8UNorm;

            frame_data.Backbuffer->realloc(tex_desc);

			m_DrawLoop.setTotalCount(area);

			m_AccumBuffer = new Spectrum[area];
			m_AccumData = 0;
        }
        else if(!m_AccumBuffer)
        {
            m_AccumBuffer = new Spectrum[width*height];
            m_AccumData = 0;
        }

		else if(frame_data.ViewProjectionInverse != view_proj_inv)
		{
			m_AccumData = 0;
		}
		else
		{
			m_AccumData++;
		}

        frame_data.ViewProjectionInverse = view_proj_inv;
        m_FrameIndex = next_frame;
    }
    
	m_DrawLoop.reset(width*height);
    m_ThreadPool.enqueueTask(&m_DrawLoop);

    return prev_frame;
}

void SphereAreaLightGeometry::sample(RTCRay& ray, SampleData* data) const
{
    data->Material = &Material;
    Vector3 norm = ToVector3(ray.org) + ToVector3(ray.dir)*ray.tfar - Light->SphereShape.Center;
    NormalizeSelf(&norm);
    data->Normal = norm;
	data->Tangent = Vector3{0.0f, 0.0f, 0.0f};
	data->Binormal = Vector3{0.0f, 0.0f, 0.0f};
}

#ifdef ENABLE_MIS_POWER_HEURISTIC
#	define BalanceBias(x) (x)*(x)
#else
#	define BalanceBias(x) (x)
#endif

float EvaluateTotalBalanceHeuristic(const std::vector<LightSource*>& light_sources, float max_light_samples, float max_surface_samples, const SampleData& sample, const Vector3& ray_pos)
{
    float total = 0.0f;
    for(auto& light_src : light_sources)
    {
        float samples = IsAreaLight(light_src->Type) ? max_light_samples : 1;
        float pdf = LightSourceComputePDFLookup[(size_t)light_src->Type]({}, *light_src, ray_pos, sample.IncidentLight);
        total += BalanceBias(pdf*samples);
        TGE_ASSERT(std::isfinite(total), "Bad probability");
    }
    total += BalanceBias(PDFLookup[(size_t)sample.Material->Model](sample) * max_surface_samples);
    TGE_ASSERT(std::isfinite(total), "Bad probability");
    TGE_ASSERT(total >= 0.0f, "Bad probability");

    return total;
}

// TODO: Shirley, et al or similar approach to sampling multiple light sources
Spectrum EvaluateLightSources(RTCScene& root_scene, const std::vector<LightSource*>& light_sources, const SampleData& sample_data, uint32_t max_light_samples, uint32_t max_surface_samples, const Vector3& ray_pos, const Spectrum& throughput, bool apply_balance, unsigned& mirand)
{
	Spectrum radiance{};
	for(auto& light_src : light_sources)
    {
		uint32_t samples = IsAreaLight(light_src->Type) != 0 ? max_light_samples : 1;
		Spectrum light_source_color{};

        SampleData light_sample = sample_data;

        Stratification strata; // TODO: Why is it unused?
        strata.XStrata = 0;
        strata.YStrata = 0;
        strata.TotalXStrata = strata.TotalYStrata = 1;

		Vector3 cur_ray_pos = ray_pos;
		for(uint32_t k = 0; k < samples; ++k)
		{
            auto light_smp = SampleLightSourceFunctionLookup[(size_t)light_src->Type]({}, *light_src, ray_pos, strata, mirand);
            if(light_smp.PDF == 0.0f)
                continue;
			RTCRay ray;
			CopyVec3ToFloatArray(cur_ray_pos, ray.org);
			CopyVec3ToFloatArray(light_smp.IncidentLight, ray.dir);
			ray.tnear = 0.0f;
			ray.tfar = light_smp.Distance;
			ray.geomID = RTC_INVALID_GEOMETRY_ID;
			ray.primID = RTC_INVALID_GEOMETRY_ID;
			ray.instID = RTC_INVALID_GEOMETRY_ID;
			ray.mask = -1;
			ray.time = 0;

			light_sample.IncidentLight = light_smp.IncidentLight;

			// Basically, accumulate extinction along the path
			float step_through_dist = 0.0f;
			Spectrum cur_throughput = throughput;
			for(;;)
			{
				rtcIntersect(root_scene, ray);

				if(ray.geomID == RTC_INVALID_GEOMETRY_ID || ray.geomID == light_smp.GeometryID)
				{
                    if(apply_balance && IsAreaLight(light_src->Type))
                    {
					    float balance = 0.5f;
				    #ifdef ENABLE_MIS
                        float total_probability = EvaluateTotalBalanceHeuristic(light_sources, (float)max_light_samples, (float)max_surface_samples, light_sample, ray_pos);
                        balance = BalanceBias(LightSourceComputePDFLookup[(size_t)light_src->Type]({}, *light_src, ray_pos, light_sample.IncidentLight) * samples) / total_probability;
                    #endif
					    cur_throughput *= balance;
                    }
					light_source_color += light_smp.Radiance * cur_throughput * TransmittanceLookup[(size_t)light_sample.Material->Model](light_sample) / light_smp.PDF;
                    TGE_ASSERT(std::isfinite(Array(light_source_color)[0]), "Invalid transmittance");
					break;
				}
                
				auto scene = (ray.instID & (1 << 31)) == 0 ? (RTCScene)rtcGetUserData(root_scene, ray.instID) : root_scene;
	
				auto obs_geom = (RTGeometry*)rtcGetUserData(scene, ray.geomID);
				if(!obs_geom)
				{
					break;
				}

				Tempest::SampleData obs_sample;
				obs_sample.IncidentLight = -light_sample.IncidentLight;
				obs_geom->sample(ray, &obs_sample);
				auto obs_material = obs_sample.Material;
				if(obs_material == nullptr)
				{
					break;
				}

				// Basically not participating media
				if(!IsVolumeIlluminationModel(obs_material->Model))
					break;

				RTVolumeSet* volume = (RTVolumeSet*)obs_geom;

				size_t woodcock_rep = 2;
				size_t success = 0;

				for(size_t rep = 0; rep < woodcock_rep; ++rep)
				{
					float step_dist = step_through_dist;
					float threshold;
					do
					{
						step_dist -= (logf(FastFloatRand(mirand)) * volume->RcpMaxExtinction);
						Vector3 step_through_pos = cur_ray_pos + light_smp.IncidentLight * step_dist;

						threshold = FastFloatRand(mirand);
						if(!volume->sampleScattering(threshold, step_through_pos, &obs_sample))
						{
							step_through_dist = step_dist;
							++success;
							break;
						}
						obs_sample.TotalDensity *= volume->RcpMaxExtinction;
					} while(obs_sample.TotalDensity < threshold);
				}

				cur_throughput *= (float)success/woodcock_rep;
				if(cur_throughput < TransmittanceThreshold)
				{
					break;
				}

				ray.tnear = std::min(light_smp.Distance - 1e-3f, step_through_dist); // TODO: Not really
				ray.tfar = light_smp.Distance;
				ray.geomID = RTC_INVALID_GEOMETRY_ID;
				ray.primID = RTC_INVALID_GEOMETRY_ID;
				ray.instID = RTC_INVALID_GEOMETRY_ID;
				ray.mask = -1;
			}
		}

		radiance += light_source_color/(float)samples;
    }
	return radiance;
}

uint64_t RayTracerScene::rayQuery(const Vector2& tc, SampleData* sample_data)
{
	Vector4 screen_tc{2.0f*tc.x - 1.0f, 1.0f - 2.0f*tc.y, -1.0f, 1.0};

	auto& frame_data = m_FrameData[m_FrameIndex];
	Vector4 pos_start = frame_data.ViewProjectionInverse*screen_tc;

	screen_tc.z = 1.0f;
	Vector4 pos_end = frame_data.ViewProjectionInverse*screen_tc;

	Vector3 start_ray_pos = ToVector3(pos_start);
	Vector3 end_ray_pos = ToVector3(pos_end);

	RTCRay ray;

	auto dir = end_ray_pos - start_ray_pos;
	NormalizeSelf(&dir);
	CopyVec3ToFloatArray(dir, ray.dir);

	CopyVec3ToFloatArray(start_ray_pos, ray.org);
	ray.tnear = 0.0f;
	ray.tfar = INFINITY;
	ray.geomID = RTC_INVALID_GEOMETRY_ID;
	ray.primID = RTC_INVALID_GEOMETRY_ID;
	ray.instID = RTC_INVALID_GEOMETRY_ID;
	ray.mask = -1;
	ray.time = 0;

	rtcIntersect(m_Scene, ray);
	if(ray.geomID == RTC_INVALID_GEOMETRY_ID)
	{
		return RTC_INVALID_GEOMETRY_ID;
	}	
	memset(sample_data, 0, sizeof(*sample_data));

	auto scene = (ray.instID & (1 << 31)) == 0 ? (RTCScene)rtcGetUserData(m_Scene, ray.instID) : m_Scene;

	auto geom_desc = (RTGeometry*)rtcGetUserData(scene, ray.geomID);
	if(!geom_desc)
	{
		sample_data->TexCoord = Vector2{ ray.u, ray.v };
		sample_data->Normal = *reinterpret_cast<Vector3*>(ray.Ng);
		return RTC_INVALID_GEOMETRY_ID;
	}
            
	geom_desc->sample(ray, sample_data);
	return ((uint64_t)ray.instID << 32ULL) | ray.geomID;
}

void RayTracerScene::subdraw(uint32_t worker_id, uint32_t p, uint32_t chunk_size)
{
    auto& frame_data = m_FrameData[m_FrameIndex];
    auto& hdr = frame_data.Backbuffer->getHeader();
    uint32_t pixels = hdr.Width*hdr.Height;
    
    static volatile unsigned rand_var = 1;
    std::hash<std::thread::id> hasher;
    unsigned int mirand = rand_var + (unsigned)hasher(std::this_thread::get_id()) + p;

	float mixing = (float)m_AccumData/(m_AccumData + 1);
	auto accum_buffer = m_AccumBuffer;

    float rays_sqr = sqrtf((float)m_SamplesGlobalIllumination);
    uint32_t rays_sq = (uint32_t)rays_sqr;
    TGE_ASSERT(rays_sqr - (float)rays_sq < 1e-3f, "Invalid ray count");

    float light_rays_sqr = sqrtf((float)m_SamplesLocalAreaLight);
    uint32_t light_rays_sq = (uint32_t)light_rays_sqr;
    TGE_ASSERT(light_rays_sqr - (float)light_rays_sq < 1e-3, "Invalid light ray count");

    for(uint32_t chunk_end = std::min(p + chunk_size, pixels); p < chunk_end; ++p)
    {
        uint32_t x = p % hdr.Width;
        uint32_t y = p / hdr.Width;
        RTCRay ray;
	
        uint32_t intersections = 0;
		Spectrum final_light{};

		for(uint32_t camera_ray = 0; camera_ray < m_SamplesCamera; ++camera_ray)
		{
			// TODO: Replace with the faster computation
			Vector4 screen_tc;
			if(m_SamplesCamera > 1)
			{
				screen_tc = Vector4{2.0f*(x + FastFloatRand(mirand))/hdr.Width - 1.0f, 2.0f*(y + FastFloatRand(mirand))/hdr.Height - 1.0f, -1.0f, 1.0};
			}
			else
			{
				screen_tc = Vector4{2.0f*x/(hdr.Width - 1) - 1.0f, 2.0f*y/(hdr.Height - 1) - 1.0f, -1.0f, 1.0};
			}

			Vector4 pos_start = frame_data.ViewProjectionInverse*screen_tc;

			screen_tc.z = 1.0f;
			Vector4 pos_end = frame_data.ViewProjectionInverse*screen_tc;

			Vector3 start_ray_pos = ToVector3(pos_start);
			Vector3 end_ray_pos = ToVector3(pos_end);

			SampleData start_data;
			start_data.ScratchMemory = m_ScratchMemory.get() + worker_id*ScratchMemoryPerThread;
			start_data.IncidentLight = end_ray_pos - start_ray_pos;
			NormalizeSelf(&start_data.IncidentLight);

			CopyVec3ToFloatArray(start_data.IncidentLight, ray.dir);

			const RTMaterial* material = nullptr;
			{
			bool free_path, end_ray = false;

			do
			{
				free_path = false;

				CopyVec3ToFloatArray(start_ray_pos, ray.org);
				ray.tnear = 0.0f;
				ray.tfar = INFINITY;
				ray.geomID = RTC_INVALID_GEOMETRY_ID;
				ray.primID = RTC_INVALID_GEOMETRY_ID;
				ray.instID = RTC_INVALID_GEOMETRY_ID;
				ray.mask = -1;
				ray.time = 0;

				rtcIntersect(m_Scene, ray);

				if(ray.geomID == RTC_INVALID_GEOMETRY_ID)
				{
					final_light += m_GlobalCubeMap ? m_GlobalCubeMap->sampleSpectrum(start_data.IncidentLight) : m_BackgroundSpectrum;
					end_ray = true;
					break;
				}
				 
                ++intersections;

				auto scene = (ray.instID & (1 << 31)) == 0 ? (RTCScene)rtcGetUserData(m_Scene, ray.instID) : m_Scene;
	
				auto geom_desc = (RTGeometry*)rtcGetUserData(scene, ray.geomID);
				if(!geom_desc)
				{
					end_ray = true;
					break;
				}
            
				geom_desc->sample(ray, &start_data);

                if(m_RenderMode != RenderMode::Normal)
                {
                    switch(m_RenderMode)
                    {
                    case RenderMode::DebugTangents: final_light += RGBToSpectrum(start_data.Tangent*0.5f + 0.5f); break;
                    case RenderMode::DebugBinormals: final_light += RGBToSpectrum(start_data.Binormal*0.5f + 0.5f); break;
                    case RenderMode::DebugNormals: final_light += RGBToSpectrum(start_data.Normal*0.5f + 0.5f); break;
                    case RenderMode::DebugLighting: final_light += Tempest::ToSpectrum(Maxf(0.0f, Dot(start_data.Normal, -start_data.IncidentLight))); break;
                    default: TGE_ASSERT(false, "Unsupported render mode");
                    }
                    end_ray = true;
                    break;
                }
				
				material = start_data.Material;
				if(material == nullptr)
				{
					end_ray = true;
					break;
				}

				start_ray_pos += ray.tfar*start_data.IncidentLight;
                start_data.Position = start_ray_pos;

				auto cache_function = MaterialCacheLookup[(size_t)material->Model];
				if(cache_function)
					cache_function(start_data, mirand);

				if(!IsVolumeIlluminationModel(material->Model))
					break;

				RTVolumeSet* volume = static_cast<RTVolumeSet*>(geom_desc);
				
				float threshold;
				do
				{
					float d = logf(FastFloatRand(mirand)) * volume->RcpMaxExtinction;
					start_ray_pos -= start_data.IncidentLight * d; 
					threshold = FastFloatRand(mirand);
					if(!volume->sampleScattering(threshold, start_ray_pos, &start_data))
					{
						free_path = true;
						break;
					}
					start_data.TotalDensity *= volume->RcpMaxExtinction;
				} while(start_data.TotalDensity < threshold);

			} while(free_path);

			if(end_ray)
			{
				continue;
			}
			}
			Spectrum global_light{};

			Spectrum start_throughput = ToSpectrum(1.0f);
			if(IsVolumeIlluminationModel(material->Model))
			{
                auto volume_material = static_cast<const RTVolumeMaterial*>(material);
				// albedo * density / density
				start_throughput *= volume_material->Albedo;
				if(start_throughput < TransmittanceThreshold)
					continue;
			}

			start_data.OutgoingLight = -start_data.IncidentLight;

			Spectrum local_light{};

			if(SampleIncidentLightLookup[(size_t)material->Model] == nullptr) // It is definitely emissive or wrong
			{
                TGE_ASSERT(PDFLookup[(size_t)material->Model] == nullptr, "Something is wrong with this material");
                auto transmittance_func = TransmittanceLookup[(size_t)material->Model];
                if(transmittance_func)
                {
                    final_light += start_throughput*transmittance_func(start_data);
                }
                else
                {
                    final_light += start_throughput*static_cast<const RTMicrofacetMaterial*>(material)->Diffuse;
                }
				continue;
			}

            if(!MirrorCheck(start_data))
            {
				local_light += EvaluateLightSources(m_Scene, m_LightSources, start_data, m_SamplesLocalAreaLight, m_SamplesGlobalIllumination, start_ray_pos, start_throughput, m_MaxRayDepth != 0 && !IsVolumeIlluminationModel(material->Model), mirand);
            }

			//*
			for(uint32_t ray_idx = 0; ray_idx < m_SamplesGlobalIllumination; ++ray_idx)
			{
				SampleData cur_sample_data = start_data;
				uint32_t surface_samples = m_SamplesGlobalIllumination;
				uint32_t light_samples = m_SamplesLocalAreaLight;
				Vector3 cur_ray_pos = start_ray_pos;
				Spectrum cur_throughput = start_throughput;

                Stratification strata;
				strata.XStrata = float(ray_idx % rays_sq);
				strata.YStrata = float(ray_idx / rays_sq);
				strata.TotalXStrata = strata.TotalYStrata = (float)rays_sq;

				for(uint32_t ray_depth = 0; ray_depth < m_MaxRayDepth; ++ray_depth)
				{
					SampleIncidentLightLookup[(size_t)cur_sample_data.Material->Model](strata, &cur_sample_data, mirand);
					if(cur_sample_data.PDF == 0.0f)
						break;

					float balance = 1.0f;
					if(!IsVolumeIlluminationModel(material->Model))
					{
                        if(!MirrorCheck(cur_sample_data))
                        {
						#ifdef ENABLE_MIS
							float total_balance = EvaluateTotalBalanceHeuristic(m_LightSources, (float)light_samples, (float)surface_samples, cur_sample_data, cur_ray_pos);
							TGE_ASSERT(BalanceBias(surface_samples * cur_sample_data.PDF) <= total_balance, "Bad PDF");
							balance = total_balance > 0.0f ? BalanceBias(surface_samples * cur_sample_data.PDF) / total_balance : 0.0f;
						#else
							balance = 0.5f;
						#endif
                        }
						cur_throughput *= TransmittanceLookup[(size_t)cur_sample_data.Material->Model](cur_sample_data) / cur_sample_data.PDF;
					}

					CopyVec3ToFloatArray(cur_sample_data.IncidentLight, ray.dir);

					{
					bool free_path, end_ray = false;
					do
					{
						free_path = false;
						CopyVec3ToFloatArray(cur_ray_pos, std::begin(ray.org));
						ray.tnear = 0.0f;
						ray.tfar = INFINITY;
						ray.geomID = RTC_INVALID_GEOMETRY_ID;
						ray.primID = RTC_INVALID_GEOMETRY_ID;
						ray.instID = RTC_INVALID_GEOMETRY_ID;
						ray.mask = -1;
						ray.time = 0;
                
						rtcIntersect(m_Scene, ray);

						if(ray.geomID == RTC_INVALID_GEOMETRY_ID)
						{
							auto env_spectrum = m_GlobalCubeMap ? m_GlobalCubeMap->sampleSpectrum(cur_sample_data.IncidentLight) : m_BackgroundSpectrum;
							global_light += env_spectrum * cur_throughput; // todo: apply balance
							end_ray = true;
							break;
						}

						auto scene = (ray.instID & (1 << 31)) == 0 ? (RTCScene)rtcGetUserData(m_Scene, ray.instID) : m_Scene;

						auto geom_desc = (RTGeometry*)rtcGetUserData(scene, ray.geomID);
						if(!geom_desc)
						{
							end_ray = true;
							break;
						}
						geom_desc->sample(ray, &cur_sample_data);

						material = cur_sample_data.Material = cur_sample_data.Material;
						if(material == nullptr)
						{
							end_ray = true;
							break;
						}

						cur_ray_pos += ray.tfar*cur_sample_data.IncidentLight;
                        cur_sample_data.Position = cur_ray_pos;

						auto cache_function = MaterialCacheLookup[(size_t)material->Model];
						if(cache_function)
							cache_function(cur_sample_data, mirand);

						if(!IsVolumeIlluminationModel(material->Model))
							break;

						RTVolumeSet* volume = static_cast<RTVolumeSet*>(geom_desc);
				
						float threshold;
						do
						{
							cur_ray_pos -= cur_sample_data.IncidentLight * (logf(FastFloatRand(mirand)) * volume->RcpMaxExtinction); 
							threshold = FastFloatRand(mirand);
							if(!volume->sampleScattering(threshold, cur_ray_pos, &cur_sample_data))
							{
								free_path = true;
								break;
							}
							cur_sample_data.TotalDensity *= volume->RcpMaxExtinction;
						} while(cur_sample_data.TotalDensity < threshold);

					} while(free_path);

					if(end_ray)
					{
						break;
					}
					}

					// use to figure out where the samples are coming from
					//color.x = cur_ray_pos.z;
					//break;

					if(IsVolumeIlluminationModel(material->Model))
					{
						// albedo * density / density
                        auto volume_material = static_cast<const RTVolumeMaterial*>(material);
						cur_throughput *= volume_material->Albedo;
						if(cur_throughput < TransmittanceThreshold)
							break;
					}

                    if(SampleIncidentLightLookup[(size_t)material->Model] == nullptr) // It is definitely emissive or wrong
				    {
                        TGE_ASSERT(PDFLookup[(size_t)material->Model] == nullptr, "Something is wrong with this material");
                        auto transmittance_func = TransmittanceLookup[(size_t)material->Model];
                        if(transmittance_func)
                        {
                            global_light += cur_throughput * balance * transmittance_func(cur_sample_data);
                        }
                        else
                        {
                            global_light += cur_throughput * balance * static_cast<const RTMicrofacetMaterial*>(material)->Diffuse;
                        }
                        break;
                    }

					cur_sample_data.OutgoingLight = -cur_sample_data.IncidentLight;
          
					// Here we start again
					surface_samples = 1;
					light_samples = 1;

                    if(!MirrorCheck(cur_sample_data))
                    {
						global_light += EvaluateLightSources(m_Scene, m_LightSources, cur_sample_data, light_samples, surface_samples, cur_ray_pos, cur_throughput, ray_depth + 1 != m_MaxRayDepth, mirand);
                    }

					cur_throughput *= balance / m_RussianRoulette;

					strata.XStrata = strata.YStrata = 0.0f;
					strata.TotalXStrata = strata.TotalYStrata = 1.0f;

					if(m_RussianRoulette < 1.0f)
					{
						float r = FastFloatRand(mirand);
						if(r > m_RussianRoulette)
						{
							break;
						}
					}
				}
			}

        #ifndef NDEBUG
            for(uint32_t spec_sample = 0; spec_sample < SPECTRUM_SAMPLES; ++spec_sample)
            {
                TGE_ASSERT(std::isfinite(Array(local_light)[spec_sample]), "Invalid light contribution");
            }
        #endif

			global_light /= (float)m_SamplesGlobalIllumination;

			final_light += local_light + global_light;
		}
        final_light /= (float)m_SamplesCamera;

    #ifndef NDEBUG
        for(uint32_t spec_sample = 0; spec_sample < SPECTRUM_SAMPLES; ++spec_sample)
        {
            TGE_ASSERT(std::isfinite(Array(final_light)[spec_sample]), "Invalid light contribution");
        }
    #endif

        if(mixing)
		{
			final_light = accum_buffer[p] = accum_buffer[p]*mixing + (1 - mixing)*final_light;
        }
		else if(accum_buffer)
		{
			accum_buffer[p] = final_light;
		}

        if(!intersections && m_TransparentBackground)
            continue;

        auto& out_color = reinterpret_cast<uint32_t*>(frame_data.Backbuffer->getData())[p];
        if(m_PicturePostProcess == PicturePostProcess::SRGB)
			out_color = ToColor(ConvertLinearToSRGB(SpectrumToRGB(final_light)));
        else if(m_PicturePostProcess == PicturePostProcess::Linear)
            out_color = ToColor(SpectrumToRGB(final_light));
        else if(m_PicturePostProcess == PicturePostProcess::ACES)
            out_color = ToColor(ConvertLinearToSRGB(ACESFilm(SpectrumToRGB(final_light))));
        else if(std::isfinite(final_light.x) && std::isfinite(final_light.y) && std::isfinite(final_light.z))
            m_LogAverages[worker_id] += Vector3Log(Epsilon + final_light);
    }

    rand_var = mirand;
}
}

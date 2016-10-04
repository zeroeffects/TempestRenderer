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

#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/image/btf.hh"

#include "svbrdf-fitting.hh"

#ifdef CUDA_ACCELERATED_RENDERING
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#	define CONSTANT __constant__
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#	define CONSTANT
#endif

struct SGGXMicroflakeDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    Tempest::Vector3            LightDirection;
    Tempest::RTSGGXSurface*     Material;
};

struct BTFDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    Tempest::Vector3    LightDirection;
    const Tempest::BTF* BTF;
    uint32_t            X,
                        Y;
    float               Normalization;
};

struct BRDFDebug: public Tempest::RTSpatiallyVaryingEmitter
{
	Tempest::Vector3              LightDirection;
	Tempest::RTMicrofacetMaterial Material;
};

struct NDFDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    const void*                 NDFTexture;
};

struct ResidualDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    Tempest::Vector3            LightDirection;
    const Tempest::BTF*         BTFData;
    Tempest::RTSGGXSurface*     SGGXMaterial;
	const float*	            LuminanceSlice;
};

float SGGXComputeSurfaceLuminanceDebug(const Tempest::RTSGGXSurface* sggx_material, const Tempest::Vector3& light_dir, const Tempest::SampleData& sample_data)
{
    Tempest::SampleData interm_sample_data{};
	interm_sample_data.Material = sggx_material;
	interm_sample_data.IncidentLight = light_dir;
	interm_sample_data.OutgoingLight = sample_data.Normal;
    interm_sample_data.TexCoord = {};
    interm_sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    interm_sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    interm_sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
    interm_sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeSurfaceDensity(interm_sample_data);

    auto spec = sggx_material->Depth != ~0u ?
        Tempest::Cpp::SGGXMicroFlakePseudoVolumeBRDF(interm_sample_data) :
        Tempest::Cpp::SGGXMicroFlakeSurfaceBRDF(interm_sample_data);

    return Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec));
}

Tempest::Spectrum SGGXMicroflakeDebugHeatMap(const Tempest::SampleData& sample_data)
{
    auto sggx_material = static_cast<const SGGXMicroflakeDebug*>(sample_data.Material);
    float lum = SGGXComputeSurfaceLuminanceDebug(sggx_material->Material, sggx_material->LightDirection, sample_data);
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(lum));
}

Tempest::Spectrum NDFDebugHeatMap(const Tempest::SampleData& sample_data)
{
    auto ndf_material = static_cast<const NDFDebug*>(sample_data.Material);

    if(sample_data.Normal.z < 0.0f)
        return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(0.0f));

    auto tc = Tempest::CartesianToParabolicMapCoordinates(sample_data.Normal);

    auto ndf_prob = Tempest::SampleRed(ndf_material->NDFTexture, tc);
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(ndf_prob));
}

Tempest::Spectrum BRDFDebugHeatMap(const Tempest::SampleData& sample_data)
{
    auto brdf_material = static_cast<const BRDFDebug*>(sample_data.Material);

    Tempest::SampleData interm_sample_data{};
	interm_sample_data.Material = &brdf_material->Material;
	interm_sample_data.IncidentLight = brdf_material->LightDirection;
	interm_sample_data.OutgoingLight = sample_data.Normal;
    interm_sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    interm_sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    interm_sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

    auto spec = Tempest::TransmittanceLookup[(size_t)brdf_material->Material.Model](interm_sample_data);

    float lum = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec));
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(lum));
}

Tempest::Spectrum BTFDebugHeatMap(const Tempest::SampleData& sample_data)
{
    auto btf_material = static_cast<const BTFDebug*>(sample_data.Material);
    auto spec = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf_material->BTF, btf_material->LightDirection, sample_data.Normal, btf_material->X, btf_material->Y);
    float lum = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(spec))*btf_material->Normalization;
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(lum));
}

Tempest::Spectrum ResidualDebugHeatMap(const Tempest::SampleData& sample_data)
{
    auto residual_material = static_cast<const ResidualDebug*>(sample_data.Material);
    float lum_slice = Tempest::BTFSampleLuminanceSlice(residual_material->BTFData, residual_material->LightDirection, sample_data.Normal, residual_material->LuminanceSlice);
    float lum_sggx = SGGXComputeSurfaceLuminanceDebug(residual_material->SGGXMaterial, residual_material->LightDirection, sample_data);
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(fabsf(lum_slice - lum_sggx)));
}


void DisplayDistributions(uint32_t image_width, uint32_t image_height,
                          const Tempest::Vector2& sggx_stddev, const Tempest::Vector3& albedo, const Tempest::Vector3& specular, const Tempest::Vector4& sggx_basis,
                          const Tempest::BTF* btf_ptr, const float* lv_lum_slice, uint32_t btf_start_x, uint32_t btf_start_y,
                          const Tempest::Texture& ndf_texture)
{
    Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(70.0f, (float)image_width / image_height, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                        origin{0.0f, 15.0f, 15.0f},
                        up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;

    view.identity();
    view.lookAt(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();

    RAY_TRACING_SYSTEM rt_sys(image_width, image_height, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Spectrum spec_sample = Tempest::RGBToSpectrum(specular);

    NDFDebug ndf_debug;
    ndf_debug.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    ndf_debug.EmitFunction = NDFDebugHeatMap;
    ndf_debug.NDFTexture = rt_scene->bindTexture(&ndf_texture);

    {
    Tempest::Vector3 sphere_pos{ target.x + (SphereRadius + 0.5f*SphereGap), target.y - (SphereRadius + 0.5f*SphereGap), target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &ndf_debug);
    }

	BRDFDebug brdf_debug;
    brdf_debug.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    brdf_debug.EmitFunction = BRDFDebugHeatMap;
    brdf_debug.LightDirection = Tempest::Normalize(Tempest::Vector3{ 0.0f, -1.0f, 1.0f });
    brdf_debug.Material.Specular = spec_sample;
	brdf_debug.Material.Model = Tempest::IlluminationModel::GGXMicrofacet;
	brdf_debug.Material.SpecularPower = { 1.1f, 1.1f };
	brdf_debug.Material.Diffuse = {};
    brdf_debug.setup();

    Tempest::Spectrum albedo_sample = Tempest::RGBToSpectrum(albedo);

    Tempest::RTSGGXSurface surf_material;
    surf_material.Depth = ~0;
    surf_material.Specular = spec_sample;
    surf_material.Diffuse = albedo_sample;
    surf_material.StandardDeviation = sggx_stddev;
    surf_material.SGGXBasis = sggx_basis;
    surf_material.setup();

    ResidualDebug lum_debug;
    lum_debug.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    lum_debug.EmitFunction = ResidualDebugHeatMap;
    lum_debug.LightDirection = Tempest::Normalize(Tempest::Vector3{ 0.0f, -1.0f, 1.0f });
    lum_debug.BTFData = btf_ptr;
    lum_debug.LuminanceSlice = lv_lum_slice;
    lum_debug.SGGXMaterial = &surf_material;
    lum_debug.setup();

    {
    Tempest::Vector3 sphere_pos{ target.x - (SphereRadius + 0.5f*SphereGap), target.y - (SphereRadius + 0.5f*SphereGap), target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &lum_debug);
    }

    BTFDebug btf_debug;
    btf_debug.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    btf_debug.LightDirection = Tempest::Normalize(Tempest::Vector3{ 0.0f, -1.0f, 1.0f });
    btf_debug.BTF = btf_ptr;
    btf_debug.X = btf_start_x;
    btf_debug.Y = btf_start_y;
    btf_debug.EmitFunction = BTFDebugHeatMap;
    btf_debug.Normalization = 1.0f; //1.0f/Tempest::RGBToLuminance(spec_sample);
    btf_debug.setup();

    {
    Tempest::Vector3 sphere_pos{ target.x - (SphereRadius + 0.5f*SphereGap), target.y + SphereRadius + 0.5f*SphereGap, target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &btf_debug);
    }

    SGGXMicroflakeDebug sggx_microflake_debug;
    sggx_microflake_debug.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    sggx_microflake_debug.EmitFunction = SGGXMicroflakeDebugHeatMap;
    sggx_microflake_debug.LightDirection = Tempest::Normalize(Tempest::Vector3{ 0.0f, -1.0f, 1.0f });

    sggx_microflake_debug.Material = &surf_material;
    sggx_microflake_debug.setup();

	{
    Tempest::Vector3 sphere_pos{ target.x + (SphereRadius + 0.5f*SphereGap), target.y + SphereRadius + 0.5f*SphereGap, target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &sggx_microflake_debug);
    }

    rt_scene->commitScene();

    rt_sys.startRendering();

    float angular_freq = 2.0f*Tempest::MathPi/10.0f;

    bool alive = true;
    Tempest::TimeQuery timer;
    do
    {
        auto cur_time = timer.time();

        rt_sys.completeFrame();

        float angle = angular_freq*cur_time*1e-6f, s, c;
        Tempest::FastSinCos(angle, &s, &c);

        Tempest::Vector3 light_dir = Tempest::Vector3{ 0.0f, s, fabsf(c) };

        brdf_debug.LightDirection = light_dir;
        btf_debug.LightDirection = light_dir;
        sggx_microflake_debug.LightDirection = light_dir;
        lum_debug.LightDirection = light_dir;

        rt_scene->repaint();

        rt_sys.completeFrameAndRestart(image_width, image_height, view_proj_inv);

        alive = rt_sys.presentFrame();
    } while(alive);
}
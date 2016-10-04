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
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/utils/viewer.hh"
#include "tempest/image/btf.hh"
#include "tempest/math/point2.hh"

#include "svbrdf-fitting.hh"

//#define CUDA_ACCELERATED 1

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#endif

const float EllipsoidSize = 0.2f;

struct SGGXDebugProbeMaterial: public Tempest::RTSpatiallyVaryingEmitter
{
    Tempest::Spectrum Specular;
};

Tempest::Spectrum SGGXDebugProbes(const Tempest::SampleData& sample_data)
{
    auto* debug_material = reinterpret_cast<const SGGXDebugProbeMaterial*>(sample_data.Material);
	return debug_material->Specular * powf(Dot(sample_data.OutgoingLight, sample_data.Normal), 2.0f);
}

Tempest::Spectrum DebugPlane(const Tempest::SampleData& sample_data)
{
	return Tempest::RGBToSpectrum({ sample_data.TexCoord.x*0.0f, sample_data.TexCoord.y, 0.0f });
}

void DisplayDistributionMap(uint32_t image_width, uint32_t image_height, const Tempest::Point2* points,
                            const Tempest::Vector2* sggx_stddev_map, const Tempest::Vector3* specular_map, const Tempest::Quaternion* sggx_basis_map, uint32_t sample_count,
                            const Tempest::BTF* btf_cpu)
{
    Tempest::FreeCamera cam;
	cam.Yaw = 0.0f;
	cam.Roll = Tempest::ToRadians(45.0f);
    cam.Offset = 10.0f;
    cam.Projection = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);

    Tempest::Matrix4 view_proj_inv = Tempest::ComputeViewProjectionInverse(cam);

    RAY_TRACING_SYSTEM rt_sys(image_width, image_height, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};


#ifdef CUDA_ACCELERATED
	Tempest::BTFGPUPtr gpu_btf(Tempest::CreateGPUBTF(btf_cpu));
	auto btf_ptr = gpu_btf.get();
#else
	auto btf_ptr = btf_cpu;
#endif

#if 1
	Tempest::RTBTF mat;
	mat.Model = Tempest::IlluminationModel::BTF;
	mat.BTFData = btf_ptr;
    mat.setup();
#else
	Tempest::RTSpatiallyVaryingEmitter mat;
	mat.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
	mat.EmitFunction = DebugPlane;
	mat.setup();
#endif

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 1.0f, 0.0f}, rect_size, &mat);

    {
    Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{0.0f, 2.5f, 0.0f };
    area_light1->SphereShape.Radius = 0.1f;
    area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{600.0f, 600.0f, 600.0f});

    rt_scene->addSphereLightSource(area_light1);
    }

    std::unique_ptr<SGGXDebugProbeMaterial[]> debug_materials(new SGGXDebugProbeMaterial[sample_count]);
    for(uint32_t point_idx = 0; point_idx < sample_count; ++point_idx)
    {
        auto& point = points[point_idx];
        auto& debug_material = debug_materials[point_idx];
        debug_material.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
        debug_material.EmitFunction = SGGXDebugProbes;
        auto spec = specular_map[point_idx];
        float max_value_spec = MaxValue(spec);
        debug_material.Specular = Tempest::RGBToSpectrum(max_value_spec ? spec/max_value_spec : Tempest::Vector3{ 1.0f, 0.0f, 1.0f });

        Tempest::Ellipsoid ellipsoid;
        ellipsoid.Center = { (2.0f*point.x/btf_cpu->Width - 1.0f)*rect_size.x, EllipsoidSize, (2.0f*point.y/btf_cpu->Height - 1.0f)*rect_size.y };
        ellipsoid.Orientation = sggx_basis_map[point_idx];

        auto stddev = sggx_stddev_map[point_idx];
        float max_value_stddev = MaxValue(stddev);
        ellipsoid.Scale = (max_value_stddev ? Tempest::Vector3{ stddev.x, stddev.y, 1.0f }/max_value_stddev : Tempest::Vector3{ 1.0f, 1.0f, 1.0f })*EllipsoidSize;

        rt_scene->addEllipsoid(ellipsoid, &debug_material);
    }

	rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(4);
    rt_scene->setSamplesLocalAreaLight(4);
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    rt_scene->commitScene();

    rt_sys.startRendering();

    RayTracingViewer(rt_sys, image_width, image_height, cam);
}
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

#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/image/btf.hh"
#include "tempest/utils/viewer.hh"

#include <cstdlib>
#include <memory>

//#define CUDA_ACCELERATED 1

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;
#endif

int TempestMain(int argc, char** argv)
{
	Tempest::CommandLineOptsParser parser("btf-viewer", true);
    parser.createOption('X', "x-start", "Specify start location on the X axis", true, "0");
    parser.createOption('Y', "y-start", "Specify start location on the Y axis", true, "0");
    parser.createOption('W', "width", "Specify sample width", true);
    parser.createOption('H', "height", "Specify sample height", true);
    parser.createOption('w', "window-size", "Specify window size", true);

	if(!parser.parse(argc, argv))
	{
		return EXIT_FAILURE;
	}

    uint32_t image_width = ImageWidth,
             image_height = ImageHeight;
	
    if(parser.isSet("window-size"))
    {
        auto status = Tempest::ParseResolution(parser.extractString("window-size"), &image_width, &image_height);
        if(!status)
            return EXIT_FAILURE;
    }
    

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-viewer: error: input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-viewer <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-viewer: error: too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-viewer <input-file>");
        return EXIT_FAILURE;
    }

	auto input_filename = parser.getUnassociatedArgument(0);
	Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(input_filename)));

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
	Tempest::BTFGPUPtr gpu_btf(Tempest::CreateGPUBTF(btf.get()));
	auto btf_ptr = gpu_btf.get();
#else
	auto btf_ptr = btf.get();
#endif

#if 1
	Tempest::RTBTF mat;
	mat.Model = Tempest::IlluminationModel::BTF;
	mat.BTFData = btf_ptr;
    mat.setup();
#else
	Tempest::RTMaterial mat;
	mat.Model = Tempest::IlluminationModel::BlinnPhong;
	mat.Diffuse = { 1.0f, 1.0f, 1.0f };
    mat.Specular = { 1.0f, 1.0f, 1.0f };
	mat.SpecularPower = { 1.0f, 1.0f };
	mat.Fresnel = { 1.0f, 0.0f };
	mat.setup();
#endif

    auto x_start = parser.extract<uint32_t>("x-start"),
         y_start = parser.extract<uint32_t>("y-start");

    if(x_start >= btf->Width)
    {
        Tempest::Log(Tempest::LogLevel::Error, "out of bounds X starting value specified: ", x_start);
        return EXIT_FAILURE;
    }

    if(y_start >= btf->Height)
    {
        Tempest::Log(Tempest::LogLevel::Error, "out of bounds Y starting value specified: ", y_start);
        return EXIT_FAILURE;
    }

    auto x_end = btf->Width,
         y_end = btf->Height;

    if(parser.isSet("width"))
    {
        x_end = x_start + parser.extract<uint32_t>("width");
    }

    if(parser.isSet("height"))
    {
        y_end = y_start + parser.extract<uint32_t>("height");
    }

    Tempest::AABB2 tex_coord;
    tex_coord.MinCorner = { (float)x_start/btf->Width, (float)y_start/btf->Height };
    tex_coord.MaxCorner = { (float)x_end/btf->Width, (float)y_end/btf->Height };

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 1.0f, 0.0f}, rect_size, &mat, &tex_coord);

    {
    Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{0.0f, 2.5f, 0.0f};
    area_light1->SphereShape.Radius = 0.1f;
    area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{600.0f, 600.0f, 600.0f});

    rt_scene->addSphereLightSource(area_light1);
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

    return EXIT_SUCCESS;
}
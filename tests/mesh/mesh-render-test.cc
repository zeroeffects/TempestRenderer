#include "tempest/utils/testing.hh"
#include "tempest/texture/texture-table.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/math/matrix4.hh"
#include "tempest/math/vector3.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/compute/compute-convenience.hh"
#include "tempest/graphics/scene-rasterizer-system.hh"

#define GPU_RASTERIZER 1

#ifndef DISABLE_CUDA
#   define CUDA_ACCELERATED 1
#endif

#if GPU_RASTERIZER
#   define RENDER_SYSTEM Tempest::SceneRasterizerSystem
#elif CUDA_ACCELERATED
#   define RENDER_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RENDER_SYSTEM Tempest::RayTracingSystem
#endif

const uint32_t ImageWidth = 640;
const uint32_t ImageHeight = 480;

TGE_TEST("Testing loading meshes")
{
	const uint32_t amp_up_geometry = 8;

	Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(45.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);
    
    Tempest::Vector3 target{0.0f, 50.0f, 0.0f},
                     origin{0.0f, 170.0f, 100.0f},
                     up{0.0f, 1.0f, 0.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAtUpTarget(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

	RENDER_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);

	auto* rt_scene = rt_sys.getRayTracer();

#if 0
	Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{0.0f, 90.0f, 0.0f};
    area_light1->SphereShape.Radius = 1.0f;
    area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{100.0f, 100.0f, 100.0f});

    rt_scene->addSphereLightSource(area_light1);
#elif 1
    Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
    dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 1.0f, 1.0f});
    dir_light->Direction = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    
    rt_scene->addLightSource(dir_light);
#else
    Tempest::PointLight* point_light = new Tempest::PointLight;
    point_light->Position = Tempest::Vector3{0.0f, 90.0f, 0.0f};
    point_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{10000.0f, 10000.0f, 10000.0f});

    rt_scene->addLightSource(point_light);
#endif    

	Tempest::RTMeshBlob mesh_blob;
    auto status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/teapot/teapot.obj", nullptr, &mesh_blob);
    TGE_CHECK(status, "Failed to load test assets");

#if defined(CUDA_ACCELERATED) || defined(GPU_RASTERIZER)
	Tempest::RebindMaterialsToGPU(rt_scene, mesh_blob);
#endif

	rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							  mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front());

	rt_scene->setMaxRayDepth(0);

	rt_scene->commitScene();
    
    Tempest::TimeQuery timer;
    auto start_time = timer.time();

	rt_sys.startRendering();

	rt_sys.completeFrame();

    auto elapsed_time = timer.time() - start_time;
    Tempest::Log(Tempest::LogLevel::Info, "Time to render: ", elapsed_time, "us");

	//rt_sys.displayImage();

    rt_sys.getWindow().show();

    bool active = true;
    do
    {
        rt_sys.completeFrameAndRestart(ImageWidth, ImageHeight, view_proj_inv);
        active = rt_sys.presentFrame();
    } while(active);
}
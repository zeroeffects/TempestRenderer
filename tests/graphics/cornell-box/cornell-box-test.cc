#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/compute/compute-convenience.hh"
#include "tempest/mesh/obj-loader.hh"

//#define CUDA_ACCELERATED 1

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#endif

const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;

TGE_TEST("Rendering the classics")
{
    const uint32_t amp_up_geometry = 8;

	Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(25.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);
    
    Tempest::Vector3 target{0.0f, 1.0f, 0.0f},
                     origin{0.0f, 1.0f, 5.5f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAtUpTarget(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

	RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);

	auto* rt_scene = rt_sys.getRayTracer();

	Tempest::RTMeshBlob mesh_blob;
    auto status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/cornell-box/CornellBox-Original.obj", nullptr, &mesh_blob);
    TGE_CHECK(status, "Failed to load test assets");

#ifdef CUDA_ACCELERATED
	Tempest::RebindMaterialsToGPU(rt_scene, mesh_blob);
#endif

	rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							  mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front());

    //rt_scene->setRenderMode(Tempest::RenderMode::DebugLighting);

    rt_scene->setSamplesCamera(256);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
	rt_scene->setMaxRayDepth(1);
    //rt_scene->setRenderMode(Tempest::RenderMode::DebugNormals);
    rt_scene->setPicturePostProcess(Tempest::PicturePostProcess::AutoExposureHDR);

	rt_scene->commitScene();
    
    Tempest::TimeQuery timer;
    auto start_time = timer.time();

	rt_sys.startRendering();

	rt_sys.completeFrame();

    auto elapsed_time = timer.time() - start_time;
    Tempest::Log(Tempest::LogLevel::Info, "Time to render: ", elapsed_time, "us");

	rt_sys.displayImage();
}
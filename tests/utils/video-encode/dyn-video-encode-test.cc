#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/refractive-indices.hh"
#include "tempest/graphics/software-rasterizer.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/utils/video-encode.hh"
#include "tempest/utils/testing.hh"

const uint32_t FPS = 30;
const uint32_t RecordFrames = 10*FPS;

#if 1
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#   define RASTERIZER_FUNCTION
#   define RASTERIZER Tempest::Rasterizer
const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#   define RASTERIZER_FUNCTION __device__
#   define RASTERIZER Tempest::RasterizerCuda
const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;
#endif

#if 0
#	define VIDEO_FILENAME "dyn-videoenc.m4v"
#	define VIDEO_ENCODER Tempest::VideoEncoder
#else
#	define VIDEO_FILENAME "dyn-videoenc.ivf"
#	define VIDEO_ENCODER Tempest::VPXVideoEncoder
#endif

TGE_TEST("Test the encoding capabilities in ray tracing scenario")
{
    Tempest::Matrix4 proj = Tempest::PerspectiveMatrix(28.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);

    Tempest::Vector3 offset{0.0f, 0.0f, -10.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.translate(offset);
    view.rotateX(-Tempest::MathPi*0.25f);

    auto view_proj = proj * view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);

    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};

    //*
    Tempest::RTMicrofacetMaterial plane_mtl;
    plane_mtl.Model = Tempest::IlluminationModel::GGXMicrofacetDielectric;
    plane_mtl.Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{0.5f, 0.02f, 0.02f});
    plane_mtl.Specular = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 0.25f, 0.25f});
    plane_mtl.SpecularPower.x = 5.0f;
    plane_mtl.Fresnel = Tempest::CelluloseRefractiveIndex;

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 0.0f, 1.0f}, rect_size, &plane_mtl);

    //*
    //if(0)
    {
    Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{0.0f, 0.0f, 2.5f};
    area_light1->SphereShape.Radius = 0.1f;
    area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{100.0f, 100.0f, 100.0f});

    rt_scene->addSphereLightSource(area_light1);
    }

    if(0)
    {
    Tempest::SphereAreaLight* area_light2 = new Tempest::SphereAreaLight;
    area_light2->SphereShape.Center = Tempest::Vector3{2.5f, -2.5f, 10.0f};
    area_light2->SphereShape.Radius = 0.1f;
    area_light2->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{100.0f, 100.0f, 100.0f});

    rt_scene->addSphereLightSource(area_light2);
    }

    if(0)
    {
    Tempest::SphereAreaLight* area_light3 = new Tempest::SphereAreaLight;
    area_light3->SphereShape.Center = Tempest::Vector3{0.0f, 2.5f, 10.0f};
    area_light3->SphereShape.Radius = 0.1f;
    area_light3->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{100.0f, 100.0f, 100.0f});

    rt_scene->addSphereLightSource(area_light3);
    }
    //*/

    //*/
    rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.1f, 0.1f, 0.1f}));

    rt_scene->commitScene();

    rt_sys.startRendering();


    float angular_frequency = 2.0f*Tempest::MathPi/10.0f;

    Tempest::VideoInfo video_info;
    video_info.FileName = VIDEO_FILENAME;
    video_info.FPS = FPS;
    video_info.Width = ImageWidth;
    video_info.Height = ImageHeight;

    VIDEO_ENCODER video_enc;
    auto status = video_enc.openStream(video_info);
    TGE_CHECK(status, "Failed to open video stream for encoding");

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = ImageWidth;
    tex_desc.Height = ImageHeight;
    tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

    auto tex_size = tex_desc.Width*tex_desc.Height*Tempest::DataFormatElementSize(tex_desc.Format);
    std::unique_ptr<uint8_t[]> img_data(new uint8_t[tex_size]);

    float elapsed_time = 1.0f/FPS;
    for(uint32_t rec_frame = 0; rec_frame < RecordFrames; ++rec_frame)
    {
        view.rotateZ(angular_frequency*elapsed_time);

        view_proj = proj * view;
        view_proj_inv = view_proj.inverse();

        rt_sys.completeFrameAndRestart(ImageWidth, ImageHeight, view_proj_inv);

        auto* last_frame_tex = rt_sys.getLastFrameTexture();

        rt_sys.presentFrame();

        auto status = video_enc.submitFrame(*last_frame_tex);
        TGE_CHECK(status, "Failed to submit new frame");
    }
}

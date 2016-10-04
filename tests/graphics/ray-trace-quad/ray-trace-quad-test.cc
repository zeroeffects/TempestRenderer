#include "tempest/utils/testing.hh"
#include "tempest/math/spectrum.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda.hh"
#include "tempest/graphics/sampling-wrapper.hh"

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#endif

const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;

struct DebugMaterial: public Tempest::RTSpatiallyVaryingEmitter
{
    const void* DiffuseMap;
};

Tempest::Spectrum DebugPlane(const Tempest::SampleData& sample_data)
{
    auto material = static_cast<const DebugMaterial*>(sample_data.Material);
	return Tempest::SampleSpectrum(material->DiffuseMap, sample_data.TexCoord);
}

TGE_TEST("Testing how ray tracing simple quad")
{
	Tempest::Matrix4 proj = Tempest::PerspectiveMatrix(28.0f, (float)ImageWidth/ImageHeight, 0.1f, 1000.0f);

	Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, 0.0f, 10.0f},
                     up{0.0f, 1.0f, 0.0f};

	Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    Tempest::Matrix4 view_proj = proj * view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

	RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};

	Tempest::Path tex_path(ROOT_SOURCE_DIR "/tests/graphics/draw-quad-texture/Mandrill.tga");
	std::unique_ptr<Tempest::Texture> tex(Tempest::LoadImage(tex_path));
	TGE_CHECK(tex, "Failed to load texture");

	DebugMaterial mat;
	mat.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
	mat.EmitFunction = DebugPlane;
	mat.DiffuseMap = rt_scene->bindTexture(tex.get());
	mat.setup();

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 0.0f, 1.0f}, rect_size, &mat);

    {
    Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
    area_light1->SphereShape.Center = Tempest::Vector3{-1.0f, -1.0f, 2.5f};
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

	rt_sys.completeFrame();

	rt_sys.displayImage();
}
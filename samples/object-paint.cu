#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/compute/ray-tracing-cuda-system.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/refractive-indices.hh"
#include "tempest/graphics/software-rasterizer.hh"
#include "tempest/compute/software-rasterizer-cuda.hh"
#include "tempest/graphics/sampling-wrapper.hh"

// DEBUG
#include <cuda_runtime_api.h>

const float BrushRadius = 10.0f;

#if 0
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

// Because CUDA can't patch the AST we are forced to resort to old style functors
struct DiskPaintShader
{
    void*    MixTextureData;

    RASTERIZER_FUNCTION void operator()(uint32_t x, uint32_t y, uint32_t width, uint32_t height, const Tempest::Vector2&)
    {
        uint8_t value = 255U;
        Tempest::Surface2DWrite(value, MixTextureData, x*sizeof(uint8_t), y);
    }
};

struct CapsulePaintShader
{
    void*    MixTextureData;

    RASTERIZER_FUNCTION void operator()(uint32_t x, uint32_t y, uint32_t width, uint32_t height, const Tempest::Vector2&, float)
    {
        uint8_t value = 255U; 
        Tempest::Surface2DWrite(value, MixTextureData, x*sizeof(uint8_t), y);
    }
};

int TempestMain(int argc, char** argv)
{
    Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(28.0f, (float)ImageWidth / ImageHeight, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, -5.0f, 5.0f},
					 //origin{0.0f, 0.0f, 10.0f},
                     up{0.0f, 1.0f, 0.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);

    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};

    //*
    auto plane_mtl = Tempest::CreateRTMixMaterial(sizeof(Tempest::RTMicrofacetMaterial), sizeof(Tempest::RTMicrofacetMaterial));
    plane_mtl->Model = Tempest::IlluminationModel::Mix;
    {
    auto* mtl0 = &plane_mtl->getSubMaterial(0);
    mtl0->Model = Tempest::IlluminationModel::GGXMicrofacetDielectric;
    mtl0->Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{0.5f, 0.02f, 0.02f});
    mtl0->Specular = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 0.25f, 0.25f});
    mtl0->SpecularPower.x = 5.0f;
    mtl0->Fresnel = Tempest::CelluloseRefractiveIndex;
    }

    {
    auto* mtl1 = &plane_mtl->getSubMaterial(1);
    mtl1->Model = Tempest::IlluminationModel::GGXMicrofacetDielectric;
    mtl1->Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{0.02f, 0.2f, 0.02f});
    mtl1->Specular = Tempest::RGBToSpectrum(Tempest::Vector3{0.25f, 1.0f, 0.25f});
    mtl1->SpecularPower.x = 1000000.0f;
    mtl1->Fresnel = Tempest::CelluloseRefractiveIndex;
    }

    Tempest::TextureDescription mix_tex_desc;
    mix_tex_desc.Width = 512;
    mix_tex_desc.Height = 512;
    mix_tex_desc.Format = Tempest::DataFormat::R8UNorm;

    size_t tex_size = mix_tex_desc.Width*mix_tex_desc.Height*sizeof(uint8_t);
    uint8_t* mix_texture_data = new uint8_t[tex_size];
    memset(mix_texture_data, 0x0, tex_size);

    Tempest::Texture mix_texture(mix_tex_desc, mix_texture_data);
	
    const void* mix_tex_obj;
    void* mix_surf_obj;
    rt_scene->bindSurfaceAndTexture(&mix_texture, &mix_tex_obj, &mix_surf_obj);

    plane_mtl->MixTexture = mix_tex_obj;
    plane_mtl->setup();
    /*/
    std::unique_ptr<Tempest::Texture> diffuse_map(LoadImage(Tempest::Path(TEST_ASSETS_DIR "/scarf/data/concrete.png")));
   
    auto plane_mtl = Tempest::UniqueMaterial<Tempest::RTMaterial>(new Tempest::RTMaterial);
    //plane_mtl.DiffuseMap = rt_scene->bindTexture(diffuse_map.get());
    plane_mtl->Diffuse = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
    plane_mtl->Specular = Tempest::RGBToSpectrum(Tempest::Vector3{ 0.0f, 0.0f, 0.0f });
    plane_mtl->SpecularPower.x = 1.0f;
    plane_mtl->setup();
    //*/

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 0.0f, 1.0f}, rect_size, plane_mtl.get());

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

    Tempest::TimeQuery timer;

    auto& window = rt_sys.getWindow();
    window.setEventMask(Tempest::COLLECT_MOUSE_EVENTS|Tempest::COLLECT_WINDOW_EVENTS);

    struct SpringInfo
    {
        Tempest::Vector2 Stretch;
    };

    std::unique_ptr<SpringInfo[]> spring_grid(new SpringInfo[mix_tex_desc.Width*mix_tex_desc.Height]);

	Tempest::SampleData sample_data;
    Tempest::WindowSystemEvent wevent;
    Tempest::Vector2 mouse_pos{ (float)window.getMouseX(), (float)window.getMouseY() },
					 image_size{ (float)mix_tex_desc.Width, (float)mix_tex_desc.Height },
					 prev_pos;
    bool alive = true, active = true, paint = false;

    CapsulePaintShader capsule_paint_shader{ mix_surf_obj };
    DiskPaintShader disk_paint_shader{ mix_surf_obj };

    do
    {
        rt_sys.completeFrameAndRestart(ImageWidth, ImageHeight, view_proj_inv);

		Tempest::Vector2 window_size{ float(window.getWidth() - 1), float(window.getHeight() - 1) };

        while(window.getEvent(&wevent))
        {
            switch(wevent.Type)
            {
            case Tempest::WindowEventType::MouseButtonPressed:
            {
                if(wevent.MouseButton != Tempest::MouseButtonId::LeftButton || !active)
                    break;
				auto geom_id = rt_scene->rayQuery(mouse_pos/window_size, &sample_data);
                if(geom_id != plane_id)
					break;
				prev_pos = sample_data.TexCoord*image_size;
				RASTERIZER::RasterizeDisk2(Tempest::Disk2{ prev_pos, 0.0f, BrushRadius },  mix_tex_desc.Width, mix_tex_desc.Height,
                                           disk_paint_shader);
				window.captureMouse();
				rt_scene->repaint();
                paint = true;
            } break;
            case Tempest::WindowEventType::MouseButtonReleased:
            {
                if(wevent.MouseButton != Tempest::MouseButtonId::LeftButton)
                    break;
				window.releaseMouse();
                paint = false;
            } break;
            case Tempest::WindowEventType::MouseMoved:
            {
                mouse_pos = Tempest::Vector2{ (float)wevent.MouseMoved.MouseX, (float)wevent.MouseMoved.MouseY };
                if(!paint || !active)
                    break;
				auto geom_id = rt_scene->rayQuery(mouse_pos/window_size, &sample_data);
				if(geom_id != plane_id)
					break;
				auto cur_pos = sample_data.TexCoord*image_size;
                RASTERIZER::RasterizeCapsule2(Tempest::Capsule2{ { prev_pos, cur_pos }, BrushRadius }, mix_tex_desc.Width, mix_tex_desc.Height,
                                              capsule_paint_shader);
                                           
				prev_pos = cur_pos;
                rt_scene->repaint();
            } break;
            case Tempest::WindowEventType::Focus:
            {
                active = wevent.Enabled != 0;
                if(!active)
                    paint = false;
            } break;
            }
        }
        
        alive = rt_sys.presentFrame();
    } while(alive);

    return EXIT_SUCCESS;
}
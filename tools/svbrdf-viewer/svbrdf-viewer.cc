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
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/utils/viewer.hh"
#include "tempest/utils/threads.hh"
#include "tempest/graphics/custom-samplers.hh"
#include "tempest/utils/video-encode.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/equirectangular-map.hh"
#include "tempest/graphics/cube-map.hh"

#include "svbrdf-viewer.hh"

#include <cstdlib>
#include <memory>

const uint32_t FPS = 30;

//#define LINEAR_SAMPLING

int TempestMain(int argc, char** argv)
{
	Tempest::CommandLineOptsParser parser("svbrdf-viewer", true);
    parser.createOption('d', "directional-light", "Use directional light source", false);
    parser.createOption('E', "environment-map", "Specify environment map for lighting", true);
    parser.createOption('t', "display-tangents", "Display tangents", false);
    parser.createOption('b', "display-binormals", "Display binormals", false);
    parser.createOption('n', "display-normals", "Display normals", false);
    parser.createOption('w', "window-size", "Specify window size", true);
    parser.createOption('T', "texcoord-multiplier", "Specify texture coordinates multiplier", true, "1.0");
    parser.createOption('A', "animate", "Specify repeat period in ms of capture combination of SGGX textures", true);
    parser.createOption('D', "discrete", "Discrete animation", false);
    parser.createOption('R', "video-record", "Specify output video file", true);
    parser.createOption('r', "record-time", "Specify time in seconds to record on video", true, "5");
    parser.createOption('I', "render-image", "Specify output image file", true);

	if(!parser.parse(argc, argv))
	{
		return EXIT_FAILURE;
	}

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::CrashMessageBox("Error",
                                 "svbrdf-viewer: error: input SGGX file is not specified\n\n"
                                 "USAGE:\n"
                                 "\tsggx-viewer <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::CrashMessageBox("Error",
                                 "svbrdf-viewer: error: too many input files specified\n\n"
                                 "USAGE:\n"
                                 "\tsggx-viewer <input-file>");
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
    
    auto input_filename = parser.getUnassociatedArgument(0);

    std::unique_ptr<SGGXMapCPU[]>        sggx_map;
    
    SGGXMapCPU                           mix_buffer;

    std::unique_ptr<SGGXMapBoundConst[]> sggx_map_bound;

    SGGXMapBound                         mix_buffer_bound;


    uint32_t mix_diffuse_width, mix_diffuse_height,
             mix_specular_width, mix_specular_height,
             mix_basis_width, mix_basis_height,
             mix_scale_width, mix_scale_height,
             basis_width, basis_height;

    Tempest::DataFormat diffuse_fmt, specular_fmt, basis_fmt, scale_fmt;

    const void* diffuse_map, *specular_map, *basis_map, *scale_map;

    uint32_t frame_count;

    Tempest::FreeCamera cam;
	cam.Yaw = 0.0f;
	cam.Roll = Tempest::ToRadians(45.0f);
    cam.Offset = 10.0f;
    cam.Projection = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);

    Tempest::Matrix4 view_proj_inv = Tempest::ComputeViewProjectionInverse(cam);

    std::unique_ptr<Tempest::RTMaterial> mat;

    RAY_TRACING_SYSTEM rt_sys(image_width, image_height, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    auto id = rt_scene->getThreadId();
    auto& pool = rt_scene->getThreadPool();

    int64_t animation_period = 0;
    if(parser.isSet("animate"))
    {
        const char* direction_suffixes[] =
        {
            "_top_center",
            "_top_right",
            "_center_right",
            "_bottom_right",
            "_bottom_center",
            "_bottom_left",
            "_center_left",
            "_top_left"
        };

        frame_count = TGE_FIXED_ARRAY_SIZE(direction_suffixes);

        animation_period = parser.extract<uint64_t>("animate")*1000;
        sggx_map = decltype(sggx_map)(new SGGXMapCPU[frame_count]);
        sggx_map_bound = decltype(sggx_map_bound)(new SGGXMapBoundConst[frame_count]);

        for(uint32_t idx = 0; idx < frame_count; ++idx)
        {
            const char* dir_suffix = direction_suffixes[idx];

            auto& sggx_slice = sggx_map[idx];
            auto& mix_sggx_slice = sggx_map_bound[idx];

            Tempest::Path diffuse_path(input_filename + dir_suffix + "_albedo.exr");
            auto& diffuse_map_ptr = sggx_slice.DiffuseMap = decltype(sggx_map[0].DiffuseMap)(Tempest::LoadImage(diffuse_path));
            if(!diffuse_map_ptr)
	        {
		        Tempest::CrashMessageBox("Error", "Failed to load albedo texture: ", diffuse_path);
		        return EXIT_FAILURE;
	        }
            auto& diffuse_hdr0 = sggx_map[0].DiffuseMap->getHeader();
            diffuse_map_ptr->convertToRGBA();
            mix_sggx_slice.DiffuseMap = rt_scene->bindBuffer(diffuse_map_ptr->getData(), diffuse_hdr0.Width*diffuse_hdr0.Height*Tempest::DataFormatElementSize(diffuse_hdr0.Format));

             Tempest::Path specular_path(input_filename + dir_suffix + "_specular.exr");
            auto& specular_map_ptr = sggx_slice.SpecularMap = decltype(sggx_map[0].SpecularMap)(Tempest::LoadImage(specular_path));
	        if(!sggx_map[0].SpecularMap)
	        {
		        Tempest::CrashMessageBox("Error", "Failed to load specular texture: ", specular_path);
		        return EXIT_FAILURE;
	        }
            auto& specular_hdr0 = sggx_map[0].SpecularMap->getHeader();
            specular_map_ptr->convertToRGBA();
            mix_sggx_slice.SpecularMap = rt_scene->bindBuffer(specular_map_ptr->getData(), specular_hdr0.Width*specular_hdr0.Height*Tempest::DataFormatElementSize(specular_hdr0.Format));

            Tempest::Path basis_path(input_filename + dir_suffix + "_sggx_basis.exr");
            auto& basis_map_ptr = sggx_slice.BasisMap = decltype(sggx_map[0].BasisMap)(Tempest::LoadImage(basis_path));
	        if(!basis_map_ptr)
	        {
		        Tempest::CrashMessageBox("Error", "Failed to load basis map: ", basis_path);
		        return EXIT_FAILURE;
	        }
            auto& basis_hdr0 = sggx_map[0].BasisMap->getHeader();
            mix_sggx_slice.BasisMap = rt_scene->bindBuffer(basis_map_ptr->getData(), basis_hdr0.Width*basis_hdr0.Height*Tempest::DataFormatElementSize(basis_hdr0.Format));

            Tempest::Path scale_path(input_filename + dir_suffix + "_sggx_scale.exr");
            auto& scale_map_ptr = sggx_slice.ScaleMap = decltype(sggx_map[0].ScaleMap)(Tempest::LoadImage(scale_path));
	        if(!sggx_map[0].ScaleMap)
	        {
		        Tempest::CrashMessageBox("Error", "Failed to load standard deviation map: ", scale_path);
		        return EXIT_FAILURE;
	        }
            auto& scale_hdr0 = sggx_map[0].ScaleMap->getHeader();
            mix_sggx_slice.ScaleMap = rt_scene->bindBuffer(scale_map_ptr->getData(), scale_hdr0.Width*scale_hdr0.Height*Tempest::DataFormatElementSize(scale_hdr0.Format));

            if(Tempest::DataFormatChannels(scale_map_ptr->getHeader().Format) != 2)
            {
                Tempest::CrashMessageBox("Error", "Invalid channel count in standard deviation map: ", scale_path);
                return EXIT_FAILURE;
            }

            if(idx)
            {
                auto& cur_diffuse_hdr = sggx_map[idx].DiffuseMap->getHeader();
                auto& cur_specular_hdr = sggx_map[idx].SpecularMap->getHeader();
                auto& cur_basis_hdr = sggx_map[idx].BasisMap->getHeader();
                auto& cur_scale_hdr = sggx_map[idx].ScaleMap->getHeader();

                if(diffuse_hdr0.Width != cur_diffuse_hdr.Width ||
                   diffuse_hdr0.Height != cur_diffuse_hdr.Height ||
                   specular_hdr0.Width != cur_specular_hdr.Width ||
                   specular_hdr0.Height != cur_specular_hdr.Height ||
                   basis_hdr0.Width != cur_basis_hdr.Width ||
                   basis_hdr0.Height != cur_basis_hdr.Height ||
                   scale_hdr0.Width != cur_scale_hdr.Width ||
                   scale_hdr0.Height != cur_scale_hdr.Height)
                {
                    Tempest::CrashMessageBox("Error", "Inconsistent texture dimensions");
                    return EXIT_FAILURE;
                }

                basis_width = basis_hdr0.Width;
                basis_height = basis_hdr0.Height;
            }
            else
            {
                mix_diffuse_width = diffuse_hdr0.Width;
                mix_diffuse_height = diffuse_hdr0.Height;
                mix_specular_width = specular_hdr0.Width;
                mix_specular_height = specular_hdr0.Height;
                basis_width = mix_basis_width = basis_hdr0.Width;
                basis_height = mix_basis_height = basis_hdr0.Height;
                mix_scale_width = scale_hdr0.Width;
                mix_scale_height = scale_hdr0.Height;
                diffuse_fmt = diffuse_hdr0.Format;
                specular_fmt = specular_hdr0.Format;
                basis_fmt = basis_hdr0.Format;
                scale_fmt = scale_hdr0.Format;
            }
        }

        auto& diffuse_hdr0 = sggx_map[0].DiffuseMap->getHeader();

        Tempest::TextureDescription diffuse_tex_desc;
        diffuse_tex_desc.Width = diffuse_hdr0.Width;
        diffuse_tex_desc.Height = diffuse_hdr0.Height;
        diffuse_tex_desc.Format = Tempest::DataFormat::RGBA32F;

        Tempest::Vector4* diffuse_map_data = new Tempest::Vector4[diffuse_tex_desc.Width*diffuse_tex_desc.Height];

        auto& mix_diffuse_map = mix_buffer.DiffuseMap = decltype(mix_buffer.DiffuseMap)(new Tempest::Texture(diffuse_tex_desc, reinterpret_cast<uint8_t*>(diffuse_map_data)));

        rt_scene->bindSurfaceAndTexture(mix_diffuse_map.get(), &diffuse_map, &mix_buffer_bound.DiffuseMap);

        auto& specular_hdr0 = sggx_map[0].SpecularMap->getHeader();

        Tempest::TextureDescription specular_tex_desc;
        specular_tex_desc.Width = specular_hdr0.Width;
        specular_tex_desc.Height = specular_hdr0.Height;
        specular_tex_desc.Format = Tempest::DataFormat::RGBA32F;

        Tempest::Vector4* specular_map_data = new Tempest::Vector4[specular_tex_desc.Width*specular_tex_desc.Height];

        auto& mix_specular_map = mix_buffer.SpecularMap = decltype(mix_buffer.SpecularMap)(new Tempest::Texture(specular_tex_desc, reinterpret_cast<uint8_t*>(specular_map_data)));

        rt_scene->bindSurfaceAndTexture(mix_specular_map.get(), &specular_map, &mix_buffer_bound.SpecularMap);

        auto& basis_hdr0 = sggx_map[0].BasisMap->getHeader();

        Tempest::TextureDescription basis_tex_desc;
        basis_tex_desc.Width = basis_hdr0.Width;
        basis_tex_desc.Height = basis_hdr0.Height;
        basis_tex_desc.Format = Tempest::DataFormat::RGBA32F;

        Tempest::Vector4* basis_map_data = new Tempest::Vector4[basis_tex_desc.Width*basis_tex_desc.Height];

        auto& mix_basis_map = mix_buffer.BasisMap = decltype(mix_buffer.BasisMap)(new Tempest::Texture(basis_tex_desc, reinterpret_cast<uint8_t*>(basis_map_data)));

        rt_scene->bindSurfaceAndTexture(mix_basis_map.get(), &basis_map, &mix_buffer_bound.BasisMap);

        auto& scale_hdr0 = sggx_map[0].ScaleMap->getHeader();

        Tempest::TextureDescription scale_tex_desc;
        scale_tex_desc.Width = scale_hdr0.Width;
        scale_tex_desc.Height = scale_hdr0.Height;
        scale_tex_desc.Format = Tempest::DataFormat::RG32F;

        Tempest::Vector2* scale_map_data = new Tempest::Vector2[scale_tex_desc.Width*scale_tex_desc.Height];

        auto& mix_scale_map = mix_buffer.ScaleMap = decltype(mix_buffer.ScaleMap)(new Tempest::Texture(scale_tex_desc, reinterpret_cast<uint8_t*>(scale_map_data)));

        rt_scene->bindSurfaceAndTexture(mix_scale_map.get(), &scale_map, &mix_buffer_bound.ScaleMap);

        #ifdef CPU_DEBUG
        mix_buffer_bound.DiffuseMap = mix_diffuse_map.get();
        mix_buffer_bound.SpecularMap = mix_specular_map.get();
        mix_buffer_bound.BasisMap = mix_basis_map.get();
        mix_buffer_bound.ScaleMap = mix_scale_map.get();
        #endif

        CopyTextures(id, pool,
                     sggx_map_bound.get(), 0, mix_buffer_bound,
                     diffuse_fmt, mix_diffuse_width, mix_diffuse_height,
                     specular_fmt, mix_specular_width, mix_specular_height,
                     basis_fmt, mix_basis_width, mix_basis_height,
                     scale_fmt, mix_scale_width, mix_scale_height);
    }
    else
    {
        frame_count = 1;

        sggx_map = decltype(sggx_map)(new SGGXMapCPU[frame_count]);

        Tempest::Path diffuse_path(input_filename + "_albedo.exr");
        auto& diffuse_map_ptr = sggx_map[0].DiffuseMap = decltype(sggx_map[0].DiffuseMap)(Tempest::LoadImage(diffuse_path));
        if(!diffuse_map_ptr)
	    {
		    Tempest::CrashMessageBox("Error", "Failed to load albedo texture: ", diffuse_path);
		    return EXIT_FAILURE;
	    }

        diffuse_map_ptr->convertToRGBA();

        diffuse_map = rt_scene->bindTexture(diffuse_map_ptr.get());

        Tempest::Path specular_path(input_filename + "_specular.exr");
        auto& specular_map_ptr = sggx_map[0].SpecularMap = decltype(sggx_map[0].SpecularMap)(Tempest::LoadImage(specular_path));
	    if(!sggx_map[0].SpecularMap)
	    {
		    Tempest::CrashMessageBox("Error", "Failed to load specular texture: ", specular_path);
		    return EXIT_FAILURE;
	    }

        specular_map_ptr->convertToRGBA();

        specular_map = rt_scene->bindTexture(specular_map_ptr.get());

        Tempest::Path basis_path(input_filename + "_sggx_basis.exr");
        auto& basis_map_ptr = sggx_map[0].BasisMap = decltype(sggx_map[0].BasisMap)(Tempest::LoadImage(basis_path));
	    if(!basis_map_ptr)
	    {
		    Tempest::CrashMessageBox("Error", "Failed to load basis map: basis_path");
		    return EXIT_FAILURE;
	    }

        basis_map = rt_scene->bindTexture(basis_map_ptr.get());

        Tempest::Path scale_path(input_filename + "_sggx_scale.exr");
        auto& scale_map_ptr = sggx_map[0].ScaleMap = decltype(sggx_map[0].ScaleMap)(Tempest::LoadImage(scale_path));
	    if(!sggx_map[0].ScaleMap)
	    {
		    Tempest::CrashMessageBox("Error", "Failed to load standard deviation map: ", scale_path);
		    return EXIT_FAILURE;
	    }

        scale_map = rt_scene->bindTexture(scale_map_ptr.get());

        auto& basis_hdr = basis_map_ptr->getHeader();

        basis_width = basis_hdr.Width;
        basis_height = basis_hdr.Height;
    }
    
    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};

    bool display_normals = parser.isSet("display-normals");
    bool display_tangents = parser.isSet("display-tangents");
    bool display_binormals = parser.isSet("display-binormals");

    uint32_t debug_modes_enabled = (uint32_t)display_tangents + (uint32_t)display_binormals + (uint32_t)display_normals;

    if(debug_modes_enabled > 1)
    {
        Tempest::CrashMessageBox("Error", "More than one debug mode is unsupported");
        return EXIT_FAILURE;
    }

    std::unique_ptr<Tempest::CubeMap> env_map;

    uint32_t depth = 0;

    if(debug_modes_enabled)
    {
        DebugMode debug_mode = DebugMode::DebugNormal;
        if(display_tangents)
            debug_mode = DebugMode::DebugTangent;
        else if(display_binormals)
            debug_mode = DebugMode::DebugBinormal;
        else if(!display_normals)
            TGE_ASSERT(false, "Unsupported mode");

        DebugNormalsMaterial* debug_normals = CreateDebugMaterial(debug_mode);
        mat = decltype(mat)(debug_normals);

        debug_normals->Width = basis_width;
        debug_normals->Height = basis_height;
        debug_normals->DiffuseMap = basis_map;
        debug_normals->setup();
    }
    else
    {
        auto sggx_mat = new Tempest::RTSGGXSurface;
	    mat = decltype(mat)(sggx_mat);
	    sggx_mat->Model = Tempest::IlluminationModel::SGGXSurface;
	    sggx_mat->SGGXBasis = { 0.0f, 0.0f, 0.0f };
	    sggx_mat->StandardDeviation = { 1.0f, 1.0f };
	    sggx_mat->Diffuse = Tempest::ToSpectrum(1.0f);
	    sggx_mat->Specular = Tempest::ToSpectrum(1.0f);
        sggx_mat->DiffuseMap = diffuse_map;
        sggx_mat->SpecularMap = specular_map;
        sggx_mat->BasisMapWidth = basis_width;
        sggx_mat->BasisMapHeight = basis_height;
	    sggx_mat->BasisMap = basis_map;
	    sggx_mat->StandardDeviationMap = scale_map;
        sggx_mat->setup();

        if(parser.isSet("environment-map"))
        {
            Tempest::Path env_map_path(parser.extractString("environment-map"));
            Tempest::TexturePtr env_map_tex(Tempest::LoadImage(env_map_path));
            if(!env_map_tex)
            {
                Tempest::CrashMessageBox("Error", "Failed to open environment map: ", env_map_path);
                return EXIT_FAILURE;
            }

            Tempest::EquirectangularMap eqrect_map(env_map_tex.get());
            
            auto& hdr = env_map_tex->getHeader();

            auto cube_map_edge = Tempest::NextPowerOf2(hdr.Width / 4);

            Tempest::TextureDescription cube_map_tex_desc;
            cube_map_tex_desc.Width = cube_map_tex_desc.Height = cube_map_edge;
            switch(hdr.Format)
            {
            case Tempest::DataFormat::RGBA8UNorm: cube_map_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm; break;
            case Tempest::DataFormat::RGB32F:
            case Tempest::DataFormat::RGBA32F: cube_map_tex_desc.Format = Tempest::DataFormat::RGBA32F; break;
            default:
                Tempest::CrashMessageBox("Error", "Unsupported environment map format: ", env_map_path);
                return EXIT_FAILURE;
            }

            env_map = decltype(env_map)(Tempest::ConvertEquirectangularMapToCubeMap(cube_map_tex_desc, eqrect_map));
            rt_scene->setGlobalCubeMap(env_map.get());
            depth = 1;
        }
        else if(parser.isSet("directional-light"))
        {
            Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	        dir_light->Direction = Tempest::Vector3{0.0f, 1.0f, 0.0f};
	        dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 1.0f, 1.0f});
	        rt_scene->addLightSource(dir_light);
        }
        else
        {
            Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
            area_light1->SphereShape.Center = Tempest::Vector3{0.0f, 2.5f, 0.0f};
            area_light1->SphereShape.Radius = 0.1f;
            area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{600.f, 600.0f, 600.0f});

            rt_scene->addSphereLightSource(area_light1);
        }
    }

    float texcoord = parser.extract<float>("texcoord-multiplier");

    Tempest::AABB2 tc{ { 0.0f, 0.0f }, { texcoord, texcoord } };

    auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 1.0f, 0.0f}, rect_size, mat.get(), &tc);
    
	rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(16384);
    rt_scene->setSamplesLocalAreaLight(4);
    rt_scene->setMaxRayDepth(depth);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    rt_scene->commitScene();

    rt_sys.startRendering();

    if(parser.isSet("render-image"))
    {
        rt_sys.completeFrame();
        auto tex = rt_sys.getLastFrameTexture();

        Tempest::Path tex_file_path(parser.extractString("render-image"));
        auto status = Tempest::SaveImage(tex->getHeader(), tex->getData(), tex_file_path);
        if(!status)
        {
            Tempest::CrashMessageBox("Error", "Failed to save image file");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    Tempest::RayTracingView<RAY_TRACING_SYSTEM> viewer(rt_sys, image_width, image_height, cam);

    Tempest::WindowSystemEvent wevent;

    bool alive = true;

    auto& window = rt_sys.getWindow();

    bool discrete = parser.isSet("discrete");

    Tempest::TimeQuery timer;

    uint32_t prev_frame = frame_count;

    auto video_file = parser.extractString("video-record");
    auto video_seconds = parser.extract<int64_t>("record-time");
    int64_t cur_time = 0LL;

    Tempest::VPXVideoEncoder video_encoder;
    bool record_video = !video_file.empty();
    if(record_video)
    {
        Tempest::VideoInfo video_info;
        video_info.FileName = video_file;
        video_info.Bitrate = 50000;
        video_info.FPS = FPS;
        video_info.Width = image_width;
        video_info.Height = image_height;
        auto status = video_encoder.openStream(video_info);
        if(!status)
        {
            Tempest::CrashMessageBox("Error", "Failed to open output video file: ", video_file);
            return EXIT_FAILURE;
        }

        Tempest::System::OpenConsoleConnection();
        Tempest::Log(Tempest::LogLevel::Info, "Started video recording");
    }
    
    do
    {
        rt_sys.completeFrame();

        for(;;)
        {
            auto status = window.getEvent(&wevent);
			if(!status)
                break;

            viewer.handleEvent(wevent);
        }

        if(!record_video)
            cur_time = timer.time();

        if(animation_period)
        {
            float t = (float)(Tempest::Modulo(cur_time, animation_period))/animation_period;
        
            float t_frame = t*frame_count;

            uint32_t index0 = Tempest::FastFloorToInt(t_frame) % frame_count;

            //Tempest::Log(Tempest::LogLevel::Debug, index0, ", ", index1, ", ", t_mix);

            if(!discrete)
            {
                uint32_t index1 = (index0 + 1) % frame_count;
                float t_mix = t_frame - (float)index0;

                MixTextures(id, pool,
                            sggx_map_bound.get(), index0, index1, mix_buffer_bound,
                            diffuse_fmt, mix_diffuse_width, mix_diffuse_height,
                            specular_fmt, mix_specular_width, mix_specular_height,
                            basis_fmt, mix_basis_width, mix_basis_height,
                            scale_fmt, mix_scale_width, mix_scale_height,
                            t_mix);

                rt_scene->repaint();
            }
            else if(prev_frame != index0)
            {
                CopyTextures(id, pool,
                             sggx_map_bound.get(), index0, mix_buffer_bound,
                             diffuse_fmt, mix_diffuse_width, mix_diffuse_height,
                             specular_fmt, mix_specular_width, mix_specular_height,
                             basis_fmt, mix_basis_width, mix_basis_height,
                             scale_fmt, mix_scale_width, mix_scale_height);

                rt_scene->repaint();
            }
        }

        if(record_video)
        {
            Tempest::Log(Tempest::LogLevel::Info, "Completed frame: ", cur_time*1e-6f, "s");

            video_encoder.submitFrame(*rt_sys.getLastFrameTexture());
        }

        alive = viewer.render();

        if(record_video)
        {
            cur_time += 1000000LL/FPS;

            if(cur_time > video_seconds*1000000LL)
            {
                break;
            }
        }


    } while(alive);

    return EXIT_SUCCESS;
}
/*
*   The MIT License
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

#include "tempest/utils/logging.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/image/image.hh"
#include "tempest/image/btf.hh"

#include <cstdlib>

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
#endif

const float RepeatFactor = 16.0f;

void ModifyMesh(void* v0, void* v1, void* v2, uint32_t stride, const Tempest::Vector3& barycentric, Tempest::SampleData* data)
{
    Tempest::DefaultMeshSample(v0, v1, v2, stride, barycentric, data);
    if(data->Material->Model == Tempest::IlluminationModel::BlinnPhong)
    {
        return;
    }

    data->TexCoord *= RepeatFactor;
}

int TempestMain(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("fabrics-tester", true);
    parser.createOption('X', "x-start", "Specify starting texture coordinate on X-axis", true, "0");
	parser.createOption('Y', "y-start", "Specify starting texture coordinate on Y-axis", true, "0");
	parser.createOption('W', "width", "Specify width of the sample image", true);
	parser.createOption('H', "height", "Specify height of the sample image", true);
    parser.createOption('d', "display", "Specify display resolution", true);
    parser.createOption('o', "output", "Specify output file", true, "output.png");

    auto status = parser.parse(argc, argv);
    if(!status)
        return EXIT_FAILURE;

    if(parser.getUnassociatedCount() != 1)
    {
        Tempest::CrashMessageBox("Error", "You must specify exactly one input material");
        return EXIT_FAILURE;
    }

    uint32_t image_width = 400;
    uint32_t image_height = 400;

    if(parser.isSet("display"))
    {
        auto res = Tempest::ParseResolution(parser.extractString("display"), &image_width, &image_height);
        if(!res)
        {
            return EXIT_FAILURE;
        }
    }

    auto projection = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);

    Tempest::Matrix4 view;

    Tempest::Vector3 base{0.0f, 0.0f, 0.0f};
    Tempest::Vector3 offset{0.0f, 0.0f, 5.0f};
    float yaw = Tempest::ToRadians(50.0f);
	float roll = Tempest::ToRadians(22.5f);

	view.identity();
    view.translate(-offset);
	view.rotateX(Tempest::MathPi*0.5f - roll);
    view.rotateY(-yaw);

    auto view_proj = projection*view;

    auto view_proj_inv = view_proj.inverse();

    Tempest::TextureDescription mix_tex_desc;
    mix_tex_desc.Width = 1024;
    mix_tex_desc.Height = 1024;
    mix_tex_desc.Format = Tempest::DataFormat::R8UNorm;
    size_t mix_tex_area = mix_tex_desc.Width*mix_tex_desc.Height;

    std::unique_ptr<Tempest::RTMaterial> raii_material;

    Tempest::TexturePtr textures[4];
    Tempest::BTFPtr btf;

    RAY_TRACING_SYSTEM rt_sys(image_width, image_height, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    auto x_start = parser.extract<uint32_t>("x-start"),
         y_start = parser.extract<uint32_t>("y-start");

    uint32_t idx = 0;

    Tempest::Path file_path(parser.getUnassociatedArgument(0));
    if(file_path.extension() == "btf")
    {
        btf = Tempest::BTFPtr(Tempest::LoadBTF(file_path));

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

        if(x_end > btf->Width || y_end > btf->Height)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Specified texture area is out of bounds: (", x_start, ", ", y_start, ") -- (", x_end, ", ", y_end, ")");
            return EXIT_FAILURE;
        }

        auto area_width = x_end - x_start,
             area_height = y_end - y_start;
        if(area_width != btf->Width && area_width ||
           area_height != btf->Height && area_height)
        {
            btf = Tempest::BTFPtr(Tempest::CutBTF(btf.get(), x_start, y_start, area_width, area_height));
        }

        auto material = new Tempest::RTBTF{};
        raii_material = decltype(raii_material)(material);
        material->Model = Tempest::IlluminationModel::BTF;
        material->BTFData = btf.get();
        material->setup();
    }
    else
    {
        auto material = new Tempest::RTSGGXSurface{};
        raii_material = decltype(raii_material)(material);
        material->Model = Tempest::IlluminationModel::SGGXSurface;
        material->Depth = 0;
        material->SampleCount = 256;
        material->BasisMapWidth = mix_tex_desc.Width;
        material->BasisMapHeight = mix_tex_desc.Height;

        Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

        material->StandardDeviation = { 1.0f, 1.0f };

        Tempest::TextureDescription stddev_tex_desc;
        stddev_tex_desc.Width = mix_tex_desc.Width;
        stddev_tex_desc.Height = mix_tex_desc.Height;
        stddev_tex_desc.Format = Tempest::DataFormat::RG32F;
    
        textures[idx] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(file_path.get() + "_albedo.exr")));
        TGE_ASSERT(textures[idx], "Failed to load diffuse map");
        material->DiffuseMap = rt_scene->bindTexture(textures[idx].get());
        ++idx;

        textures[idx] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(file_path.get() + "_sggx_basis.exr")));
        TGE_ASSERT(textures[idx], "Failed to load basis map");
        material->BasisMap = rt_scene->bindTexture(textures[idx].get());
        ++idx;

        textures[idx] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(file_path.get() + "_sggx_scale.exr")));
        TGE_ASSERT(textures[idx], "Failed to load stddev map");
        material->StandardDeviationMap = rt_scene->bindTexture(textures[idx].get());
        ++idx;

        textures[idx] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(file_path.get() + "_specular.exr")));
        TGE_ASSERT(textures[idx], "Failed to load specular map");
        material->SpecularMap = rt_scene->bindTexture(textures[idx].get());
        ++idx;

        uint32_t width = 0, height = 0;
        if(parser.isSet("width"))
        {
            width = parser.extract<uint32_t>("width");
        }

        if(parser.isSet("height"))
        {
            height = parser.extract<uint32_t>("height");
        }

        for(auto& tex : textures)
        {
            auto& hdr = tex->getHeader();

            if(x_start >= hdr.Width)
            {
                Tempest::Log(Tempest::LogLevel::Error, "out of bounds X starting value specified: ", x_start);
                return EXIT_FAILURE;
            }

            if(y_start >= hdr.Height)
            {
                Tempest::Log(Tempest::LogLevel::Error, "out of bounds Y starting value specified: ", y_start);
                return EXIT_FAILURE;
            }

            auto x_end = hdr.Width,
                 y_end = hdr.Height;

	        if(parser.isSet("width"))
            {
                x_end = x_start + width;
            }

            if(parser.isSet("height"))
            {
                y_end = y_start + height;
            }

            if(x_end > hdr.Width || y_end > hdr.Height)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Specified texture area is out of bounds: (", x_start, ", ", y_start, ") -- (", x_end, ", ", y_end, ")");
                return EXIT_FAILURE;
            }

            width = x_end - x_start;
            height = y_end - y_start;

            if(hdr.Width != width && width != 0 ||
               hdr.Height != height && height != 0)
            {
                auto src_data = tex->getData();

                Tempest::TextureDescription tex_desc;
                tex_desc.Width = x_end - x_start;
                tex_desc.Height = y_end - y_start;
                tex_desc.Format = hdr.Format;

                auto data_size = Tempest::DataFormatElementSize(hdr.Format);
                uint8_t* data = new uint8_t[tex_desc.Width*tex_desc.Height];
                auto dst_pitch = tex_desc.Width*data_size;
                for(uint16_t y = 0; y < tex_desc.Height; ++y)
                {
                    memcpy(data + y*dst_pitch, src_data + ((y_start + y)*hdr.Width + x_start)*data_size, dst_pitch);
                }

                tex->realloc(tex_desc, data);
            }
        }

        if(width == 1 && height == 1)
        {
            material->Specular = Tempest::FetchRGB(material->SpecularMap, 0, 0);
            material->Diffuse = Tempest::FetchRGB(material->DiffuseMap, 0, 0);
            material->SGGXBasis = Tempest::FetchRGBA(material->BasisMap, 0, 0);
            material->StandardDeviation = Tempest::FetchRG(material->StandardDeviationMap, 0, 0);

            material->DiffuseMap = nullptr;
            material->BasisMap = nullptr;
            material->StandardDeviationMap = nullptr;
            material->SpecularMap = nullptr;
        }
        else
        {
            material->Specular = Tempest::ToSpectrum(1.0f);
            material->Diffuse = Tempest::ToSpectrum(1.0f);
        }

	    material->setup();
    }

	Tempest::RTMeshBlob mesh_blob;

    status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/cloth/clothsuperhd.obj", nullptr, &mesh_blob, Tempest::TEMPEST_OBJ_LOADER_GENERATE_TANGENTS|Tempest::TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS);
    TGE_ASSERT(status, "Failed to load test assets");
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Missing cloth mesh");
        return false;
    }

#if defined(CUDA_ACCELERATED) || defined(GPU_RASTERIZER)
	Tempest::RebindMaterialsToGPU(rt_scene, mesh_blob);
#endif

	auto submesh_ids = TGE_TYPED_ALLOCA(uint64_t, mesh_blob.SubmeshCount);

	const uint32_t draw_surf_idx = 2;

	mesh_blob.Submeshes[draw_surf_idx].Material = raii_material.get();

    Tempest::MeshOptions mesh_opts;
    mesh_opts.TwoSided = true;
    mesh_opts.GeometrySampler = ModifyMesh;

	rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							  mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front(), &mesh_opts, submesh_ids);

    if(0) //flags & MATERIAL_MANIPULATION_POINT_LIGHT)
    {
        Tempest::PointLight* point_light = new Tempest::PointLight;
        point_light->Position = Tempest::Vector3{1.5f, 4.0f, 1.5f};
        point_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1200.0f, 1200.0f, 1200.0f});

        rt_scene->addLightSource(point_light);
    }
    else
    {
        Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
        area_light1->SphereShape.Center = Tempest::Vector3{1.5f, 4.0f, 1.5f};
        area_light1->SphereShape.Radius = 0.1f;
        area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{4000.0f, 4000.0f, 4000.0f});

        rt_scene->addSphereLightSource(area_light1);
    }

    rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(4);

    rt_scene->setSamplesLocalAreaLight(64);

    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(1.0f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    rt_scene->commitScene();

    rt_sys.startRendering();
    
    rt_sys.completeFrame();

    auto filename = parser.extractString("output");
    status = rt_sys.saveImage(Tempest::Path(filename));
    if(!status)
    {
        Tempest::CrashMessageBox("Error", "Failed to save file: ", filename);
    }

    rt_sys.displayImage();

    return EXIT_SUCCESS;
}
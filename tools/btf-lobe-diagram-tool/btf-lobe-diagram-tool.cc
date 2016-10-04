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

#include "tempest/image/btf.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/image/eps-draw.hh"
#include "tempest/math/hdr.hh"

#include <cstdlib>

Tempest::Spectrum HeatMap(const Tempest::SampleData& sample_data)
{
	return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(sample_data.DirectionalDensity))*Tempest::Dot(sample_data.Normal, sample_data.OutgoingLight);
}

void GeometrySampler(void* v0, void* v1, void* v2, uint32_t stride, const Tempest::Vector3& barycentric, Tempest::SampleData* data)
{
	size_t normal_offset = stride - 3*sizeof(float);

	auto pos0 = reinterpret_cast<Tempest::Vector3*>(v0);
	auto pos1 = reinterpret_cast<Tempest::Vector3*>(v1);
	auto pos2 = reinterpret_cast<Tempest::Vector3*>(v2);

	auto norm0 = reinterpret_cast<Tempest::Vector3*>((char*)v0 + normal_offset);
    auto norm1 = reinterpret_cast<Tempest::Vector3*>((char*)v1 + normal_offset);
    auto norm2 = reinterpret_cast<Tempest::Vector3*>((char*)v2 + normal_offset);

	Tempest::Vector3 norm = Tempest::Normalize(*norm0*barycentric.z + *norm1*barycentric.x + *norm2*barycentric.y);

	Tempest::Vector3 pos = *pos0*barycentric.z + *pos1*barycentric.x + *pos2*barycentric.y;

	data->Normal = norm;
	data->DirectionalDensity = Tempest::Length(pos);
}

int main(int argc, char* argv[])
{
    Tempest::CommandLineOptsParser parser("btf-lobe-diagram-tool", true);
    parser.createOption('l', "light-index", "Specify light index used to create light-view slice", true);
    parser.createOption('v', "view-index", "Specify view index used for sampling a pixel", true);
    parser.createOption('L', "light-direction", "Specify light direction used to create interpolated light-view slice (example: 0:0:1)", true);
    parser.createOption('V', "view-direction", "Specify view direction used for sampling a pixel (example: 0:0:1)", true);
    parser.createOption('o', "output", "Specify output file (default: \"btf-lobe.eps\")", true, "btf-lobe.eps");
	parser.createOption('X', "x-coordinate", "Specify texture coordinate on X-axis", true, "0");
	parser.createOption('Y', "y-coordinate", "Specify texture coordinate on Y-axis", true, "0");
    parser.createOption('s', "scale", "Specify scale factor", true, "1.0");

    if(!parser.parse(argc, argv))
	{
		return EXIT_FAILURE;
	}

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-lobe-diagram-tool: error: input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-lobe-diagram-tool [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "btf-lobe-diagram-tool: error: too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tbtf-lobe-diagram-tool [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }

    auto input_filename = parser.getUnassociatedArgument(0);
	Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(input_filename)));
    if(!btf)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to load btf file: ", input_filename);
        return EXIT_FAILURE;
    }

    uint32_t btf_width = btf->Width,
             btf_height = btf->Height,
             btf_light_count = btf->LightCount,
             btf_view_count = btf->LightCount;

    bool light_dir_set = parser.isSet("light-direction"),
         view_dir_set = parser.isSet("view-direction"),
         light_idx_set = parser.isSet("light-index"),
         view_idx_set = parser.isSet("view-index");

    uint32_t view_idx = 0;
    if(view_idx_set)
    {
        view_idx = parser.extract<uint32_t>("view-index");
    }
    else if(view_dir_set)
    {
        if(!view_dir_set)
        {
            Tempest::Log(Tempest::LogLevel::Error, "you must specify both light(-L) and view(-V) direction");
            return EXIT_FAILURE;
        }

        if(view_idx_set)
        {
            Tempest::Log(Tempest::LogLevel::Error, "mixed light direction and indexing mode is unsupported");
            return EXIT_FAILURE;
        }

        auto //light_dir_str = parser.extractString("light-direction"),
             view_dir_str = parser.extractString("view-direction");

        Tempest::Vector3 //light_dir,
                         view_dir;

        /*
        auto status = ParseDirection(light_dir_str.c_str(), &light_dir);
        if(!status)
        {
            return EXIT_FAILURE;
        }
        Tempest::NormalizeSelf(&light_dir);
        */

        auto status = ParseDirection(view_dir_str.c_str(), &view_dir);
        if(!status)
        {
            return EXIT_FAILURE;
        }
        Tempest::NormalizeSelf(&view_dir);

        auto view_idx_set = Tempest::BTFNearestAngle(btf.get(), view_dir);
    }

	auto x_coord = parser.extract<uint32_t>("x-coordinate"),
         y_coord = parser.extract<uint32_t>("y-coordinate");

    float scale = parser.extract<float>("scale");

    auto vert_count = btf_view_count + btf->EdgeCount;
    std::unique_ptr<Tempest::ShapeVertex[]> verts(new Tempest::ShapeVertex[vert_count]);
    float span = 1.0f;
    for(uint32_t idx = 0; idx < btf_view_count; ++idx)
    {
        auto& vert = verts[idx];
        auto len = Tempest::RGBToLuminance(Tempest::SpectrumToRGB(Tempest::BTFFetchSpectrum(btf.get(), idx, view_idx, x_coord, y_coord)))*scale;
        //span = Maxf(len, span);
        vert.Position = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[idx])*len;
        vert.Normal = {};
    }

    for(uint32_t idx = 0; idx < btf->EdgeCount; ++idx)
    {
        auto& vert = verts[btf_view_count + idx];
        auto data_index = btf->Edges[idx].Index0;
        auto dir = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[data_index]);
        dir.z = 0.0f;
        Tempest::NormalizeSelf(&dir);
        vert.Position = dir*Tempest::RGBToLuminance(Tempest::SpectrumToRGB(Tempest::BTFFetchSpectrum(btf.get(), idx, view_idx, x_coord, y_coord)));
        vert.Normal = {};
    }

    auto base_index_count = btf->LightTriangleCount*3;

    auto index_count = base_index_count + 6*btf->EdgeCount;
    std::unique_ptr<uint32_t[]> indices(new uint32_t[index_count]);
    std::copy_n(btf->LightIndices, base_index_count, indices.get());

    auto idx = base_index_count;
    for(uint32_t edge_idx = 0; edge_idx < btf->EdgeCount; ++edge_idx)
    {
        auto i0 = btf->Edges[edge_idx].Index0;
        auto i1 = btf->Edges[edge_idx].Index1;
        auto i2 = btf_view_count + edge_idx;
        auto i3 = btf_view_count + (edge_idx + 1) % btf->EdgeCount;

        indices[idx++] = i0;
        indices[idx++] = i3;
        indices[idx++] = i1;

        indices[idx++] = i0;
        indices[idx++] = i2;
        indices[idx++] = i3;
    }
    TGE_ASSERT(idx == index_count, "Invalid index count");

    for(size_t idx = 0; idx < index_count;)
    {
        auto prev_idx = indices[idx++];
        auto current_idx = indices[idx++];
        auto next_idx = indices[idx++];
        
        auto& prev = verts[prev_idx];
        auto& current = verts[current_idx];
        auto& next = verts[next_idx];
        auto d0 = prev.Position - current.Position;
        auto d1 = next.Position - current.Position;
        Tempest::Vector3 norm = Cross(d1, d0);
        prev.Normal += norm;
        current.Normal += norm;
        next.Normal += norm;
    }
    
    for(size_t idx = 0; idx < vert_count; ++idx)
    {
        NormalizeSelf(&verts[idx].Normal);
    }

    float scale_image = 100;
    int32_t image_width = 400;
	int32_t image_height = 400;
    uint32_t tesselate = 128;

    auto max_dim = std::max(image_width, image_height);

    //Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);
    Tempest::Matrix4 proj = Tempest::OrthoMatrix(-(float)1.5f*image_width/max_dim, (float)1.5f*image_width/max_dim, -(float)1.5f*image_height/max_dim, (float)1.5f*image_height/max_dim, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{2.0f, 2.0f, 2.0f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view, view_proj;
    
    view.identity();
    view.lookAt(origin, target, up);

    view_proj = proj*view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    std::unique_ptr<Tempest::RayTracerScene> rt_scene(new Tempest::RayTracerScene(image_width, image_height, view_proj_inv));

    Tempest::Sphere sphere{ { 0.0f, 0.0f, 0.0f }, 1.0f };

    /*
    Tempest::SphereAreaLight* area_light = new Tempest::SphereAreaLight;
    area_light->SphereShape.Center = Tempest::Vector3{0.0f, 0.0f, 0.0f};
    area_light->SphereShape.Radius = 0.1f;
    area_light->Radiance = Tempest::Vector3{5000.0f, 5000.0f, 5000.0f};

    rt_scene->addLightSource(area_light);
	*/

	Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	dir_light->Direction = Tempest::Normalize(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });
    NormalizeSelf(&dir_light->Direction);
	dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{1.0f, 1.0f, 1.0f});
	rt_scene->addLightSource(dir_light);

	rt_scene->setSamplesCamera(64);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);
    rt_scene->setBackgroundSpectrum(Tempest::ToSpectrum(1.0f));

    Tempest::RTSpatiallyVaryingEmitter material;
	material.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
	material.EmitFunction = HeatMap;
    
    Tempest::RTSubmesh submesh;
    submesh.BaseIndex = 0;
    submesh.Material = &material;
    submesh.Stride = sizeof(Tempest::ShapeVertex);
    submesh.VertexCount = index_count;
    submesh.VertexOffset = 0;

    Tempest::MeshOptions mesh_opts;
	mesh_opts.GeometrySampler = GeometrySampler;
    
    Tempest::RTMicrofacetMaterial base_material;
    base_material.Model = Tempest::IlluminationModel::Emissive;
    base_material.Diffuse = Tempest::ToSpectrum(0.75f);
    base_material.Specular = {};
    base_material.setup();

    rt_scene->addRect({}, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { span, span }, &base_material);

    rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), 1, &submesh, index_count/3, reinterpret_cast<int32_t*>(indices.get()), vert_count*sizeof(Tempest::ShapeVertex), verts.get(), &mesh_opts);
    
    //rt_scene->setRenderMode(Tempest::RenderMode::DebugNormals);

	rt_scene->commitScene();

    rt_scene->initWorkers();

    auto output_file = parser.extractString("output");

    uint32_t border = 10;
    uint32_t plot_size = image_height/2;

    Tempest::EPSImageInfo info;
    info.Width = image_width + 4*plot_size + 2*border;
    info.Height = image_height + btf_height + 2*border;

    Tempest::EPSDraw eps_image(info);

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = btf_width;
    tex_desc.Height = btf_height;
    tex_desc.Format = Tempest::DataFormat::RGB32F;
    Tempest::Vector3* tex_data(new Tempest::Vector3[tex_desc.Width*tex_desc.Height]);
    Tempest::Texture tex(tex_desc, reinterpret_cast<uint8_t*>(tex_data));
    
    Tempest::Spectrum cur_value;
    
    Tempest::Vector3 view_dir;

    if(light_dir_set || view_dir_set)
    {
        
        Tempest::Vector3 light_dir = { 0.0f, 0.0f, 1.0f };

        if(light_dir_set)
        {
            auto light_dir_str = parser.extractString("light-direction");

            auto status = ParseDirection(light_dir_str.c_str(), &light_dir);
            if(!status)
            {
                return EXIT_FAILURE;
            }
            Tempest::NormalizeSelf(&light_dir);
        }

        if(light_idx_set || view_idx_set)
        {
            Tempest::Log(Tempest::LogLevel::Error, "mixed light direction and indexing mode is unsupported");
            return EXIT_FAILURE;
        }

        auto view_dir_str = parser.extractString("view-direction");
        
        auto status = ParseDirection(view_dir_str.c_str(), &view_dir);
        if(!status)
        {
            return EXIT_FAILURE;
        }
        Tempest::NormalizeSelf(&view_dir);

        uint32_t light_prim_id, view_prim_id;
        Tempest::Vector3 light_barycentric, view_barycentric;
        status = Tempest::BTFFetchLightViewDirection(btf.get(), light_dir, view_dir, &light_prim_id, &light_barycentric, &view_prim_id, &view_barycentric);
        if(!status)
        {
            Tempest::Log(Tempest::LogLevel::Error, "failed to intersect BTF geometry. TODO!!! Implement grazing angle BTF");
            return EXIT_FAILURE;
        }

        for(uint32_t btf_y = 0; btf_y < btf_height; ++btf_y)
        {
            for(uint32_t btf_x = 0; btf_x < btf_width; ++btf_x)
            {
                auto spec = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf.get(), light_prim_id, light_barycentric, view_prim_id, view_barycentric, btf_x, btf_y);
                tex_data[btf_y*btf_width + btf_x] = Tempest::SpectrumToRGB(spec);
            }
        }

        cur_value = Tempest::BTFFetchPixelSampleLightViewSpectrum(btf.get(), light_prim_id, light_barycentric, view_prim_id, view_barycentric, x_coord, y_coord);
    }
    else
    {
        uint32_t light_idx = 0, view_idx = 0;
		if(light_idx_set)
			light_idx = parser.extract<uint32_t>("light-index");
		if(view_idx_set)
            view_idx = parser.extract<uint32_t>("view-index");

        view_dir = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);

        if(light_idx >= btf_light_count ||
           view_idx >= btf_view_count)
        {
            Tempest::Log(Tempest::LogLevel::Error, "out of bounds light or view index: ", light_idx, ", ", view_idx);
            return EXIT_FAILURE;
        }

        for(uint32_t btf_y = 0; btf_y < btf_height; ++btf_y)
        {
            for(uint32_t btf_x = 0; btf_x < btf_width; ++btf_x)
            {
                auto spec = Tempest::BTFFetchSpectrum(btf.get(), light_idx, view_idx, btf_x, btf_y);
                tex_data[btf_y*btf_width + btf_x] = Tempest::SpectrumToRGB(spec);
            }
        }

        cur_value = Tempest::BTFFetchSpectrum(btf.get(), light_idx, view_idx, x_coord, y_coord);
    }

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();
    float exp_factor;

    std::unique_ptr<Tempest::Texture> out_tex(Tempest::ParallelConvertHDRToSRGB(id, pool, &tex, 64, 0.18f, &exp_factor));

    eps_image.drawImage(*out_tex, (float)border, float(border + image_height));

    {
    Tempest::Vector2 path[] = { { border + x_coord - 3.0f, border + image_height + y_coord - 3.0f },
                                { border + x_coord + 3.0f, border + image_height + y_coord - 3.0f },
                                { border + x_coord + 3.0f, border + image_height + y_coord + 3.0f },
                                { border + x_coord - 3.0f, border + image_height + y_coord + 3.0f } };

    eps_image.drawPath(path, TGE_FIXED_ARRAY_SIZE(path), true, 5);
    }

    float rect_edge = 100.0f;

    {
    Tempest::Rect2 rect;
    rect.Center = { 2*border + btf_width + rect_edge*0.5f, border + image_height + btf_height - rect_edge*0.5f};
    rect.Size = { rect_edge*0.5f, rect_edge*0.5f };
    rect.Orientation = 0.0f;

    //                     start1{ border + x_coord + 5.5f, border + image_height + y_coord + 6.0f };

    eps_image.drawLine(border + x_coord + 2.5f, border + image_height + y_coord + 3.0f, 
                       float(2*border + btf_width), float(border + image_height + btf_height), 1.0f);

    eps_image.drawLine(border + x_coord + 2.5f, border + image_height + y_coord - 3.0f,
                       float(2*border + btf_width), float(border + image_height + btf_height - rect_edge), 1.0f);

    eps_image.drawRect(rect, Tempest::ToColor(out_tex->fetchRGB(x_coord, y_coord)), 5.0f);
    }

    {
    std::stringstream ss;
    ss << "X: " << x_coord;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 20, float(2*border + btf_width), float(border + image_height + btf_height - rect_edge - 20));
    }

    {
    std::stringstream ss;
    ss << "Y: " << y_coord;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 20, float(2*border + btf_width), float(border + image_height + btf_height - rect_edge - 40));
    }

    //auto v0 = view_proj*Tempest::Vector3{-span, -span, 0.0f};
    //v0 = v0*0.5f + 0.5f;
    auto v1 = view_proj*Tempest::Vector3{span, -span, 0.0f};
    v1 = v1*0.5f + 0.5f;
    auto v2 = view_proj*Tempest::Vector3{-span, span, 0.0f};
    v2 = v2*0.5f + 0.5f;    
    auto v3 = view_proj*Tempest::Vector3{span, span, 0.0f};
    v3 = v3*0.5f + 0.5f;
    auto v4 = view_proj*Tempest::Vector3{span, -span, span};
    v4 = v4*0.5f + 0.5f;

    proj = Tempest::OrthoMatrix(-(float)1.5f, (float)1.5f, -(float)1.5f*0.5f, (float)1.5f*0.5f, 0.1f, 1000.0f);

    Tempest::Vector3 cam_view_vec = Tempest::Normalize(origin - target);

    target = {0.0f, 0.0f, 0.75f};
    origin = {0.0f, 2.0f, 0.75f};
 
    view.identity();
    view.lookAt(origin, target, up);

    view_proj = proj*view;
    view_proj_inv = view_proj.inverse();

    auto* frame_data = rt_scene->draw(2*plot_size, plot_size, view_proj_inv);
    TGE_ASSERT(frame_data->Backbuffer, "Invalid backbuffer");
    auto* backbuffer = frame_data->Backbuffer.get();

    eps_image.drawImage(*backbuffer, (float)border, (float)border);

    {
    std::stringstream ss;
    ss << "Camera View: " << cam_view_vec;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 16, (float)border + 20, (float)(border + image_height - 30));
    }

    cam_view_vec = Tempest::Normalize(origin - target);

    origin = {2.0f, 0.0f, 0.75f};
 
    view.identity();
    view.lookAt(origin, target, up);

    view_proj = proj*view;
    view_proj_inv = view_proj.inverse();

    frame_data = rt_scene->draw(2*plot_size, plot_size, view_proj_inv);
    TGE_ASSERT(frame_data->Backbuffer, "Invalid backbuffer");
    backbuffer = frame_data->Backbuffer.get();

    eps_image.drawImage(*backbuffer, (float)(image_width + border), (float)border );

    {
    std::stringstream ss;
    ss << "Camera View: " << cam_view_vec;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 16, (float)(image_width + border + 20), (float)(border + plot_size - 30));
    }

    proj = Tempest::OrthoMatrix(-(float)1.5f, (float)1.5f, -(float)1.5f, (float)1.5f, 0.1f, 1000.0f);

    cam_view_vec = Tempest::Normalize(origin - target);

    target = {0.0f, 0.0f, 0.0f};
    origin = {0.0f, 0.0f, 2.0f};
    up = { 0.0f, 1.0f, 0.0f };

    view.identity();
    view.lookAt(origin, target, up);

    view_proj = proj*view;
    view_proj_inv = view_proj.inverse();

    base_material.Diffuse = Tempest::ToSpectrum(1.0f);

    frame_data = rt_scene->draw(2*plot_size, 2*plot_size, view_proj_inv);
    TGE_ASSERT(frame_data->Backbuffer, "Invalid backbuffer");
    backbuffer = frame_data->Backbuffer.get();

    eps_image.drawImage(*backbuffer, (float)(image_width + border), (float)(border + plot_size) );

    {
    std::stringstream ss;
    ss << "Camera View: " << cam_view_vec;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 16, (float)(image_width + border + 20), (float)(border + 2*plot_size - 30));
    }

    cam_view_vec = Tempest::Normalize(origin - target);

    frame_data = rt_scene->draw(2*plot_size, 2*plot_size, view_proj_inv);
    TGE_ASSERT(frame_data->Backbuffer, "Invalid backbuffer");
    backbuffer = frame_data->Backbuffer.get();

    eps_image.drawImage(*backbuffer, (float)(image_width + 2*plot_size + border), (float)border );

    {
    std::stringstream ss;
    ss << "Camera View: " << cam_view_vec;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 16, (float)(image_width + 2*plot_size + border + 20), (float)(border + 2*plot_size - 30));
    }

    //eps_image.drawLine(v0.x*hdr.Width, v0.y*hdr.Height, v1.x*hdr.Width, v1.y*hdr.Height, 1);
    //eps_image.drawLine(v0.x*hdr.Width, v0.y*hdr.Height, v2.x*hdr.Width, v2.y*hdr.Height, 1);
    eps_image.drawLine(v3.x*image_width + border, v3.y*image_height + border, v1.x*image_width + border, v1.y*image_height + border, 1);
    eps_image.drawLine(v3.x*image_width + border, v3.y*image_height + border, v2.x*image_width + border, v2.y*image_height + border, 1);
    eps_image.drawLine(v1.x*image_width + border, v1.y*image_height + border, v4.x*image_width + border, v4.y*image_height + border, 1);

    eps_image.drawText("-x", 20, v2.x*image_width - 10 + border, v2.y*image_height - 20 + border);
    eps_image.drawText("-y", 20, v1.x*image_width - 10 + border, v1.y*image_height - 20 + border);
    eps_image.drawText("z", 20, v4.x*image_width - 5 + border, v4.y*image_height + 5 + border);

    {
    std::stringstream ss;
    ss << "V: " << view_dir;
    auto str = ss.str();
    eps_image.drawText(str.c_str(), 16, (float)(border), (float)border);
    eps_image.drawText("L: var", 16, (float)(border), (float)border + 20.0f);
    }

    const uint32_t cut_count = 7;
    const uint32_t arc_count = 3;

    for(uint32_t plot_idx = 0; plot_idx < 2; ++plot_idx)
    {
        Tempest::Vector2 center{ float(image_width + plot_size + border), float(border + plot_idx*plot_size) };

        for(uint32_t arc_idx = 0; arc_idx < arc_count; ++arc_idx)
        {
            eps_image.drawArc(center.x, center.y, (float)plot_size*0.7f*(arc_idx + 1)/arc_count, 0, 180, 1.0f, Tempest::rgba(32, 32, 32, 255));
        }

        for(uint32_t cut_idx = 0; cut_idx < cut_count; ++cut_idx)
        {
            Tempest::Vector2 dir;
            float angle = Tempest::MathPi*cut_idx/(cut_count - 1);
            Tempest::FastSinCos(angle, &dir.y, &dir.x);

            dir *= (float)plot_size;

            eps_image.drawLine(center.x, center.y, center.x + 0.8f*dir.x, center.y + 0.8f*dir.y, 2);

            std::stringstream ss;
            ss << floorf(Tempest::ToDegress(angle) + 0.5f);
            auto str = ss.str();
            eps_image.drawText(str.c_str(), 16, center.x + 0.875f*dir.x - 10, center.y + 0.875f*dir.y);
        }
    }
 
    Tempest::Vector2 center{ float(image_width + 3*plot_size + border), float(border + plot_size) };
    for(uint32_t arc_idx = 0; arc_idx < arc_count; ++arc_idx)
    {
        eps_image.drawCircle(center.x, center.y, (float)plot_size*0.7f*(arc_idx + 1)/arc_count, 1.0f, Tempest::rgba(32, 32, 32, 255));
    }

    const uint32_t circle_cut_count = 12;

    for(uint32_t cut_idx = 0; cut_idx < circle_cut_count; ++cut_idx)
    {
        Tempest::Vector2 dir;
        float angle = Tempest::MathTau*cut_idx/circle_cut_count;
        Tempest::FastSinCos(angle, &dir.y, &dir.x);

        dir *= (float)plot_size;

        eps_image.drawLine(center.x, center.y, center.x + 0.8f*dir.x, center.y + 0.8f*dir.y, 2);

        std::stringstream ss;
        ss << floorf(Tempest::ToDegress(angle) + 0.5f);
        auto str = ss.str();
        eps_image.drawText(str.c_str(), 16, center.x + 0.875f*dir.x - 10, center.y + 0.875f*dir.y - 10);
    }
    
    auto status = eps_image.saveImage(output_file.c_str());
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Failed to save to output file: ", output_file);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
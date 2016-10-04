#include "tempest/utils/logging.hh"
#include "tempest/graphics/ray-tracing/ray-tracing-system.hh"
#include "tempest/utils/parse-command-line.hh"
#include "tempest/mesh/obj-loader.hh"
#include "tempest/utils/viewer.hh"

#include <cstdlib>

#if CUDA_ACCELERATED
#   define RAY_TRACING_SYSTEM Tempest::RayTracingCudaSystem
const uint32_t ImageWidth = 800;
const uint32_t ImageHeight = 800;
#else
#   define RAY_TRACING_SYSTEM Tempest::RayTracingSystem
const uint32_t ImageWidth = 400;
const uint32_t ImageHeight = 400;
#endif

Tempest::Spectrum DebugNormals(const Tempest::SampleData& sample_data)
{
	auto color = sample_data.DirectionalDensity < 0.0f ? Tempest::RGBToSpectrum({ 1.0f, 0.0f, 0.0f}) : Tempest::ToSpectrum(1.0f);

    return color*Tempest::Dot(sample_data.OutgoingLight, sample_data.Normal);
}

void DebugGeometrySampler(void* v0, void* v1, void* v2, uint32_t stride, const Tempest::Vector3& barycentric, Tempest::SampleData* data)
{
	size_t normal_offset = stride - 3*sizeof(float);

	auto pos0 = reinterpret_cast<Tempest::Vector3*>(v0);
	auto pos1 = reinterpret_cast<Tempest::Vector3*>(v1);
	auto pos2 = reinterpret_cast<Tempest::Vector3*>(v2);

	auto norm0 = reinterpret_cast<Tempest::Vector3*>((char*)v0 + normal_offset);
    auto norm1 = reinterpret_cast<Tempest::Vector3*>((char*)v1 + normal_offset);
    auto norm2 = reinterpret_cast<Tempest::Vector3*>((char*)v2 + normal_offset);

	Tempest::Vector3 norm = Tempest::Normalize(*norm0*barycentric.z + *norm1*barycentric.x + *norm2*barycentric.y);

	Tempest::Vector3 edge0, edge1;
	edge0 = *pos0 - *pos1;
	edge1 = *pos2 - *pos1;

	auto geom_norm = Tempest::Normalize(Tempest::Cross(edge1, edge0));

	data->Normal = norm;
	data->DirectionalDensity = Tempest::Dot(geom_norm, norm);
}

int TempestMain(int argc, char** argv)
{
	Tempest::CommandLineOptsParser parser("mesh-debug-tool", true);

	if(!parser.parse(argc, argv))
	{
		return EXIT_FAILURE;
	}

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "mesh-debug-tool: error: input mesh file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tmesh-debug-tool <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "mesh-debug-tool: error: too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tmesh-debug-tool <input-file>");
        return EXIT_FAILURE;
    }

	auto input_filename = parser.getUnassociatedArgument(0);

	Tempest::Path input_path(input_filename);

	Tempest::SubdirectoryFileLoader subdir_loader(input_path.directoryPath());

	Tempest::RTMeshBlob mesh_blob;
	uint32_t flags = Tempest::TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS;
	auto status = Tempest::LoadObjFileStaticRTGeometry(input_filename, &subdir_loader, &mesh_blob, flags);
    TGE_ASSERT(status, "Failed to load test assets");

	if(!status)
	{
		Tempest::Log(Tempest::LogLevel::Error, "failed to load the specified mesh: ", input_filename);
		return EXIT_FAILURE;
	}

    Tempest::FreeCamera cam;
	cam.Yaw = 0.0f;
	cam.Roll = Tempest::ToRadians(45.0f);
    cam.Offset = 30.0f;
    cam.Projection = Tempest::PerspectiveMatrix(40.0f, (float)ImageWidth/ImageHeight, 0.1f, 1000.0f);

    Tempest::Matrix4 view_proj_inv = Tempest::ComputeViewProjectionInverse(cam);

    RAY_TRACING_SYSTEM rt_sys(ImageWidth, ImageHeight, view_proj_inv);
    auto* rt_scene = rt_sys.getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    Tempest::Vector2 rect_size{2.0f, 2.0f};

    std::unique_ptr<Tempest::RTMaterial> mat;

    auto debug_normals = new Tempest::RTSpatiallyVaryingEmitter;
    mat = decltype(mat)(debug_normals);
    debug_normals->Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    debug_normals->EmitFunction = DebugNormals;
    debug_normals->setup();

	for(uint32_t submesh_idx = 0; submesh_idx < mesh_blob.SubmeshCount; ++submesh_idx)
		mesh_blob.Submeshes[submesh_idx].Material = mat.get();

	Tempest::MeshOptions mesh_opts;
	mesh_opts.GeometrySampler = DebugGeometrySampler;

    rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							  mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front(), &mesh_opts);
    
	rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(4);
    rt_scene->setSamplesLocalAreaLight(4);
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    rt_scene->commitScene();

    rt_sys.startRendering();

    RayTracingViewer(rt_sys, ImageWidth, ImageHeight, cam);

    return EXIT_SUCCESS;
}
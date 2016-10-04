/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 2016 Zdravko Velinov
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

#ifndef NDEBUG
#   define LOW_SPEC
#endif

#define ENABLE_PROFILER
#include "tempest/utils/display-image.hh"
#include "tempest/math/sampling1.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/refractive-indices.hh"
#include "tempest/graphics/software-rasterizer.hh"
#include "tempest/compute/software-rasterizer-cuda.hh"
#include "tempest/math/ode.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"
#include "tempest/graphics/sampling-wrapper.hh"
#include "tempest/compute/compute-convenience.hh"
#include "tempest/utils/profiler.hh"

#include "material-manipulation.hh"

#include <cuda_runtime_api.h>

#undef CurrentTime

#if 0
#	define VIDEO_FILENAME "fabric-videoenc.m4v"
#	define VIDEO_ENCODER Tempest::NVVideoEncoder
#else
#	define VIDEO_FILENAME "fabric-videoenc.ivf"
#	define VIDEO_ENCODER Tempest::VPXVideoEncoder
#endif

#define RECORD_FILE "record.bin"

#define VSYNC 0

#define BASE_TEXTURE_NAME TEST_ASSETS_DIR "/alcantara/alcantara"

//#define TANGENT_PERTURBATION

const uint32_t FPS = 30;
const uint32_t Bitrate = 2500;

#define SGGX_SURFACE            3
#define SGGX_PSEUDO_VOLUME		4

#define MATERIAL_TYPE SGGX_SURFACE

#ifdef TANGENT_PERTURBATION
const float StandardDeviationCloth = 2.0f/3.0f; // 2.0f/(spec_power + 2.0)
#else
#endif

CONSTANT float FiberLength = 4.0f;
EXPORT_CONSTANT float TouchRadius = 15.0f;
CONSTANT float MaximumStretch = 1.0f;

#ifdef ENABLE_PHYSICS
CONSTANT Tempest::Vector2 ResponsePreference = { 0.0f, 1.0f };
CONSTANT float SimulationThreshold = 30e-3f;
#endif

CONSTANT float TouchStrength = 2.0f;
#ifdef ENABLE_PHYSICS
const uint64_t UpdateStep = 1000;
#endif
const float Velocity = 10e-7f;
const float WheelVelocity = 1e-3f;
const float DragVelocity = 1e-2f;
const float MouseSpeed = 1e-2f;


inline EXPORT_CUDA Tempest::Vector3 SamplePerturbatedDirection(const Tempest::Vector3& sggx_stddev, const Tempest::Matrix3& basis, uint32_t seed)
{
    Tempest::Matrix3 alt_matrix = basis;
    alt_matrix.scale(sggx_stddev);

    auto uniform_sample = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
    auto fiber_tan = Normalize(alt_matrix.transformRotationInverse(uniform_sample));
    if(Tempest::Dot(fiber_tan, basis.tangent()) < 0.0f)
        fiber_tan = -fiber_tan;
    return fiber_tan;
}

RASTERIZER_FUNCTION void ClothInteractionShader::operator()(uint32_t x, uint32_t y, uint32_t width, uint32_t height, const Tempest::Vector2& dist_vector, float tangent_ratio)
{
    uint32_t index = y*width + x;

    auto old_stretch_dir = Data[index].Stretch;
    /*
    Tempest::Vector2 force_dir = dist_vector;
    NormalizeSelf(&force_dir);
    force_dir *= MaximumStretch;
    /*/
                                               
    Tempest::Vector2 force_dir = ForceDirection;
    Tempest::Vector2 norm_force_dir = force_dir/ForceMagnitude;
  
    float binorm_ratio = fabsf(Tempest::WedgeZ(dist_vector, norm_force_dir));

    float circular_shift = sqrtf(TouchRadius*TouchRadius - binorm_ratio*binorm_ratio)/ForceMagnitude;

    float tr = circular_shift ? 1.0f - Tempest::Clamp(-(tangent_ratio - circular_shift)/(2.0f*circular_shift), 0.0f, 1.0f) : 1.0f;
                                              
    float falloff = fabsf(binorm_ratio)/TouchRadius;

    tr *= Tempest::Clampf((1.0f - falloff)*TouchStrength, 0.0f, 1.0f);

    //tr *= expf(-falloff*falloff*10.0f);

    //stretch_dir += force_dir;
    auto stretch_dir = 10.0f * norm_force_dir; // * Maxf(1 - falloff, 0.0f) /* * exp(-falloff*falloff*10.0f/(TouchRadius*TouchRadius)) */;
    float len = Length(stretch_dir);

    float maximum_stretch = MaximumStretch;

    // TODO: Apply Curve
    if(len > maximum_stretch)
    {
        stretch_dir *= maximum_stretch/len;
    }

    auto& spring_data = Data[index];

    if(old_stretch_dir.x || old_stretch_dir.y)
    {
        Tempest::Matrix2 mat = Tempest::Matrix2::slerpLerpMatrix(old_stretch_dir, stretch_dir, tr);
        spring_data.Stretch = mat.transform(old_stretch_dir);
    //     spring_data.Stretch = old_stretch_dir * (1 - tr) + stretch_dir * tr;
    }
    else
    {
        spring_data.Stretch = stretch_dir * tr;
    }
#ifdef ENABLE_PHYSICS
    spring_data.Velocity = {};
#endif
}

struct TestRotation
{
    SpringData*			Data;
    uint32_t			Width;
    uint32_t            Height;
    Tempest::Vector2    Rotation;

    RASTERIZER_FUNCTION void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
    {
        if(x >= Width || y >= Height)
            return;

        Data[y*Width + x].Stretch = Rotation;
    }
};


RASTERIZER_FUNCTION void ClothDataDrivenInteractionConversion::operator()(uint32_t worker_id, uint32_t x, uint32_t y)
{
    if(x >= Width || y >= Height)
        return;

    auto& spring_data = Data[y*Width + x];

    auto damp_mag = Tempest::Length(spring_data.Stretch);
    Tempest::Vector2 plane_rod_dir = damp_mag > 0.0f ? spring_data.Stretch/damp_mag : Tempest::Vector2{ 0.0f, 1.0f };

    float idx_f = TGE_FIXED_ARRAY_SIZE(BasisTextures)*Tempest::Clampf((atan2f(plane_rod_dir.y, plane_rod_dir.x) + Tempest::MathPi)/Tempest::MathTau, 0.0f, 1.0f);

    int idx_trunc = static_cast<int>(idx_f);
    float factor = idx_f - idx_trunc;

    int i0 = Tempest::Modulo(idx_trunc, (int)TGE_FIXED_ARRAY_SIZE(BasisTextures)),
        i1 = (i0 + 1) % (int)TGE_FIXED_ARRAY_SIZE(BasisTextures);

        Tempest::Vector2 tc{ RepeatFactor*x/Width, RepeatFactor*y/Height };

    {
    auto diffuse_tex0 = DiffuseTextures[i0],
         diffuse_tex1 = DiffuseTextures[i1];

    auto diffuse0 = Tempest::SampleRGBA(diffuse_tex0, tc),
         diffuse1 = Tempest::SampleRGBA(diffuse_tex1, tc);

    auto mix_diffuse = Tempest::GenericLinearInterpolate(diffuse0, diffuse1, factor)*RemapDiffuseColor*DiffuseMultiplier;

    Tempest::Surface2DWrite(mix_diffuse, MixDiffuseTextureData, x*sizeof(Tempest::Vector4), y);
    }

    {
    auto specular_tex0 = SpecularTextures[i0],
         specular_tex1 = SpecularTextures[i1];

    auto specular0 = Tempest::SampleRGBA(specular_tex0, tc),
         specular1 = Tempest::SampleRGBA(specular_tex1, tc);

    auto mix_specular = Tempest::GenericLinearInterpolate(specular0, specular1, factor)*RemapSpecularColor*SpecularMultiplier;

    Tempest::Surface2DWrite(mix_specular, MixSpecularTextureData, x*sizeof(Tempest::Vector4), y);
    }
             
    {
    auto stddev_tex0 = StandardDeviationTextures[i0],
         stddev_tex1 = StandardDeviationTextures[i1];

    auto stddev0 = Tempest::SampleRG(stddev_tex0, tc),
         stddev1 = Tempest::SampleRG(stddev_tex1, tc);

    auto mix_stddev = Tempest::GenericLinearInterpolate(stddev0, stddev1, factor);

    float center_stddev = (mix_stddev.x + mix_stddev.y)*0.5f;

    mix_stddev = AnisotropyModifier > 0.5f ? mix_stddev*(2.0f - 2.0f*AnisotropyModifier) + (2.0f*AnisotropyModifier - 1.0f)*Tempest::Vector2{ float(mix_stddev.x > center_stddev), float(mix_stddev.y > center_stddev) }:
                                             mix_stddev*2.0f*AnisotropyModifier + (1.0f - 2.0f*AnisotropyModifier)*center_stddev;

    float smoothness = 1.0f - SmoothnessModifier;
    mix_stddev = smoothness > 0.5f ? mix_stddev*(2.0f - 2.0f*smoothness) + (2.0f*smoothness - 1.0f):
                                     mix_stddev*(2.0f*smoothness);

    Tempest::Surface2DWrite(mix_stddev, MixStandardDeviationTextureData, x*sizeof(Tempest::Vector2), y);
    }

    {
    auto basis_tex0 = BasisTextures[i0],
            basis_tex1 = BasisTextures[i1];

    Tempest::Quaternion basis0, basis1;
    basis0.V4 = Tempest::SampleRGBA(basis_tex0, tc);
    basis1.V4 = Tempest::SampleRGBA(basis_tex1, tc);

    auto quat = Tempest::Slerp(basis0, basis1, factor);

    Tempest::Surface2DWrite(quat, MixTangentTextureData, x*sizeof(Tempest::Vector4), y);
    }
}

RASTERIZER_FUNCTION void ClothInteractionConversion::operator()(uint32_t worker_id, uint32_t x, uint32_t y)
{
    if(x >= Width || y >= Height)
        return;

    auto& spring_data = Data[y*Width + x];

    auto damp_mag = Tempest::Length(spring_data.Stretch);
    float sin_theta = fabsf(damp_mag)/FiberLength;
    float cos_theta = sqrtf(1.0f - sin_theta*sin_theta);

    Tempest::Vector2 plane_rod_dir = damp_mag > 0.0f ? spring_data.Stretch/damp_mag : Tempest::Vector2{};
    Tempest::Vector3 rod_dir{ plane_rod_dir.x*sin_theta, plane_rod_dir.y*sin_theta, cos_theta };
    auto rod_quat = Tempest::ToQuaternionNormal(rod_dir);
    Tempest::Surface2DWrite(rod_quat, MixTangentTextureData, x*sizeof(Tempest::Vector4), y);
}

#ifdef ENABLE_PHYSICS
struct Spring
{
    float Mass;
    float Damping;
    float SpringConstant;
};

struct ClothPhysicsSimulation
{
	bool				Paint = false;
	Tempest::Vector2	Cursor;
	SpringData*			Data;
	void*				MixTangentTextureData;

	uint32_t			Width;
    uint32_t            Height;
	uint64_t			PreviousUpdate;
	uint64_t			CurrentTime;

    Spring              StructuralSprings,
                        ShearingSprings,
                        BendingSprings,
					    AnchorSprings;

    ClothPhysicsSimulation()
    {
  //      StructuralSprings = { 20.0f, 100.0f, 50.0f };
  //      ShearingSprings = { 20.0f, 100.0f, 50.0f };
  //      BendingSprings = { 20.0f, 100.0f, 50.0f };
  //	  AnchorSprings = { 20.0f, 100.0f, 10.0f };
        StructuralSprings = { 20.0f, 100.0f, 0.0f };
        ShearingSprings = { 20.0f, 100.0f, 0.0f };
        BendingSprings = { 20.0f, 100.0f, 0.0f };
		AnchorSprings = { 20.0f, 100.0f, 1.0f };
    }

	RASTERIZER_FUNCTION void operator()(uint32_t worker_id, uint32_t x, uint32_t y)
	{
        if(x >= Width || y >= Height)
            return;

        auto& spring_data = Data[y*Width + x];
        if(spring_data.Stretch.x == 0.0f && spring_data.Stretch.y == 0.0f)
            return;

        if(Paint)
        {
            auto dist_vec = Tempest::Vector2{ (float)x, (float)y } - Cursor;
            if(Tempest::Dot(dist_vec, dist_vec) < TouchRadius*TouchRadius)
            {
                auto damp_mag = Tempest::Length(spring_data.Stretch);
                float sin_theta = fabsf(damp_mag)/FiberLength;
                float cos_theta = sqrtf(1.0f - sin_theta*sin_theta);

                Tempest::Vector2 plane_rod_dir = damp_mag > 0.0f ? spring_data.Stretch/damp_mag : Tempest::Vector2{};
                Tempest::Vector3 rod_dir{ plane_rod_dir.x*sin_theta, plane_rod_dir.y*sin_theta, cos_theta };
                auto rod_quat = Tempest::ToQuaternionNormal(rod_dir);
                Tempest::Surface2DWrite(rod_quat, MixTangentTextureData, x*sizeof(Tempest::Vector4), y);
                return;
            }
        }

        //uint32_t x_end = Width, y_end = Height;

        auto update_time = PreviousUpdate;
        for(; CurrentTime - update_time > UpdateStep; update_time += UpdateStep)
        {
            float step_size = UpdateStep*1e-6f;
            
			auto anchoring_damping = AnchorSprings.Damping;
			auto anchoring_sprconst = AnchorSprings.SpringConstant;

            /*
            auto structural_damping_x = StructuralSprings.Damping;
			auto structural_sprconst_x = StructuralSprings.SpringConstant;

			auto structural_damping_y = StructuralSprings.Damping;
			auto structural_sprconst_y = StructuralSprings.SpringConstant;

			auto shearing_damping = ShearingSprings.Damping;
			auto shearing_sprconst = ShearingSprings.SpringConstant;

            if(x > 0)
            {
                auto& prev_point = Data[y*x_end + x - 1];
                Tempest::SolveSecondOrderLinearODEStretch(StructuralSprings.Mass, structural_damping_x, structural_sprconst_x, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(x < x_end - 1)
            {
                auto& prev_point = Data[y*x_end + x + 1];
                Tempest::SolveSecondOrderLinearODEStretch(StructuralSprings.Mass, structural_damping_x, structural_sprconst_x, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(y > 0)
            {
                auto& prev_point = Data[(y - 1)*x_end + x];
                Tempest::SolveSecondOrderLinearODEStretch(StructuralSprings.Mass, structural_damping_y, structural_sprconst_y, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(y < y_end - 1)
            {
                auto& prev_point = Data[(y + 1)*x_end + x];
                Tempest::SolveSecondOrderLinearODEStretch(StructuralSprings.Mass, structural_damping_y, structural_sprconst_y, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(x > 0 && y > 0)
            {
                auto& prev_point = Data[(y - 1)*x_end + x - 1];
                Tempest::SolveSecondOrderLinearODEStretch(ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(x > 0 && y < y_end - 1)
            {
                auto& prev_point = Data[(y + 1)*x_end + x - 1];
                Tempest::SolveSecondOrderLinearODEStretch(ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(x < x_end - 1 && y > 0)
            {
                auto& prev_point = Data[(y - 1)*x_end + x + 1];
                Tempest::SolveSecondOrderLinearODEStretch(ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }

            if(x < x_end - 1 && y < y_end - 1)
            {
                auto& prev_point = Data[(y + 1)*x_end + x + 1];
                Tempest::SolveSecondOrderLinearODEStretch(ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, prev_point.Stretch, &spring_data.Stretch, &spring_data.Velocity);
            }
            //*/
			Tempest::SolveSecondOrderLinearODE(AnchorSprings.Mass, anchoring_damping, anchoring_sprconst, 0.0f, step_size, &spring_data.Stretch, &spring_data.Velocity);

            auto damp_mag = Tempest::Length(spring_data.Stretch);
            if(damp_mag < SimulationThreshold && Tempest::Length(spring_data.Velocity) < SimulationThreshold)
            {
                auto rod_dir = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

                auto rod_quat = Tempest::ToQuaternionNormal(rod_dir);
                Tempest::Surface2DWrite(rod_quat, MixTangentTextureData, x*sizeof(Tempest::Vector4), y);

				spring_data.Stretch = {};
                spring_data.Velocity = {};
                break;
            }

            Tempest::Vector2 plane_rod_dir = spring_data.Stretch/damp_mag;

            float sin_theta = fabsf(damp_mag)/FiberLength;
            float cos_theta = sqrtf(1.0f - sin_theta*sin_theta);

            // Lame dynamics
            Tempest::Vector3 rod_dir{ plane_rod_dir.x*sin_theta, plane_rod_dir.y*sin_theta, cos_theta };

            auto rod_quat = Tempest::ToQuaternionNormal(rod_dir);
            Tempest::Surface2DWrite(rod_quat, MixTangentTextureData, x*sizeof(Tempest::Vector4), y);
        }
	}
};
#endif

#ifdef CUDA_ACCELERATED
#   define EXECUTE_PARALLEL_FOR_LOOP_2D ExecuteParallelForLoop2DGPU
#else
#   define EXECUTE_PARALLEL_FOR_LOOP_2D ExecuteParallelForLoop2DCPU
#endif

MaterialManipulation::~MaterialManipulation()
{
    auto& backend = m_RayTracingSystem->getBackend();
    auto& compiler = m_RayTracingSystem->getShaderCompiler();
    backend.destroyRenderResource(m_CopyCommandBuffer);
    backend.destroyRenderResource(m_TouchCommandBuffer);

    backend.destroyRenderResource(m_BackbufferStorage);

    backend.destroyRenderResource(m_ImageStateObject);
    compiler.destroyRenderResource(m_ImageShader);
    backend.destroyRenderResource(m_IntermediateRenderTarget);
    backend.destroyRenderResource(m_IntermediateFramebuffer);
    backend.destroyRenderResource(m_TouchImage);
    backend.destroyRenderResource(m_RotateImage);
}

bool MaterialManipulation::init(uint32_t image_width, uint32_t image_height, uint32_t flags, Tempest::PreferredWindow* window_ptr, Tempest::PreferredBackend* backend_ptr, Tempest::PreferredShaderCompiler* shader_compiler_ptr)
{
#ifdef LOW_SPEC
    flags |= MATERIAL_MANIPULATION_LOW_SPEC;
#endif

    bool status;
    auto projection = Tempest::PerspectiveMatrix(40.0f, (float)image_width/image_height, 0.1f, 1000.0f);

    Tempest::Matrix4 view;

	view.identity();
    view.translate(-m_Offset);
	view.rotateX(Tempest::MathPi*0.5f - m_Roll);
    view.rotateY(-m_Yaw);

    auto view_proj = projection*view;

    m_ViewProjectionInverse = view_proj.inverse();

    Tempest::TextureDescription mix_tex_desc;
    mix_tex_desc.Width = 1024;
    mix_tex_desc.Height = 1024;
    mix_tex_desc.Format = Tempest::DataFormat::R8UNorm;
    size_t mix_tex_area = mix_tex_desc.Width*mix_tex_desc.Height;

    m_Material = std::unique_ptr<Tempest::RTSGGXSurface>(new Tempest::RTSGGXSurface{});
#if MATERIAL_TYPE == SGGX_SURFACE
    m_Material->Model = Tempest::IlluminationModel::SGGXSurface;
#elif MATERIAL_TYPE == SGGX_PSEUDO_VOLUME
    m_Material->Model = Tempest::IlluminationModel::SGGXPseudoVolume;
#else
#   error "Unknown material type"
#endif

    m_RayTracingSystem = decltype(m_RayTracingSystem)(new RAY_TRACING_SYSTEM(image_width, image_height, m_ViewProjectionInverse, {}, window_ptr, backend_ptr, shader_compiler_ptr));
    auto* rt_scene = m_RayTracingSystem->getRayTracer();

    Tempest::Matrix4 world = Tempest::Matrix4::identityMatrix();

    m_Material->Depth = 0;
    m_Material->SampleCount = 256;
    m_Material->BasisMapWidth = mix_tex_desc.Width;
    m_Material->BasisMapHeight = mix_tex_desc.Height;

#ifndef ENABLE_DATA_DRIVEN
    Tempest::Vector2 stddev = { 0.55f, 0.82f };
#endif

	m_SpringGrid = decltype(m_SpringGrid)(new SpringData[mix_tex_area]);

#ifdef ENABLE_DATA_DRIVEN
    std::fill(m_SpringGrid.get(), m_SpringGrid.get() + mix_tex_area, SpringData{ { 0.0f, 1.0f } });
#else
    memset(spring_grid.get(), 0, mix_tex_area*sizeof(SpringData));
#endif

	m_SpringBuffer = reinterpret_cast<SpringData*>(rt_scene->bindBuffer(m_SpringGrid.get(), mix_tex_area*sizeof(SpringData)));

    m_Interaction.Data = m_SpringBuffer;
    m_Interaction.ForceDirection = {};
    m_Interaction.ForceMagnitude = 0.0f;

	m_Simulation.Data = m_SpringBuffer;
	m_Simulation.Width = mix_tex_desc.Width;
    m_Simulation.Height = mix_tex_desc.Height;

#ifndef ENABLE_MESH
    const void* rect_tangent_map = nullptr;
#endif

    Tempest::Quaternion* rotation_map = new Tempest::Quaternion[mix_tex_area];
    for(uint32_t pixel_idx = 0; pixel_idx < mix_tex_area; ++pixel_idx)
    {
        rotation_map[pixel_idx] = Tempest::ToQuaternionNormal(Tempest::Vector3{ 0.0f, 0.0f, 1.0f });
    }

    Tempest::TextureDescription basis_tex_desc;
    basis_tex_desc.Width = mix_tex_desc.Width;
    basis_tex_desc.Height = mix_tex_desc.Height;
    basis_tex_desc.Format = Tempest::DataFormat::RGBA32F;
    m_BasisMap = Tempest::TexturePtr(new Tempest::Texture(basis_tex_desc, reinterpret_cast<uint8_t*>(rotation_map)));

    rt_scene->bindSurfaceAndTexture(m_BasisMap.get(), &m_Material->BasisMap, &m_Simulation.MixTangentTextureData);

#ifdef ENABLE_DATA_DRIVEN
    const char* MaterialNames[8] =
    {
        BASE_TEXTURE_NAME "_center_left",   // '-'
        BASE_TEXTURE_NAME "_bottom_left",   // '/'
        BASE_TEXTURE_NAME "_bottom_center", // '|'
        BASE_TEXTURE_NAME "_bottom_right",  // '\'
        BASE_TEXTURE_NAME "_center_right",  // '-'
        BASE_TEXTURE_NAME "_top_right",     // '/'
        BASE_TEXTURE_NAME "_top_center",    // '|'
        BASE_TEXTURE_NAME "_top_left"       // '\'
    };

    Tempest::TexturePtr surface_textures[8*4];

    for(size_t tex_idx = 0, garbage_idx = 0, tex_count = TGE_FIXED_ARRAY_SIZE(MaterialNames); tex_idx < tex_count; ++tex_idx)
    {
        {
        std::string basis_texname = std::string(MaterialNames[tex_idx]) + "_sggx_basis.exr";
        auto& cur_basis_tex = surface_textures[garbage_idx++] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(basis_texname)));
        if(!cur_basis_tex)
        {
            Tempest::CrashMessageBox("Error", "Failed to load basis texture: ", basis_texname);
            return false;
        }
        m_Simulation.BasisTextures[tex_idx] = rt_scene->bindTexture(cur_basis_tex.get());
        }

        {
        std::string stddev_texname = std::string(MaterialNames[tex_idx]) + "_sggx_scale.exr";
        auto& cur_stddev_tex = surface_textures[garbage_idx++] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(stddev_texname)));
        if(!cur_stddev_tex)
        {
            Tempest::CrashMessageBox("Error", "Failed to load standard deviation texture: ", stddev_texname);
            return false;
        }
        m_Simulation.StandardDeviationTextures[tex_idx] = rt_scene->bindTexture(cur_stddev_tex.get());
        }

        {
        std::string diffuse_texname = std::string(MaterialNames[tex_idx]) + "_albedo.exr";
        auto& cur_diffuse_tex = surface_textures[garbage_idx++] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(diffuse_texname)));
        if(!cur_diffuse_tex)
        {
            Tempest::CrashMessageBox("Error", "Failed to load diffuse texture: ", diffuse_texname);
            return false;
        }
        cur_diffuse_tex->convertToRGBA();
        m_Simulation.DiffuseTextures[tex_idx] = rt_scene->bindTexture(cur_diffuse_tex.get());
        }

        {
        std::string specular_texname = std::string(MaterialNames[tex_idx]) + "_specular.exr";
        auto& cur_specular_tex = surface_textures[garbage_idx++] = Tempest::TexturePtr(Tempest::LoadImage(Tempest::Path(specular_texname)));
        if(!cur_specular_tex)
        {
            Tempest::CrashMessageBox("Error", "Failed to load specular texture: ", specular_texname);
            return false;
        }
        cur_specular_tex->convertToRGBA();
        m_Simulation.SpecularTextures[tex_idx] = rt_scene->bindTexture(cur_specular_tex.get());
        }
    }

    auto& diffuse_map = surface_textures[2];
    auto& diffuse_hdr = diffuse_map->getHeader();
    Tempest::Vector3 best_diffuse{};
    float best_luminance = 0.0f;
    for(uint32_t y = 0, yend = diffuse_hdr.Height; y < yend; ++y)
        for(uint32_t x = 0, xend = diffuse_hdr.Width; x < xend; ++x)
        {
            Tempest::Vector3 diffuse = diffuse_map->fetchRGB(x, y);
            float luminance = Tempest::RGBToLuminance(diffuse);
            if(luminance > best_luminance)
            {
                best_diffuse = diffuse;
                best_luminance = luminance;
            }
        }
    
    m_BaseDiffuseColor = Tempest::ToVector4(best_diffuse);

    auto& specular_map = surface_textures[3];
    auto& specular_hdr = specular_map->getHeader();
    Tempest::Vector3 best_specular{};
    best_luminance = 0.0f;
    for(uint32_t y = 0, yend = specular_hdr.Height; y < yend; ++y)
        for(uint32_t x = 0, xend = specular_hdr.Width; x < xend; ++x)
        {
            Tempest::Vector3 specular = diffuse_map->fetchRGB(x, y);
            float luminance = Tempest::RGBToLuminance(specular);
            if(luminance > best_luminance)
            {
                best_specular = specular;
                best_luminance = luminance;
            }
        }

    m_BaseSpecularColor = Tempest::ToVector4(best_specular);

    m_Material->StandardDeviation = { 1.0f, 1.0f };

    Tempest::TextureDescription stddev_tex_desc;
    stddev_tex_desc.Width = mix_tex_desc.Width;
    stddev_tex_desc.Height = mix_tex_desc.Height;
    stddev_tex_desc.Format = Tempest::DataFormat::RG32F;
    
    Tempest::Vector2* stddev_map_data = new Tempest::Vector2[mix_tex_area];
    m_StandardDeviationMap = Tempest::TexturePtr(new Tempest::Texture(stddev_tex_desc, reinterpret_cast<uint8_t*>(stddev_map_data)));

    rt_scene->bindSurfaceAndTexture(m_StandardDeviationMap.get(), &m_Material->StandardDeviationMap, &m_Simulation.MixStandardDeviationTextureData);

    Tempest::TextureDescription diffuse_tex_desc;
    diffuse_tex_desc.Width = mix_tex_desc.Width;
    diffuse_tex_desc.Height = mix_tex_desc.Height;
    diffuse_tex_desc.Format = Tempest::DataFormat::RGBA32F;

    Tempest::Vector4* diffuse_map_data = new Tempest::Vector4[mix_tex_area];
    m_DiffuseMap = Tempest::TexturePtr(new Tempest::Texture(diffuse_tex_desc, reinterpret_cast<uint8_t*>(diffuse_map_data)));

    rt_scene->bindSurfaceAndTexture(m_DiffuseMap.get(), &m_Material->DiffuseMap, &m_Simulation.MixDiffuseTextureData);

    Tempest::TextureDescription specular_tex_desc;
    specular_tex_desc.Width = mix_tex_desc.Width;
    specular_tex_desc.Height = mix_tex_desc.Height;
    specular_tex_desc.Format = Tempest::DataFormat::RGBA32F;

    Tempest::Vector4* specular_map_data = new Tempest::Vector4[mix_tex_area];
    m_SpecularMap = Tempest::TexturePtr(new Tempest::Texture(specular_tex_desc, reinterpret_cast<uint8_t*>(specular_map_data)));

    rt_scene->bindSurfaceAndTexture(m_SpecularMap.get(), &m_Material->SpecularMap, &m_Simulation.MixSpecularTextureData);

    m_Material->Specular = Tempest::ToSpectrum(1.0f);
    m_Material->Diffuse = Tempest::ToSpectrum(1.0f);
#else
    m_Material->StandardDeviation = { stddev.x, stddev.y };

    m_Material->Specular = Tempest::RGBToSpectrum(Tempest::Vector3{0.36f, 0.23f, 0.15f});
    m_Material->Diffuse = {};
#endif

	m_Material->setup();

#ifdef ENABLE_MESH
	Tempest::RTMeshBlob mesh_blob;

    if(flags & MATERIAL_MANIPULATION_LOW_SPEC)
    {
        status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/cloth/cloth.obj", nullptr, &mesh_blob, Tempest::TEMPEST_OBJ_LOADER_GENERATE_TANGENTS|Tempest::TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS);
    }
    else
    {
        status = Tempest::LoadObjFileStaticRTGeometry(TEST_ASSETS_DIR "/cloth/clothsuperhd.obj", nullptr, &mesh_blob, Tempest::TEMPEST_OBJ_LOADER_GENERATE_TANGENTS|Tempest::TEMPEST_OBJ_LOADER_GENERATE_CONSISTENT_NORMALS);
    }
    TGE_ASSERT(status, "Failed to load test assets");
    if(!status)
    {
        Tempest::Log(Tempest::LogLevel::Error, "Missing cloth mesh");
        return false;
    }

	// This one is only if you want to run GPU accelerated code - you need to rebind materials - i will probably make a convenience function for this case
#if defined(CUDA_ACCELERATED) || defined(GPU_RASTERIZER)
	Tempest::RebindMaterialsToGPU(rt_scene, mesh_blob);
#endif

	auto submesh_ids = TGE_TYPED_ALLOCA(uint64_t, mesh_blob.SubmeshCount);

	const uint32_t draw_surf_idx = 2;

	mesh_blob.Submeshes[draw_surf_idx].Material = m_Material.get();

    Tempest::MeshOptions mesh_opts;
    mesh_opts.TwoSided = true;

	rt_scene->addTriangleMesh(Tempest::Matrix4::identityMatrix(), mesh_blob.SubmeshCount, mesh_blob.Submeshes,
							    mesh_blob.IndexData.size()/3, &mesh_blob.IndexData.front(), mesh_blob.VertexData.size(), &mesh_blob.VertexData.front(), &mesh_opts, submesh_ids);

	m_PlaneID = submesh_ids[draw_surf_idx];
#else
    Tempest::Vector2 rect_size{2.0f, 2.0f};

    m_PlaneID = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f}, Tempest::Vector3{0.0f, 0.0f, 1.0f}, rect_size, plane_mtl.get(), rect_tangent_map);
#endif

#ifdef ENABLE_DATA_DRIVEN
    Tempest::EXECUTE_PARALLEL_FOR_LOOP_2D(rt_scene->getThreadId(), rt_scene->getThreadPool(), (uint32_t)mix_tex_desc.Width, (uint32_t)mix_tex_desc.Height, m_Simulation);
#endif

    if(flags & MATERIAL_MANIPULATION_AREA_LIGHT)
    {
        Tempest::SphereAreaLight* area_light1 = new Tempest::SphereAreaLight;
        area_light1->SphereShape.Center = Tempest::Vector3{1.5f, 4.0f, 1.5f};
        area_light1->SphereShape.Radius = 0.1f;
        area_light1->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{2000.0f, 2000.0f, 2000.0f});

        rt_scene->addSphereLightSource(area_light1);
    }
    else
    {
        Tempest::PointLight* point_light = new Tempest::PointLight;
        point_light->Position = Tempest::Vector3{1.5f, 4.0f, 1.5f};
        point_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{600.0f, 600.0f, 600.0f});

        rt_scene->addLightSource(point_light);
    }

    rt_scene->setSamplesCamera(1);
    rt_scene->setSamplesGlobalIllumination(4);

    if(flags & MATERIAL_MANIPULATION_LOW_SPEC)
    {
        rt_scene->setSamplesLocalAreaLight(1);
    }
    else
    {
        uint32_t sample_count = (flags & MATERIAL_MANIPULATION_AREA_LIGHT) ? 4 : 1;
        rt_scene->setSamplesLocalAreaLight(sample_count);
    }
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::SRGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    rt_scene->commitScene();

    m_RayTracingSystem->startRendering();

    auto& window = m_RayTracingSystem->getWindow();
    window.setEventMask(Tempest::COLLECT_MOUSE_EVENTS|Tempest::COLLECT_WINDOW_EVENTS);

    m_ImageSize = { (float)mix_tex_desc.Width, (float)mix_tex_desc.Height };

    m_Interaction.Cursor = {};

#ifdef ENABLE_PHYSICS
    m_Simulation.PreviousUpdate = prev_time;
#endif

    // auto plane_id = rt_scene->addRect(Tempest::Vector3{0.0f, 0.0f, 0.0f}, Tempest::Vector3{1.0f, 0.0f, 0.0f},
	//								  Tempest::Vector3{0.0f, 0.0f, 1.0f}, rect_size, plane_mtl.get(), mix_tangent_texture_obj);

    if(m_Status & OPTION_PLAYBACK_AND_RECORD)
    {
        m_RecordStream.open(RECORD_FILE, std::ios::binary|std::ios::in);
        if(!m_RecordStream)
        {
            Tempest::CrashMessageBox("Error", "Failed to open file for playback");
            return false;
        }

        m_RecordStream.read(reinterpret_cast<char*>(&m_EventRecord), sizeof(m_EventRecord));
        if(!m_RecordStream)
        {
            m_RecordStream.close();
            return false;
        }
        m_RecordStatus = RecordStatus::Playback;

	    Tempest::VideoInfo video_info;
        video_info.FileName = VIDEO_FILENAME;
        video_info.FPS = FPS;
        video_info.Bitrate = Bitrate;
        video_info.Width = image_width;
        video_info.Height = image_height;

        m_VideoEncoder = decltype(m_VideoEncoder)(new VIDEO_ENCODER);
        status = m_VideoEncoder->openStream(video_info);
        if(!status)
        {
            Tempest::CrashMessageBox("Error", "Failed to open video stream for encoding: " VIDEO_FILENAME);
            return false;
        }
    }

	auto& backend = m_RayTracingSystem->getBackend();

	Tempest::CommandBufferDescription cmd_desc;
	cmd_desc.CommandCount = 1;
	cmd_desc.ConstantsBufferSize = 1024;

	m_TouchCommandBuffer = backend.createCommandBuffer(cmd_desc);

    Tempest::BasicFileLoader loader;
	m_ImageShader = m_RayTracingSystem->getShaderCompiler().compileShaderProgram(CURRENT_SOURCE_DIR "/image-draw.tfx", &loader);
    TGE_ASSERT(m_ImageShader, "Expecting successful compilation");
    m_ImageResourceTable = Tempest::CreateResourceTable(m_ImageShader, "Globals", 1);
    TGE_ASSERT(m_ImageResourceTable, "Expecting valid resource table");
	m_TransformIndex = m_ImageResourceTable->getResourceIndex("Globals[0].Transform");

    {
    std::unique_ptr<Tempest::Texture> tex(Tempest::LoadImage(Tempest::Path(TEST_ASSETS_DIR "/hand/touch.png")));
	m_TouchImage = tex ? backend.createTexture(tex->getHeader(), 0, tex->getData()) : nullptr;
    }
    TGE_ASSERT(m_TouchImage, "Failed to get touch texture");
	m_TouchImage->setFilter(Tempest::FilterMode::Linear, Tempest::FilterMode::Linear, Tempest::FilterMode::Linear);
    m_TouchImage->setWrapMode(Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp);

	auto res_table = Tempest::CreateResourceTable(m_ImageShader, "Resources", 0);
    res_table->setResource("Texture", *m_TouchImage);
	m_TouchBakedTable = Tempest::ExtractBakedResourceTable(res_table.get());
	res_table->resetBakedTable();

    {
    std::unique_ptr<Tempest::Texture> tex(Tempest::LoadImage(Tempest::Path(TEST_ASSETS_DIR "/hand/rotate.png")));
	m_RotateImage = tex ? backend.createTexture(tex->getHeader(), 0, tex->getData()) : nullptr;
    }
    TGE_ASSERT(m_RotateImage, "Failed to get touch texture");
	m_RotateImage->setFilter(Tempest::FilterMode::Linear, Tempest::FilterMode::Linear, Tempest::FilterMode::Linear);
    m_RotateImage->setWrapMode(Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp, Tempest::WrapMode::Clamp);

	res_table->setResource("Texture", *m_RotateImage);
	m_RotateBakedTable = Tempest::ExtractBakedResourceTable(res_table.get());

	Tempest::Vector4 transform_touch{ 1.0f, 1.0f, 0.0f, 0.0f };

	m_ImageResourceTable->setResource(m_TransformIndex, transform_touch);

    Tempest::BlendStates blend_states;
    auto& blend_rt = blend_states.RenderTargets[0];
    blend_rt.BlendEnable = true;
    blend_rt.SrcBlend = Tempest::BlendFactorType::SrcAlpha;
    blend_rt.DstBlend = Tempest::BlendFactorType::InvSrcAlpha;
    blend_rt.DstBlendAlpha = Tempest::BlendFactorType::One;

	Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
	m_ImageStateObject = backend.createStateObject(&rt_fmt, 1, Tempest::DataFormat::Unknown, m_ImageShader, Tempest::DrawModes::TriangleStrip, nullptr, &blend_states);

	Tempest::PreferredBackend::CommandBufferType::DrawBatchType draw_batch;
	draw_batch.BaseIndex = 0;
    draw_batch.BaseVertex = 0;
    draw_batch.IndexBuffer = nullptr;
    draw_batch.VertexBuffers[0].Offset = 0;
    draw_batch.VertexBuffers[0].Stride = 0;
    draw_batch.VertexBuffers[0].VertexBuffer = nullptr;
    draw_batch.VertexCount = 4;
    draw_batch.SortKey = ~0;
    draw_batch.ResourceTable = m_ImageResourceTable->getBakedTable();
    draw_batch.PipelineState = m_ImageStateObject;

	m_TouchCommandBuffer->enqueueBatch(draw_batch);

	Tempest::TextureDescription rt_tex_fmt;
	rt_tex_fmt.Width = image_width;
    rt_tex_fmt.Height = image_height;
	rt_tex_fmt.Format = Tempest::DataFormat::RGBA8UNorm;
	m_IntermediateRenderTarget = backend.createRenderTarget(rt_tex_fmt, 0);

	Tempest::PreferredBackend::RenderTargetType* rts[] = { m_IntermediateRenderTarget };

	m_IntermediateFramebuffer = backend.createFramebuffer(rts, TGE_FIXED_ARRAY_SIZE(rts));

    uint32_t data_size = rt_tex_fmt.Width*rt_tex_fmt.Height*Tempest::DataFormatElementSize(rt_tex_fmt.Format);
	m_BackbufferStorage = backend.createStorageBuffer(Tempest::StorageMode::PixelPack, data_size);

    Tempest::IOCommandBufferDescription io_desc;
	io_desc.CommandCount = 1;
	m_CopyCommandBuffer = backend.createIOCommandBuffer(io_desc);

	Tempest::PreferredBackend::IOCommandBufferType::IOCommandType copy_backbuffer_cmd;
	copy_backbuffer_cmd.CommandType = Tempest::IOCommandMode::CopyTextureToStorage;
    copy_backbuffer_cmd.Source.Texture = m_IntermediateRenderTarget;
    copy_backbuffer_cmd.Destination.Storage = m_BackbufferStorage;
	copy_backbuffer_cmd.SourceCoordinate.X = 0;
	copy_backbuffer_cmd.SourceCoordinate.Y = 0;
	copy_backbuffer_cmd.Width = rt_tex_fmt.Width;
	copy_backbuffer_cmd.Height = rt_tex_fmt.Height;
    copy_backbuffer_cmd.DestinationOffset = 0;
    m_CopyCommandBuffer->enqueueCommand(copy_backbuffer_cmd);

	m_BackbufferTexture = Tempest::TexturePtr(new Tempest::Texture(rt_tex_fmt, new uint8_t[data_size]));

    return true;
}

bool MaterialManipulation::run(bool swap_buffers)
{
    FRAME_SCOPE();

    auto& gen_tex_desc = m_BasisMap->getHeader();
    auto* rt_scene = m_RayTracingSystem->getRayTracer();
    auto& window = m_RayTracingSystem->getWindow();
    auto& backend = m_RayTracingSystem->getBackend();

    Tempest::Vector2 mouse_pos{ (float)window.getMouseX(), (float)window.getMouseY() };

    uint64_t cur_time;

    auto window_width = window.getWidth(),
         window_height = window.getHeight();

    Tempest::Vector2 window_size{ float(window_width - 1), float(window_height - 1) };

    auto rt_tex_fmt = m_IntermediateRenderTarget->getDescription();
    if(rt_tex_fmt.Width != window_width ||
       rt_tex_fmt.Height != window_height)
    {
        backend.destroyRenderResource(m_IntermediateRenderTarget); m_IntermediateRenderTarget = nullptr;
        backend.destroyRenderResource(m_IntermediateFramebuffer); m_IntermediateFramebuffer = nullptr;
        backend.destroyRenderResource(m_BackbufferStorage); m_BackbufferStorage = nullptr;

	    rt_tex_fmt.Width = window_width;
        rt_tex_fmt.Height = window_height;
	    rt_tex_fmt.Format = Tempest::DataFormat::RGBA8UNorm;
	    m_IntermediateRenderTarget = backend.createRenderTarget(rt_tex_fmt, 0);

	    Tempest::PreferredBackend::RenderTargetType* rts[] = { m_IntermediateRenderTarget };

	    m_IntermediateFramebuffer = backend.createFramebuffer(rts, TGE_FIXED_ARRAY_SIZE(rts));

        uint32_t data_size = rt_tex_fmt.Width*rt_tex_fmt.Height*Tempest::DataFormatElementSize(rt_tex_fmt.Format);
	    m_BackbufferStorage = backend.createStorageBuffer(Tempest::StorageMode::PixelPack, data_size);

	    Tempest::PreferredBackend::IOCommandBufferType::IOCommandType copy_backbuffer_cmd;
	    copy_backbuffer_cmd.CommandType = Tempest::IOCommandMode::CopyTextureToStorage;
        copy_backbuffer_cmd.Source.Texture = m_IntermediateRenderTarget;
        copy_backbuffer_cmd.Destination.Storage = m_BackbufferStorage;
	    copy_backbuffer_cmd.SourceCoordinate.X = 0;
	    copy_backbuffer_cmd.SourceCoordinate.Y = 0;
	    copy_backbuffer_cmd.Width = rt_tex_fmt.Width;
	    copy_backbuffer_cmd.Height = rt_tex_fmt.Height;
        copy_backbuffer_cmd.DestinationOffset = 0;
        m_CopyCommandBuffer->clear();
        m_CopyCommandBuffer->enqueueCommand(copy_backbuffer_cmd);

        m_BackbufferTexture->realloc(rt_tex_fmt);
    }

    {
        CPU_EVENT_SCOPE("RTRT Complete Frame");
        CUDA_EVENT_SCOPE("RTRT Complete Frame");
        m_RayTracingSystem->completeFrame();
    }

    {
        CPU_EVENT_SCOPE("Event process");
        CUDA_EVENT_SCOPE("Event process");

        if(m_PlaybackAndRecord)
        {
            cur_time = m_PreviousTime + 1000000/FPS;
        }
        else
        {
            cur_time = m_Timer.time();
        }

        // TODO: Multithreading
        bool ignore_event = false;
        Tempest::WindowSystemEvent wevent;
        for(;;)
        {
            auto status = window.getEvent(&wevent);
            if((m_RecordStatus == RecordStatus::Playback &&
                (wevent.Type != Tempest::WindowEventType::KeyReleased && wevent.Key != Tempest::KeyboardKey::Key_P || !status)) ||
                (m_Status & OPTION_PLAYBACK_AND_RECORD))
            {
                auto elapsed_time = cur_time - m_RecordStart;
                if(m_EventRecord.Timestamp > elapsed_time)
                    break;
                
                memcpy(&wevent, &m_EventRecord.Event, sizeof(wevent));
                m_RecordStream.read(reinterpret_cast<char*>(&m_EventRecord), sizeof(m_EventRecord));
                if(!m_RecordStream)
                {
                    if(m_Status & OPTION_PLAYBACK_AND_RECORD)
                    {
                        return false;
                    }
                    else
                    {
                        m_RecordStream.close();
                        m_RecordStatus = RecordStatus::Idle;
                    }
                }
            }
            else if(!status)
                break;

            switch(wevent.Type)
            {
            case Tempest::WindowEventType::Resize:
            {
                m_Status |= MOVEMENT_FLUSH;
            } break;
            case Tempest::WindowEventType::MouseButtonPressed:
            {
                if(!(m_Status & STATUS_ACTIVE))
                    break;

                switch(wevent.MouseButton)
                {
                case Tempest::MouseButtonId::LeftButton:
                {
                    Tempest::SampleData sample_data;
				    auto geom_id = rt_scene->rayQuery(mouse_pos/window_size, &sample_data);
                    if(geom_id != m_PlaneID)
					    break;
                    m_Interaction.Cursor = sample_data.TexCoord*m_ImageSize; 
				    window.captureMouse();
                    m_Simulation.Paint = true;
                } break;
                case Tempest::MouseButtonId::RightButton:
                {
                    m_Status |= MOVEMENT_ROTATION;
                } break;
				case Tempest::MouseButtonId::MiddleButton:
				{
					m_Status |= MOVEMENT_DRAG;
				} break;
                }
            } break;
            case Tempest::WindowEventType::MouseButtonReleased:
            {
                switch(wevent.MouseButton)
                {
                case Tempest::MouseButtonId::LeftButton:
                {
                    window.releaseMouse();
                    m_Simulation.Paint = false;
                } break;
                case Tempest::MouseButtonId::RightButton:
                {
                    m_Status &= ~MOVEMENT_ROTATION;
                } break;
				case Tempest::MouseButtonId::MiddleButton:
				{
					m_Status &= ~MOVEMENT_DRAG;
				} break;
                }
            } break;
            case Tempest::WindowEventType::MouseMoved:
            {
                /*
                auto calc_mouse_pos = Tempest::Vector2{ (float)wevent.MouseMoved.MouseX, (float)wevent.MouseMoved.MouseY };
                auto rel_mouse_pos = calc_mouse_pos - m_PreviousMousePosition;
                if(Tempest::Dot(rel_mouse_pos, rel_mouse_pos) < MouseInsensitivity)
                    break;
                mouse_pos = calc_mouse_pos;
                /*/
                mouse_pos = Tempest::Vector2{ (float)wevent.MouseMoved.MouseX, (float)wevent.MouseMoved.MouseY };
                //*/
                if(!m_Simulation.Paint || !(m_Status & STATUS_ACTIVE) || m_Interaction.Cursor == mouse_pos || (m_Status & MOVEMENT_DRAG))
                    break;
                Tempest::SampleData sample_data;
				auto geom_id = rt_scene->rayQuery(mouse_pos/window_size, &sample_data);
				if(geom_id != m_PlaneID)
					break;
                auto prev_pos = m_Interaction.Cursor;
				m_Interaction.Cursor = sample_data.TexCoord*m_ImageSize;
                m_Interaction.ForceDirection = m_Interaction.Cursor - prev_pos;
                m_Interaction.ForceMagnitude = Tempest::Length(m_Interaction.ForceDirection);
                if(m_Interaction.ForceMagnitude < 1e-3f)
                    break;

                RASTERIZER::RasterizeCapsule2(Tempest::Capsule2{ { prev_pos, m_Interaction.Cursor }, TouchRadius }, gen_tex_desc.Width, gen_tex_desc.Height,
                                                m_Interaction);
				prev_pos = m_Interaction.Cursor;
                rt_scene->repaint();
            } break;
            case Tempest::WindowEventType::Focus:
            {
                if(wevent.Enabled != 0)
                {
                    m_Status |= STATUS_ACTIVE;
                }
                else
                {
                    m_Simulation.Paint = false;
				    m_Status &= ~STATUS_MASK;
                }
            } break;
			case Tempest::WindowEventType::MouseScroll:
			{
				if(!(m_Status & STATUS_ACTIVE) || wevent.MouseMoved.MouseDeltaX == 0.0f)
					break;

				m_Offset.z = std::max(m_Offset.z - WheelVelocity*wevent.MouseMoved.MouseDeltaX, 0.0f);
				m_Status |= MOVEMENT_FLUSH;
			} break;
            case Tempest::WindowEventType::KeyPressed:
            {
                if(!(m_Status & STATUS_ACTIVE))
                    break;
                switch(wevent.Key)
                {
                case Tempest::KeyboardKey::Up:
                case Tempest::KeyboardKey::Key_W:
                {
                    m_Status |= MOVEMENT_FORWARD;
                } break;
                case Tempest::KeyboardKey::Left:
                case Tempest::KeyboardKey::Key_A:
                {
                    m_Status |= MOVEMENT_LEFT;
                } break;
                case Tempest::KeyboardKey::Right:
                case Tempest::KeyboardKey::Key_D:
                {
                    m_Status |= MOVEMENT_RIGHT;
                } break;
                case Tempest::KeyboardKey::Down:
                case Tempest::KeyboardKey::Key_S:
                {
                    m_Status |= MOVEMENT_BACKWARD;
                } break;
                case Tempest::KeyboardKey::Key_C:
                case Tempest::KeyboardKey::LCtrl:
                {
                    m_Status |= MOVEMENT_ROTATION;
                } break;
                case Tempest::KeyboardKey::Key_R:
                {
                    m_Status |= MOVEMENT_UP;
                } break;
                case Tempest::KeyboardKey::Key_F:
                {
                    m_Status |= MOVEMENT_DOWN;
                } break;
				case Tempest::KeyboardKey::Key_Q:
				{
					m_Status |= MOVEMENT_ZOOM_IN;
				} break;
                case Tempest::KeyboardKey::Key_E:
				{
					m_Status |= MOVEMENT_ZOOM_OUT;
				} break;
                default: break;
                }
            } break;
            case Tempest::WindowEventType::KeyReleased:
            {
                if(!(m_Status & STATUS_ACTIVE))
                    break;
                switch(wevent.Key)
                {
                case Tempest::KeyboardKey::Up:
                case Tempest::KeyboardKey::Key_W:
                {
                    m_Status &= ~MOVEMENT_FORWARD;
                } break;
                case Tempest::KeyboardKey::Left:
                case Tempest::KeyboardKey::Key_A:
                {
                    m_Status &= ~MOVEMENT_LEFT;
                } break;
                case Tempest::KeyboardKey::Right:
                case Tempest::KeyboardKey::Key_D:
                {
                    m_Status &= ~MOVEMENT_RIGHT;
                } break;
                case Tempest::KeyboardKey::Down:
                case Tempest::KeyboardKey::Key_S:
                {
                    m_Status &= ~MOVEMENT_BACKWARD;
                } break;
                case Tempest::KeyboardKey::Key_C:
                case Tempest::KeyboardKey::LCtrl:
                {
                    m_Status &= ~MOVEMENT_ROTATION;
                } break;
                case Tempest::KeyboardKey::Key_R:
                {
                    m_Status &= ~MOVEMENT_UP;
                } break;
                case Tempest::KeyboardKey::Key_F:
                {
                    m_Status &= ~MOVEMENT_DOWN;
                } break;
				case Tempest::KeyboardKey::Key_Q:
				{
					m_Status &= ~MOVEMENT_ZOOM_IN;
				} break;
                case Tempest::KeyboardKey::Key_E:
				{
					m_Status &= ~MOVEMENT_ZOOM_OUT;
				} break;
                case Tempest::KeyboardKey::Key_T:
                {
					rt_scene->setRenderMode(rt_scene->getRenderMode() != Tempest::RenderMode::DebugTangents ?
												Tempest::RenderMode::DebugTangents : 
												Tempest::RenderMode::Normal);
                } break;
                case Tempest::KeyboardKey::Key_N:
                {
					rt_scene->setRenderMode(rt_scene->getRenderMode() != Tempest::RenderMode::DebugNormals ?
												Tempest::RenderMode::DebugNormals : 
												Tempest::RenderMode::Normal);
                } break;
				case Tempest::KeyboardKey::Key_B:
                {
					rt_scene->setRenderMode(rt_scene->getRenderMode() != Tempest::RenderMode::DebugBinormals ?
												Tempest::RenderMode::DebugBinormals : 
												Tempest::RenderMode::Normal);
                } break;
				case Tempest::KeyboardKey::Key_L:
                {
                    rt_scene->setRenderMode(rt_scene->getRenderMode() != Tempest::RenderMode::DebugLighting ?
												Tempest::RenderMode::DebugLighting : 
												Tempest::RenderMode::Normal);
                } break;
                case Tempest::KeyboardKey::Key_U:
                {
                    if(m_Status & OPTION_DISABLE_RECORD)
                        break;

                    ignore_event = true;
                    TGE_ASSERT((m_RecordStream.is_open() && m_RecordStatus != RecordStatus::Idle) || m_RecordStatus == RecordStatus::Idle, "Invalid recording state");
                    if(m_RecordStream.is_open())
                    {
                        if(m_RecordStatus != RecordStatus::Record)
                        {
                            Tempest::Log(Tempest::LogLevel::Error, "Cannot start playing back while commands are still being recorded");
                            break;
                        }
                        else
                        {
                            Tempest::Log(Tempest::LogLevel::Info, "ENDED RECORDING EVENTS");
                            m_RecordStream.close();
                            m_RecordStatus = RecordStatus::Idle;
                        }
                    }
                    else
                    {
                        Tempest::Log(Tempest::LogLevel::Info, "STARTED RECORDING EVENTS");
                        m_RecordStart = cur_time;
                        m_RecordStream.open(RECORD_FILE, std::ios::binary|std::ios::out);
                        if(!m_RecordStream)
                        {
                            Tempest::Log(Tempest::LogLevel::Error, "Failed to open file for recording");
                            break;
                        }
                        m_RecordStatus = RecordStatus::Record;
                    }
                } break;
                case Tempest::KeyboardKey::PrintScreen:
				{
                    m_Status |= ENQUEUE_SCREENSHOT;					    
				} break;
                case Tempest::KeyboardKey::Key_P:
                {
                    if(m_Status & OPTION_DISABLE_RECORD)
                        break;

                    ignore_event = true;
                    TGE_ASSERT((m_RecordStream.is_open() && m_RecordStatus != RecordStatus::Idle) || m_RecordStatus == RecordStatus::Idle, "Invalid recording state");
                    if(m_RecordStream.is_open())
                    {
                        if(m_RecordStatus != RecordStatus::Playback)
                        {
                            Tempest::Log(Tempest::LogLevel::Error, "Cannot start recording while commands are still being played back");
                            break;
                        }
                        else
                        {
                            m_RecordStream.close();
                            m_RecordStatus = RecordStatus::Idle;
                        }
                    }
                    else
                    {
                        m_RecordStart = cur_time;
                        m_RecordStream.open(RECORD_FILE, std::ios::binary|std::ios::in);
                        if(!m_RecordStream)
                        {
                            Tempest::Log(Tempest::LogLevel::Error, "Failed to open file for playback");
                            break;
                        }

                        m_RecordStream.read(reinterpret_cast<char*>(&m_EventRecord), sizeof(m_EventRecord));
                        if(!m_RecordStream)
                        {
                            m_RecordStream.close();
                            break;
                        }
                        m_RecordStatus = RecordStatus::Playback;
                    }
                } break;
				case Tempest::KeyboardKey::Key_M:
				{
					m_RayTracingSystem->toggleFpsCounter();
				} break;
				default: break;
                }
            } break;
            }

            if(m_RecordStatus == RecordStatus::Record && !ignore_event)
            {
                m_EventRecord.Timestamp = cur_time - m_RecordStart;
                memcpy(&m_EventRecord.Event, &wevent, sizeof(wevent));
                m_RecordStream.write(reinterpret_cast<char*>(&m_EventRecord), sizeof(m_EventRecord));
            }
        }

    #ifdef ENABLE_PHYSICS
        m_Simulation.CurrentTime = cur_time;
        m_Simulation.Cursor = interaction.Cursor;
    #endif

        float elapsed_time = (float)(cur_time - m_PreviousTime);
        m_PreviousTime = cur_time;
        if(m_Status & MOVEMENT_MASK)
        {
			Tempest::Vector3 left, forward;
			Tempest::FastSinCos(m_Yaw, &left.z, &left.x);
			left.y = 0.0f;
			forward = { -left.z, 0.0f, left.x };

			auto rel_cursor = mouse_pos - m_PreviousMousePosition;

			if((m_Status & MOVEMENT_DRAG) && rel_cursor.x && rel_cursor.y)
				m_Base += DragVelocity*(rel_cursor.x*left + rel_cursor.y*forward);
            if(m_Status & MOVEMENT_FORWARD)
                m_Base -= forward*Velocity*elapsed_time;
            if(m_Status & MOVEMENT_BACKWARD)
                m_Base += forward*Velocity*elapsed_time;
            if(m_Status & MOVEMENT_RIGHT)
                m_Base -= left*Velocity*elapsed_time;
            if(m_Status & MOVEMENT_LEFT)
                m_Base += left*Velocity*elapsed_time;
            if(m_Status & MOVEMENT_UP)
                m_Base.y -= Velocity*elapsed_time;
            if(m_Status & MOVEMENT_DOWN)
                m_Base.y += Velocity*elapsed_time;
			if(m_Status & MOVEMENT_ZOOM_IN)
				m_Offset.z = std::max(m_Offset.z - Velocity*elapsed_time, 0.0f);
			if(m_Status & MOVEMENT_ZOOM_OUT)
				m_Offset.z += Velocity*elapsed_time;

            if((m_Status & MOVEMENT_ROTATION) && (rel_cursor.x || rel_cursor.y))
            {
				m_Yaw -= MouseSpeed*rel_cursor.x;
				m_Roll -= MouseSpeed*rel_cursor.y;

				m_Roll = Tempest::Clampf(m_Roll, 0.0f, Tempest::MathPi*0.5f);
            }

            Tempest::Matrix4 view;
			view.identity();
			view.translate(-m_Offset);
            view.rotateX(Tempest::MathPi*0.5f - m_Roll);
            view.rotateY(-m_Yaw);
			view.translate(m_Base);

            auto projection = Tempest::PerspectiveMatrix(40.0f, (float)window.getWidth()/window.getHeight(), 0.1f, 1000.0f);
            Tempest::Matrix4 view_proj = projection*view;
            m_ViewProjectionInverse = view_proj.inverse();
			m_Status &= ~MOVEMENT_FLUSH;
        }
            
        m_PreviousMousePosition = mouse_pos;
    }

    {
        CPU_EVENT_SCOPE("Simulation");
        CUDA_EVENT_SCOPE("Simulation");
       	
        if(m_TestRotationPeriod)
        {
            float sin_angle, cos_angle;
            float angle = Tempest::MathTau*Tempest::Modulo(cur_time, m_TestRotationPeriod)/m_TestRotationPeriod;
            Tempest::FastSinCos(angle, &sin_angle, &cos_angle);

            TestRotation test_rotation;
            test_rotation.Width = gen_tex_desc.Width;
            test_rotation.Height = gen_tex_desc.Height;
            test_rotation.Data = m_SpringBuffer;
            test_rotation.Rotation = MaximumStretch*Tempest::Vector2{ cos_angle, sin_angle };
            Tempest::EXECUTE_PARALLEL_FOR_LOOP_2D(rt_scene->getThreadId(), rt_scene->getThreadPool(), (uint32_t)gen_tex_desc.Width, (uint32_t)gen_tex_desc.Height, test_rotation);
        }
            
        Tempest::EXECUTE_PARALLEL_FOR_LOOP_2D(rt_scene->getThreadId(), rt_scene->getThreadPool(), (uint32_t)gen_tex_desc.Width, (uint32_t)gen_tex_desc.Height, m_Simulation);
    }

        
    {
        CPU_EVENT_SCOPE("RTRT Complete and Restart");
        CUDA_EVENT_SCOPE("RTRT Complete and Restart");

    #ifdef ENABLE_PHYSICS
        uint64_t steps = (m_Simulation.CurrentTime - m_Simulation.PreviousUpdate)/UpdateStep;
        m_Simulation.PreviousUpdate += steps*UpdateStep;
    #endif

        m_RayTracingSystem->completeFrameAndRestart(window.getWidth(), window.getHeight(), m_ViewProjectionInverse);
    }

    {
        CPU_EVENT_SCOPE("Blits and Gestures");
        rt_scene->repaint();

		if(m_Status & (OPTION_PLAYBACK_AND_RECORD|ENQUEUE_SCREENSHOT))
			backend.setFramebuffer(m_IntermediateFramebuffer);
		
        auto alive = m_RayTracingSystem->submitFrame();

		if(!alive)
			return false;

        if(m_Status & MOVEMENT_ROTATION)
		{
			auto& desc = m_RotateImage->getDescription();

		    auto rel_mouse = mouse_pos/window_size;
            Tempest::Vector4 transform_touch{ 1.0f, 1.0f, 0.0f, 0.0f };
            transform_touch.x = desc.Width/window_size.x;
            transform_touch.y = desc.Height/window_size.y;
		    transform_touch.z = rel_mouse.x - 0.5f*transform_touch.x;
            transform_touch.w = 1.0f - rel_mouse.y - 0.5f*transform_touch.y;

		    backend.setTextures(m_RotateBakedTable.get());

		    m_ImageResourceTable->setResource(m_TransformIndex, transform_touch);

		    backend.submitCommandBuffer(m_TouchCommandBuffer);
		}
        else if(m_Simulation.Paint)
        {
            auto& desc = m_TouchImage->getDescription();

		    auto rel_mouse = mouse_pos/window_size;
     	    Tempest::Vector4 transform_touch{ 1.0f, 1.0f, 0.0f, 0.0f };
            transform_touch.x = desc.Width/window_size.x;
            transform_touch.y = desc.Height/window_size.y;
		    transform_touch.z = rel_mouse.x;
            transform_touch.w = 1.0f - rel_mouse.y;

		    backend.setTextures(m_TouchBakedTable.get());

		    m_ImageResourceTable->setResource(m_TransformIndex, transform_touch);

		    backend.submitCommandBuffer(m_TouchCommandBuffer);
        }

		if(m_Status & (OPTION_PLAYBACK_AND_RECORD|ENQUEUE_SCREENSHOT))
		{
			backend.blitAttachmentToScreen(Tempest::AttachmentType::Color, 0, 0, 0, 0, 0, window.getWidth(), window.getHeight());
		
			backend.submitCommandBuffer(m_CopyCommandBuffer);
		}
    }

    {
        CPU_EVENT_SCOPE("Present");

        if(swap_buffers)
		    window.swapBuffers(VSYNC);
    }

    if(m_Status & ENQUEUE_SCREENSHOT)
    {
        std::stringstream ss;
		ss << "screenshot.png";
		uint32_t counter = 0;
		for(;;)
		{
			if(!Tempest::System::Exists(ss.str()))
				break;
			ss.str("");
			ss << "screenshot_" << counter++ << ".png";
		}

        uint32_t data_size = rt_tex_fmt.Width*rt_tex_fmt.Height*Tempest::DataFormatElementSize(rt_tex_fmt.Format);

        m_BackbufferStorage->extractLinearBuffer(0, data_size, m_BackbufferTexture->getData());

		Tempest::SaveImage(m_BackbufferTexture->getHeader(), m_BackbufferTexture->getData(), Tempest::Path(ss.str()));
		Tempest::Log(Tempest::LogLevel::Info, "Saved screenshot: ", ss.str());
        m_Status &= ~ENQUEUE_SCREENSHOT;
    }

    if(m_Status & OPTION_PLAYBACK_AND_RECORD)
    {
        uint32_t data_size = rt_tex_fmt.Width*rt_tex_fmt.Height*Tempest::DataFormatElementSize(rt_tex_fmt.Format);

		m_BackbufferStorage->extractLinearBuffer(0, data_size, m_BackbufferTexture->getData());

		m_VideoEncoder->submitFrame(*m_BackbufferTexture);
    }

    return true;
}
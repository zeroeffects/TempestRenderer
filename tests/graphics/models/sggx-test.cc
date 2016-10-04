#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/math/numerical-methods.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"

const float StandardDeviation = 0.1f;
const float SphereGap = 0.1f;
const float SphereRadius = 4.0f;
const size_t SphereCount = 9;
const uint32_t IntegratorSamples = 1024;
const uint32_t RandomSamples = 1024;
const uint32_t LowRandomSamples = 4;
const uint32_t LightSubDirections = 256;

struct ZhaoMicroflakeDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    float Constant0, Constant1;
    Tempest::Vector3 Axis = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
};

struct SGGXMicroflakeDebug: public Tempest::RTSpatiallyVaryingEmitter
{
    Tempest::Vector3 StandardDeviation;
    Tempest::Matrix3 GGXBasis;
};

Tempest::Spectrum ZhaoMicroflake(const Tempest::SampleData& sample_data)
{
    auto* material = static_cast<const ZhaoMicroflakeDebug*>(sample_data.Material);

    float dot_theta = Dot(material->Axis, sample_data.Normal);

    float value = Tempest::ZhaoMicroFlakeNDF(material->Constant0, material->Constant1, dot_theta);

    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(value));
}

float GGXMicroFacetNDF(const SGGXMicroflakeDebug* material, const Tempest::Vector3& norm)
{
	Tempest::Vector2 tangent_plane_angle = Tempest::Normalize(Tempest::Vector2{ norm.x, norm.y });

	float cos_norm_half = norm.z;
	float cos_tan_micro = tangent_plane_angle.x;
	float sin_tan_micro = tangent_plane_angle.y;

    auto& stddev = material->StandardDeviation;
    float cos_norm_half_2 = cos_norm_half*cos_norm_half;
    float cos_norm_half_4 = cos_norm_half_2*cos_norm_half_2;
    float tan_norm_half_2 = (1.0f - cos_norm_half_2)/cos_norm_half_2;

    TGE_CHECK(cos_tan_micro*cos_tan_micro + sin_tan_micro*sin_tan_micro > 0.99f || (cos_tan_micro == 0.0f && sin_tan_micro == 0.0f), "Bad sin cos");

    float ratio1 = cos_tan_micro/stddev.x;
    float ratio2 = sin_tan_micro/stddev.y;

    float aniso_term = (ratio1*ratio1 + ratio2*ratio2);

    float denom_factor = (1.0f + tan_norm_half_2*aniso_term);

    return 1.0f/(Tempest::MathPi*cos_norm_half_4*stddev.x*stddev.y*denom_factor*denom_factor);
}

Tempest::Spectrum SGGXMicroflake(const Tempest::SampleData& sample_data)
{
    auto sggx_material = static_cast<const SGGXMicroflakeDebug*>(sample_data.Material);
    float value = SGGXMicroFlakeNDF(sggx_material->StandardDeviation, sggx_material->GGXBasis, sample_data.Normal);
    return Tempest::RGBToSpectrum(Tempest::ColorCodeHSL4ToRGB(value));
}

float SeeligerLaw(const Tempest::SampleData& sample_data)
{
    float cos_inc = Dot(sample_data.Normal, sample_data.IncidentLight);
    float cos_out = Dot(sample_data.Normal, sample_data.OutgoingLight);

    return cos_inc/(cos_inc + cos_out);
}


TGE_TEST("Testing SGGX lobes")
{
    uint32_t image_width = 500;
    uint32_t image_height = 500;

    float span = SphereCount*0.5f*2.0f*(SphereRadius + SphereGap);

    //Tempest::Matrix4 view_proj = Tempest::OrthoMatrix(-span, span, -span, span, 0.1f, 1000.0f);
    Tempest::Matrix4 view_proj = Tempest::PerspectiveMatrix(70.0f, (float)image_width / image_height, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, 15.0f, 5.0f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    view_proj *= view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    std::unique_ptr<Tempest::RayTracerScene> rt_scene(new Tempest::RayTracerScene(image_width, image_height, view_proj_inv));

    rt_scene->setSamplesGlobalIllumination(64);
    rt_scene->setMaxRayDepth(1);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    ZhaoMicroflakeDebug zhao_microflake;
    zhao_microflake.EmitFunction = ZhaoMicroflake;
    float c0 = zhao_microflake.Constant0 = 2.0f*StandardDeviation*StandardDeviation;
    float c1 = zhao_microflake.Constant1 = 1.0f/(powf(2.0f*Tempest::MathPi, 1.5f)*StandardDeviation*erff(1.0f/(sqrtf(2.0f)*StandardDeviation)));

    zhao_microflake.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    zhao_microflake.setup();
    
    {
    Tempest::Vector3 sphere_pos{ target.x - (SphereRadius + 0.5f*SphereGap), target.y + SphereRadius, target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &zhao_microflake);
    }

    unsigned seed = 1;
    Tempest::Vector2 sggx_stddev_v2{ 0.2f + 0.8f*Tempest::FastFloatRand(seed), 0.2f + 0.8f*Tempest::FastFloatRand(seed) };

    Tempest::RTSGGXMicroFlakeMaterial sggx_render_material;
    auto sggx_stddev = sggx_render_material.SGGXStandardDeviation = { sggx_stddev_v2.x, sggx_stddev_v2.y, 1.0f };

    Tempest::SampleData sample_data;
    sample_data.OutgoingLight = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
    sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
    sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

    Tempest::RTSGGXSurface sggx_surface_render_material;
    sggx_surface_render_material.StandardDeviation = sggx_stddev_v2;
    sggx_surface_render_material.Diffuse = {};
    sggx_surface_render_material.Specular = Tempest::ToSpectrum(1.0f);
    sggx_surface_render_material.Depth = 1;
    sggx_surface_render_material.SampleCount = 256;

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();

    for(uint32_t i = 0; i < LowRandomSamples; ++i)
    {
        Tempest::Vector3 out_light = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Vector3 inc_light = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Vector3 rand_basis_norm = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        sample_data.Material = &sggx_render_material;

        rand_basis_norm.z = fabsf(rand_basis_norm.z);

        sample_data.IncidentLight = inc_light;
        sample_data.OutgoingLight = out_light;

        Tempest::Matrix3 sggx_basis;
        sggx_basis.makeBasis(rand_basis_norm);

        auto quat = Tempest::ToQuaternion(sggx_basis);
        sggx_surface_render_material.SGGXBasis = reinterpret_cast<Tempest::Vector4&>(quat);

        float dir_density = SGGXProjectedArea(sggx_stddev, sggx_basis, out_light);

        //Tempest::Spectrum radiance{};

        float inc_cos_theta = inc_light.z;

        float escape_density = SGGXProjectedArea(sggx_stddev, sggx_basis, sample_data.IncidentLight);

        float minus_escape_density = SGGXProjectedArea(sggx_stddev, sggx_basis, -sample_data.IncidentLight);

        TGE_CHECK(Tempest::ApproxEqual(escape_density, minus_escape_density, 1e-3f), "Bad assumptions");

        auto micro_norm = Normalize(out_light + inc_light);

        Tempest::Spectrum radiance = Tempest::SimpsonsCompositeRuleQuadratureIntegrator<Tempest::Spectrum>(0.0f, 1.0f, RandomSamples,
            [escape_density, &micro_norm, &sggx_stddev, &sggx_basis, inc_cos_theta, &out_light, dir_density](float t)
            {
                if(t == 0.0f)
                    return Tempest::Spectrum{};

                float log_step = -logf(t);
    
                float step_size = log_step/dir_density;

                float penetration_depth = step_size*out_light.z;

                if(penetration_depth < 0.0f)
                    return Tempest::Spectrum{};

                float escape_distance = penetration_depth / inc_cos_theta;

                //cur_extinction *= expf(-log_step);

                float escape_extinction = expf(-escape_density*escape_distance);

                return escape_extinction*SGGXMicroFlakeBRDF(escape_density, micro_norm, sggx_stddev, sggx_basis)*dir_density/dir_density;
            });

        auto regular = SGGXMicroFlakeBRDF(escape_density, micro_norm, sggx_stddev, sggx_basis);
        float contrib = 1.0f/(dir_density + escape_density*out_light.z/inc_cos_theta);

        Tempest::Spectrum expected_radiance = contrib*SGGXMicroFlakeBRDF(dir_density, micro_norm, sggx_stddev, sggx_basis);
        float rel_error = fabsf(Array(expected_radiance)[0] - Array(radiance)[0])/Array(expected_radiance)[0];

        sample_data.Material = &sggx_surface_render_material;
        sample_data.DirectionalDensity = dir_density;
        Tempest::Spectrum surface_radiance = Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data)/dir_density;

        rel_error = fabsf(Array(expected_radiance)[0] - Array(surface_radiance)[0])/Array(expected_radiance)[0];
        TGE_CHECK(rel_error < 0.2f, "Invalid computation");

        auto seeliger_int = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples,
                                [inc_light](const Tempest::Vector3& out_light)
                                {
                                    Tempest::SampleData sample_data;
                                    sample_data.OutgoingLight = out_light;
                                    sample_data.IncidentLight = inc_light;
                                    sample_data.Normal = { 0.0f, 0.0f, 1.0f };

                                    return SeeligerLaw(sample_data)/Tempest::MathPi;
                                });

        sggx_surface_render_material.Depth = 0;

        Tempest::Spectrum pseudo_volume_radiance = Tempest::Cpp::SGGXMicroFlakePseudoVolumeBRDF(sample_data);

        rel_error = fabsf(Array(expected_radiance)[0] - Array(pseudo_volume_radiance)[0])/Array(expected_radiance)[0];
        TGE_CHECK(rel_error < 0.2f, "Invalid computation");

        sggx_surface_render_material.Depth = 0;
        sggx_surface_render_material.SampleCount = 256;

        /*
        auto pdf = Tempest::StratifiedMonteCarloIntegratorHemisphere(RandomSamples,
            [&inc_light, &sggx_surface_render_material](const Tempest::Vector3& out_light)
            {
                Tempest::SampleData sample_data {};
                sample_data.Material = &sggx_surface_render_material;
                sample_data.OutgoingLight = out_light;
                sample_data.IncidentLight = inc_light;
                sample_data.Tangent = { 1.0f, 0.0f, 0.0f };
	            sample_data.Binormal = { 0.0f, 1.0f, 0.0f };
                sample_data.Normal = { 0.0f, 0.0f, 1.0f };
                sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeSurfaceDensity(sample_data);

                return Tempest::SGGXMicroFlakeSurfacePDF(sample_data);
            });

        TGE_CHECK(Tempest::ApproxEqual(pdf, 1.0f, 2e-1f), "Invalid PDF");
         */
        // Energy conservation
        if(out_light.z > 0.1f)
        {
            Tempest::Spectrum total_energy = Tempest::ParallelStratifiedMonteCarloIntegratorHemisphere<Tempest::Spectrum>(id, pool, RandomSamples, 8,
                                                [&out_light, dir_density, &sggx_surface_render_material](const Tempest::Vector3& dir)
                                                {
                                                    auto* surf = reinterpret_cast<const Tempest::RTSGGXSurface*>(&sggx_surface_render_material);
                                                    Tempest::SampleData sample_data;
                                                    sample_data.Material = surf;
                                                    sample_data.OutgoingLight = out_light;
                                                    sample_data.IncidentLight = dir;
                                                    sample_data.Tangent = { 1.0f, 0.0f, 0.0f };
	                                                sample_data.Binormal = { 0.0f, 1.0f, 0.0f };
                                                    sample_data.Normal = { 0.0f, 0.0f, 1.0f };
                                                    sample_data.DirectionalDensity = dir_density;

                                                    auto radiance = Tempest::Cpp::SGGXMicroFlakeSurfaceBRDF(sample_data);
                                                    return radiance;
                                                });
            TGE_CHECK(Array(total_energy)[0] <= 1.1f, "Not energy conserving");
        }
    }

    auto id_quat = Tempest::IdentityQuaternion();
    sggx_surface_render_material.SGGXBasis = reinterpret_cast<Tempest::Vector4&>(id_quat); 

    auto sggx_basis = Tempest::ToMatrix3(reinterpret_cast<Tempest::Quaternion&>(sggx_surface_render_material.SGGXBasis));

    uint32_t anisotropy = 0;

    for(uint32_t y = 0; y < LightSubDirections; ++y)
        for(uint32_t x = 0; x < LightSubDirections; ++x)
        {
            sggx_surface_render_material.StandardDeviation.x = 0.54f;
            sggx_surface_render_material.StandardDeviation.y = 0.75f;

            sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_surface_render_material.StandardDeviation.x, sggx_surface_render_material.StandardDeviation.y, 1.0f }, sggx_basis, sample_data.IncidentLight);
            Tempest::Spectrum r0 = Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);

            sggx_surface_render_material.StandardDeviation.x = 0.75f;
            sggx_surface_render_material.StandardDeviation.y = 0.54f;

            sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_surface_render_material.StandardDeviation.x, sggx_surface_render_material.StandardDeviation.y, 1.0f }, sggx_basis, sample_data.IncidentLight);
            Tempest::Spectrum r1 = Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);

            if(r0 != r1)
                ++anisotropy;
        }

    TGE_CHECK(anisotropy > LightSubDirections*LightSubDirections/4, "The BRDF is actually isotropic or has fixed anisotropy based on standard deviation ratios");

    float norm_value = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                                                                               [c0, c1](float t)
                                                                                               {
                                                                                                   float sin_theta, cos_theta;
                                                                                                   Tempest::FastSinCos(t, &sin_theta, &cos_theta);
                                                                                                   return fabsf(cos_theta)*Tempest::ZhaoMicroFlakeNDF(c0, c1, cos_theta)*sin_theta;
                                                                                               });
    
    sample_data.Material = &sggx_render_material;

    sggx_render_material.SGGXStandardDeviation = sggx_stddev = Tempest::ConvertZhaoMicroFlakeToSGGX(StandardDeviation);
    TGE_CHECK(Tempest::ApproxEqual(norm_value, sggx_stddev.x, 1e-3f), "Bad numerical integrator");

    SGGXMicroflakeDebug sggx_microflake;
    sggx_microflake.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    sggx_microflake.EmitFunction = SGGXMicroflake;
    sggx_microflake.StandardDeviation = sggx_stddev;
    //sggx_microflake.StandardDeviation.x = sggx_microflake.StandardDeviation.y = 2.0f;
    //sggx_microflake.StandardDeviation.z = 1.0f;
    sggx_microflake.GGXBasis.makeBasisTangent(zhao_microflake.Axis);
    sggx_microflake.setup();

    float zhao_x = Tempest::ZhaoMicroFlakeNDF(c0, c1, Tempest::Dot(sggx_microflake.GGXBasis.tangent(), Tempest::Vector3{ 1.0f, 0.0f, 0.0f }));
    float zhao_y = Tempest::ZhaoMicroFlakeNDF(c0, c1, Tempest::Dot(sggx_microflake.GGXBasis.tangent(), Tempest::Vector3{ 0.0f, 1.0f, 0.0f }));
    float zhao_z = Tempest::ZhaoMicroFlakeNDF(c0, c1, Tempest::Dot(sggx_microflake.GGXBasis.tangent(), Tempest::Vector3{ 0.0f, 0.0f, 1.0f }));

    float sggx_x = Tempest::SGGXMicroFlakeNDF(sggx_microflake.StandardDeviation, sggx_microflake.GGXBasis, Tempest::Vector3{ 1.0f, 0.0f, 0.0f });
    float sggx_y = Tempest::SGGXMicroFlakeNDF(sggx_microflake.StandardDeviation, sggx_microflake.GGXBasis, Tempest::Vector3{ 0.0f, 1.0f, 0.0f });
    float sggx_z = Tempest::SGGXMicroFlakeNDF(sggx_microflake.StandardDeviation, sggx_microflake.GGXBasis, Tempest::Vector3{ 0.0f, 0.0f, 1.0f });
    
    //TGE_CHECK(0.6f < sggx_x && sggx_x < 0.9f, "Invalid BRDF - doesn't match paper");
    //TGE_CHECK(0.6f < sggx_y && sggx_y < 0.9f, "Invalid BRDF - doesn't match paper");
    
    TGE_CHECK(Tempest::ApproxEqual(zhao_x, sggx_x, 3e-1f), "Invalid BRDF");
    TGE_CHECK(Tempest::ApproxEqual(zhao_y, sggx_y, 3e-1f), "Invalid BRDF");
    TGE_CHECK(Tempest::ApproxEqual(zhao_z, sggx_z, 3e-1f), "Invalid BRDF");

    float circle = Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, 2.0f*Tempest::MathPi, IntegratorSamples,
                            [] (float k) { return Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples, [](float t) { return sinf(t); }); });

    TGE_CHECK(Tempest::ApproxEqual(circle, 4.0f*Tempest::MathPi, 1e-1f), "Bad integrator");

    float distro = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                [&sggx_microflake](float t)
                                {
                                    float c, s;
                                    Tempest::FastSinCos(t, &s, &c);
                                    return Tempest::SGGXMicroFlakeNDF(sggx_microflake.StandardDeviation, sggx_microflake.GGXBasis, Tempest::Vector3{ 0.0f, s, c })*fabsf(c)*s; // jacobian?
                                });

    float distro_zhao = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                [c0, c1](float t)
                                {
                                    float c, s;
                                    Tempest::FastSinCos(t, &s, &c);
                                    return Tempest::ZhaoMicroFlakeNDF(c0, c1, c)*fabsf(c)*s; // jacobian?
                                });

    float distro_sggx_norm = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                [&sggx_microflake](float t)
                                {
                                    float c, s;
                                    Tempest::FastSinCos(t, &s, &c);
                                    return Tempest::SGGXMicroFlakeNDF(sggx_microflake.StandardDeviation, sggx_microflake.GGXBasis, Tempest::Vector3{ 0.0f, s, c })*s; // jacobian?
                                });

    float distro_zhao_norm = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                [c0, c1](float t)
                                {
                                    float c, s;
                                    Tempest::FastSinCos(t, &s, &c);
                                    return Tempest::ZhaoMicroFlakeNDF(c0, c1, c)*s; // jacobian?
                                });

    SGGXMicroflakeDebug ggx_microflake;
    ggx_microflake.Model = Tempest::IlluminationModel::SpatiallyVaryingEmissive;
    ggx_microflake.EmitFunction = SGGXMicroflake;
    ggx_microflake.StandardDeviation.x = sggx_stddev.x*10.0f;
	ggx_microflake.StandardDeviation.y = sggx_stddev.x;
    ggx_microflake.StandardDeviation.z = 1.0f;
    //sggx_microflake.StandardDeviation.x = sggx_microflake.StandardDeviation.y = 2.0f;
    //sggx_microflake.StandardDeviation.z = 1.0f;
    ggx_microflake.GGXBasis.identity();
    ggx_microflake.setup();

    float distro_ggx = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples*IntegratorSamples,
                                    [&ggx_microflake](const Tempest::Vector3& norm)
                                    {
										float c = norm.z;
                                        return GGXMicroFacetNDF(&ggx_microflake, norm)*fabsf(c);
                                    });

    float distro_ggx_from_sggx = Tempest::StratifiedMonteCarloIntegratorHemisphere(IntegratorSamples*IntegratorSamples,
                                    [&ggx_microflake](const Tempest::Vector3& norm)
                                    {
                                        float c = norm.z;
                                        return 2.0f*Tempest::SGGXMicroFlakeNDF(ggx_microflake.StandardDeviation, ggx_microflake.GGXBasis, norm)*(c > 0.0f ? c : 0.0f);
                                    });

    TGE_CHECK(Tempest::ApproxEqual(distro_ggx, 1.0f, 2e-2f) && Tempest::ApproxEqual(distro_ggx_from_sggx, 1.0f, 2e-2f), "Invalid surface GGX");
    TGE_CHECK(Tempest::ApproxEqual(distro_zhao, distro, 1e-3f), "Bad SGGX approximation of Zhao's BRDF");
    
    // Individual functions test
    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 basis;
        basis.makeBasis(dir);

        auto stddev2 = sggx_render_material.SGGXStandardDeviation*sggx_render_material.SGGXStandardDeviation;
        auto smatrix = basis.transformCovariance(stddev2);

        TGE_CHECK(Tempest::ApproxEqual(smatrix(0, 1), smatrix(1, 0)) &&
                   Tempest::ApproxEqual(smatrix(0, 2), smatrix(2, 0)) &&
                   Tempest::ApproxEqual(smatrix(2, 1), smatrix(1, 2)), "Invalid covariance matrix");

        auto wm = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        float proj_area = sqrtf(wm.x*wm.x*smatrix(0, 0) +
                                wm.y*wm.y*smatrix(1, 1) +
                                wm.z*wm.z*smatrix(2, 2) +
                                2.0f*(wm.x*wm.y*smatrix(0, 1) +
                                      wm.x*wm.z*smatrix(0, 2) +
                                      wm.y*wm.z*smatrix(1, 2)));

        float proj_area_func = Tempest::SGGXProjectedArea(sggx_render_material.SGGXStandardDeviation, basis, wm);
        TGE_CHECK(Tempest::ApproxEqual(proj_area_func, proj_area), "Invalid projected area function");

        float smat_proj_area = sqrtf(Tempest::Dot(wm, smatrix.transform(wm)));
        TGE_CHECK(Tempest::ApproxEqual(proj_area, smat_proj_area, 1e-3f), "According to the actual paper this relationship should also hold");

        const float detS = smatrix(0, 0)*smatrix(1, 1)*smatrix(2, 2) - smatrix(0, 0)*smatrix(1, 2)*smatrix(1, 2) - smatrix(1, 1)*smatrix(0, 2)*smatrix(0, 2) - smatrix(2, 2)*smatrix(0, 1)*smatrix(0, 1) + 2.0f*smatrix(0, 1)*smatrix(0, 2)*smatrix(1, 2);
        const float den = wm.x*wm.x*(smatrix(1, 1)*smatrix(2, 2)-smatrix(1, 2)*smatrix(1, 2)) + wm.y*wm.y*(smatrix(0, 0)*smatrix(2, 2)-smatrix(0, 2)*smatrix(0, 2)) + wm.z*wm.z*(smatrix(0, 0)*smatrix(1, 1)-smatrix(0, 1)*smatrix(0, 1))
                         + 2.0f*(wm.x*wm.y*(smatrix(0, 2)*smatrix(1, 2)-smatrix(2, 2)*smatrix(0, 1)) + wm.x*wm.z*(smatrix(0, 1)*smatrix(1, 2)-smatrix(1, 1)*smatrix(0, 2)) + wm.y*wm.z*(smatrix(0, 1)*smatrix(0, 2)-smatrix(0, 0)*smatrix(1, 2)));
        const float D = powf(fabsf(detS), 1.5f) / (Tempest::MathPi*den*den);

        // same stuff but with library functions
        auto adj_tempest = smatrix.adjugate();
        float det_tempest = smatrix.determinant();
        float den_tempest = Dot(wm, adj_tempest.transform(wm));

        const float D_tempest = powf(fabsf(det_tempest), 1.5f) / (Tempest::MathPi*den_tempest*den_tempest);
        TGE_CHECK(Tempest::ApproxEqual(den_tempest, den, 1e-3f), "Invalid denumerator");
        TGE_CHECK(Tempest::ApproxEqual(det_tempest, detS, 1e-3f), "Invalid determinant");
        TGE_CHECK(Tempest::ApproxEqual(D_tempest, D, 1e-3f), "Invalid matrix operation");

        float D_ndf = Tempest::SGGXMicroFlakeNDF(sggx_render_material.SGGXStandardDeviation, basis, wm);
        TGE_CHECK(Tempest::ApproxEqual(2.0f*D_ndf, D_tempest, 1e-3f), "Invalid NDF");

        auto inv_cov = basis.transformCovariance(1.0f/stddev2);
        float den_alt = Dot(wm, inv_cov.transform(wm));
        float det_alt = inv_cov.determinant();
        float D_alt = sqrtf(det_alt) / (Tempest::MathPi*den_alt*den_alt);
        TGE_CHECK(Tempest::ApproxEqual(D_alt, D_tempest, 1e-3f), "Invalid NDF");
    }

    // Importance sampling tests
    for(size_t i = 0; i < RandomSamples; ++i)
    {
        auto flake_dir = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        Tempest::Matrix3 flake_basis;
        flake_basis.makeBasis(flake_dir);

        Tempest::SampleData sample_data;
        sample_data.Material = &sggx_render_material;
        sample_data.OutgoingLight = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
        sample_data.Tangent =  flake_basis.tangent();
        sample_data.Binormal = flake_basis.binormal();
        sample_data.Normal =   flake_basis.normal();
        sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeDensity(sample_data);

        unsigned shared_seed = seed;

        Tempest::SGGXMicroFlakeSampleIncidentLight({ 0, 1, 0, 1 }, &sample_data, seed);

        Tempest::Matrix3 out_basis;
        out_basis.makeBasis(sample_data.OutgoingLight);

        auto uniform_sample = Tempest::UniformSampleSphere(Tempest::FastFloatRand(shared_seed), Tempest::FastFloatRand(shared_seed));

        Tempest::Matrix3 basis(sample_data.Tangent, sample_data.Binormal, sample_data.Normal);
        auto s_matrix = basis.transformCovariance(sggx_render_material.SGGXStandardDeviation*sggx_render_material.SGGXStandardDeviation);

        s_matrix = out_basis*s_matrix*out_basis.transpose();

        auto cholesky_decompose = s_matrix.choleskyDecomposition();

        auto s_matrix_back = cholesky_decompose*cholesky_decompose.transpose();

        auto visible_norm = Normalize(cholesky_decompose.transformRotationInverse(uniform_sample));
        visible_norm = out_basis.transform(visible_norm);
        auto inc_light = Reflect(sample_data.OutgoingLight, visible_norm);

        Tempest::Matrix3 alt_matrix = basis;
        alt_matrix.scale(sggx_render_material.SGGXStandardDeviation);

        alt_matrix = out_basis*alt_matrix;

        auto alt_s_matrix = alt_matrix*alt_matrix.transpose();

        TGE_CHECK(alt_s_matrix == s_matrix_back, "Obviously the same covariance is guaranteed");

        auto alt_visible_norm = Normalize(alt_matrix.transformRotationInverse(uniform_sample));
        auto diff_space = cholesky_decompose.inverse() * alt_matrix;
		auto alt_alt_vn = diff_space.transform(alt_visible_norm); 
		alt_alt_vn = out_basis.transform(alt_alt_vn);
		
		TGE_CHECK(Tempest::ApproxEqual(alt_alt_vn, visible_norm, 1e-5f), "Should be the same but we don't expect the same result of two wildly different matrices");

		alt_visible_norm = out_basis.transform(alt_visible_norm);

        auto alt_inc_light = Reflect(sample_data.OutgoingLight, visible_norm);

        // And here is where statistics are needed for validation
    }

	// BRDF and PDF properties
	for(size_t i = 0; i < RandomSamples; ++i)
    {
		sample_data.IncidentLight = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeDensity(sample_data);
        if(sample_data.DirectionalDensity == 0.0f)
            continue;

		float pdf = Tempest::SGGXMicroFlakePDF(sample_data);

		auto brdf = Tempest::Cpp::SGGXMicroFlakeBRDF(sample_data);

		TGE_CHECK(Tempest::ApproxEqual(pdf, Array(brdf)[0]), "PDF is not exactly according to BRDF in this particular case");
	}

    Tempest::RTMicroFlakeMaterial microflake_render_material;
    microflake_render_material.Model = Tempest::IlluminationModel::MicroFlake;
    microflake_render_material.StandardDeviation = StandardDeviation;
    microflake_render_material.setup();

    sample_data.Material = &sggx_render_material;
    sample_data.IncidentLight = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeDensity(sample_data);
    auto sggx_energy_conservation = Tempest::StratifiedMonteCarloIntegratorSphere<Tempest::Spectrum>(IntegratorSamples,
                                        [&sample_data](const Tempest::Vector3& dir)
                                        {
                                            sample_data.OutgoingLight = dir;
                                            return Tempest::Cpp::SGGXMicroFlakeBRDF(sample_data);
                                        });

    sample_data.Material = &microflake_render_material;
    sample_data.IncidentLight = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
    sample_data.DirectionalDensity = Tempest::MicroFlakeDensity(sample_data);
    auto zhao_energy_conservation = Tempest::StratifiedMonteCarloIntegratorSphere<Tempest::Spectrum>(IntegratorSamples,
                                        [&sample_data](const Tempest::Vector3& dir)
                                        {
                                            sample_data.OutgoingLight = dir;
                                            return Tempest::MicroFlakeTransmittance(sample_data);
                                        });

    TGE_CHECK(Tempest::ApproxEqual(Array(sggx_energy_conservation)[0], 1.0f, 1e-1f), "SGGX phase function doesn't conserve energy");
    TGE_CHECK(Tempest::ApproxEqual(Array(zhao_energy_conservation)[0], 1.0f, 1e-1f), "Zhao's phase function doesn't conserve energy");

#if 0
    //*
    sample_data.OutgoingLight = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
    float pdf_total = Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, 2.0f*Tempest::MathPi, IntegratorSamples*10,
                          [&sample_data](float phi) 
                          {
                              float sin_phi, cos_phi;
                              Tempest::FastSinCos(phi, &sin_phi, &cos_phi);
                              return Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                                                                                        [cos_phi, sin_phi, &sample_data](float t)
                                                                                        {
                                                                                            float sin_theta, cos_theta;
                                                                                            Tempest::FastSinCos(t, &sin_theta, &cos_theta);
                                                                      
                                                                                            sample_data.DirectionalDensity = Tempest::SGGXMicroFlakeDensity(sample_data);
                                                                                            sample_data.IncidentLight = Tempest::Vector3{cos_phi*sin_theta, sin_phi*sin_theta, cos_theta};

                                                                                            float pdf = Tempest::SGGXMicroFlakePDF(sample_data);

                                                                                            TGE_CHECK(pdf >= 0.0f, "Invalid PDF");

                                                                                            return pdf*sin_theta;
                                                                                        });
                          });
    /*/
    float pdf_total = 2.0f*Tempest::MathPi*Tempest::SimpsonsCompositeRuleQuadratureIntegrator(0.0f, Tempest::MathPi, IntegratorSamples,
                          [&sample_data](float t)
                          {
                              float sin_theta, cos_theta;
                              Tempest::FastSinCos(t, &sin_theta, &cos_theta);
                                                                      
                              sample_data.IncidentLight = Tempest::Vector3{sin_theta, 0.0f, cos_theta};

                              float pdf = Tempest::SGGXMicroFlakePDF(sample_data);

                              TGE_CHECK(pdf >= 0.0f, "Invalid PDF");

                              return pdf*sin_theta;
                          });
    //*/

    TGE_CHECK(Tempest::ApproxEqual(pdf_total, 1.0f, 1e-1f), "PDF should sum up to 1.0f otherwise integrators won't work correctly");
#endif

   
    for(size_t j = 0; j < 3; ++j)
    {
        auto fixed_inc_light = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        float zhao_density_estimated_start = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSamples,
                                                [&zhao_microflake, fixed_inc_light](const Tempest::Vector3& dir)
                                                {
                                                    return fabsf(Dot(dir, fixed_inc_light))*Tempest::ZhaoMicroFlakeNDF(zhao_microflake.Constant0, zhao_microflake.Constant1, dir.z); // Assume wf = (0, 0, 1)
                                                });

        for(size_t i = 0; i < RandomSamples; ++i)
        {
            Tempest::Matrix3 rot_mat;
            rot_mat.rotateTangentPlane(Tempest::Matrix2::rotation(2.0f*Tempest::MathPi*i/RandomSamples));
            auto new_inc_light = rot_mat.transform(fixed_inc_light);

            float zhao_density_estimated_new = Tempest::StratifiedMonteCarloIntegratorSphere(IntegratorSamples,
                                                [&zhao_microflake, fixed_inc_light](const Tempest::Vector3& dir)
                                                {
                                                    return fabsf(Dot(dir, fixed_inc_light))*Tempest::ZhaoMicroFlakeNDF(zhao_microflake.Constant0, zhao_microflake.Constant1, dir.z); // Assume wf = (0, 0, 1)
                                                });

            TGE_CHECK(Tempest::ApproxEqual(zhao_density_estimated_start, zhao_density_estimated_new, 1e-2f), "Zhao Micro flake model uses table built under unreasonable assumptions");
        }
    }

	float zhao_density_x = Tempest::StratifiedMonteCarloIntegratorSphere(256*IntegratorSamples,
            [&zhao_microflake](const Tempest::Vector3& dir)
            {
                return fabsf(dir.x)*Tempest::ZhaoMicroFlakeNDF(zhao_microflake.Constant0, zhao_microflake.Constant1, dir.x); // Assume wf = (0, 0, 1)
            });

	float sggx_density_x = Tempest::StratifiedMonteCarloIntegratorSphere(16*IntegratorSamples,
            [&sggx_render_material](const Tempest::Vector3& dir)
            {
                return fabsf(dir.x)*Tempest::SGGXMicroFlakeNDF(sggx_render_material.SGGXStandardDeviation, Tempest::Matrix3::identityMatrix(), dir); // Assume wf = (0, 0, 1)
            });

    // Comparison of results from Zhao's phase function
    for(size_t i = 0; i < LowRandomSamples; ++i)
    {
        auto inc_light = sample_data.IncidentLight = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        auto tangent = Tempest::UniformSampleSphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));

        Tempest::Matrix3 basis;
        basis.makeBasisTangent(tangent);

		sample_data.Tangent = basis.tangent();
		sample_data.Binormal = basis.binormal();
		sample_data.Normal = basis.normal();

        sample_data.Material = &microflake_render_material;
        float microflake_density = Tempest::MicroFlakeDensity(sample_data);

        float zhao_density_estimated = Tempest::StratifiedMonteCarloIntegratorSphere(256*IntegratorSamples,
            [&zhao_microflake, inc_light, tangent](const Tempest::Vector3& dir)
            {
                return fabsf(Dot(inc_light, dir))*Tempest::ZhaoMicroFlakeNDF(zhao_microflake.Constant0, zhao_microflake.Constant1, Dot(tangent, dir)); // Assume wf = (0, 0, 1)
            });

        TGE_CHECK(Tempest::ApproxEqual(microflake_density, zhao_density_estimated, 1e-2f), "Somewhat trashy quality of table");

        sample_data.Material = &sggx_render_material;
        float sggx_density = Tempest::SGGXMicroFlakeDensity(sample_data);

        float sggx_density_estimated = Tempest::StratifiedMonteCarloIntegratorSphere(16*IntegratorSamples,
            [&basis, &sggx_render_material, inc_light](const Tempest::Vector3& dir)
            {
                return fabsf(Dot(dir, inc_light))*Tempest::SGGXMicroFlakeNDF(sggx_render_material.SGGXStandardDeviation, basis, dir); // Assume wf = (0, 0, 1)
            });
        
		TGE_CHECK(Tempest::ApproxEqual(sggx_density, sggx_density_estimated, 1e-2f), "Invalid projected area computation");

        TGE_CHECK(Tempest::ApproxEqual(microflake_density, sggx_density, 1e-1f), "Not that good of a approximation");
    }

    {
    Tempest::Vector3 sphere_pos{ target.x + (SphereRadius + 0.5f*SphereGap), target.y + SphereRadius, target.z };
    
    Tempest::Sphere sphere_geom { sphere_pos, SphereRadius };
    rt_scene->addSphere(sphere_geom, &sggx_microflake);
    }

    rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();
	
    auto* frame_data = rt_scene->drawOnce();
    TGE_CHECK(frame_data->Backbuffer, "Invalid backbuffer");

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");

    auto* backbuffer = frame_data->Backbuffer.get();
    Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Tempest::Path("test.tga"));
    Tempest::DisplayImage(backbuffer->getHeader(), backbuffer->getData());
}
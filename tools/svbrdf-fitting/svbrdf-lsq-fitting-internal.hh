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

#ifndef _SGGX_BRDF_EVALUATOR_HH_
#define _SGGX_BRDF_EVALUATOR_HH_
typedef double FittingType;

// Bypasses broken initializers in CUDA
struct RTMicrofacetMaterialProxy
{
    Tempest::IlluminationModel  Model;
    Tempest::Spectrum			Specular;
    Tempest::Vector2			StandardDeviation;
    Tempest::Spectrum           Diffuse;
    Tempest::Vector2   	 		Fresnel;
    float					    Normalization;
    
    const void*					DiffuseMap;
    const void*					SpecularMap;
};

struct RTSGGXSurfaceProxy: public RTMicrofacetMaterialProxy
{
	uint32_t				Depth, // Only applicable for the pseudo volume approach
							SampleCount,
                            BasisMapWidth,
                            BasisMapHeight;
	Tempest::Vector4		SGGXBasis;
	const void*				BasisMap;
	const void*				StandardDeviationMap;
};

struct StepConstraints
{
    FittingType operator()(uint32_t parameter_count, const FittingType* input_parameters, const FittingType* step_size)
    {
        FittingType cur_step = 0.75f;

		for(uint32_t i = 0; i < 4; ++i)
			cur_step = MinD(cur_step, input_parameters[i]/step_size[i]);

        return cur_step;
    }
};

struct AcceptStepConstraints
{
    bool operator()(uint32_t parameter_count, const FittingType* input_parameters, const FittingType* step_size)
    {
		for(uint32_t i = 0; i < 4; ++i)
			if(input_parameters[i] - step_size[i] < 0.0f)
				return false;

        return true;
    }
};

struct AcceptStepSpecularConstraints
{
    bool operator()(uint32_t parameter_count, const FittingType* input_parameters, const FittingType* step_size)
    {
        for(uint32_t i = 0; i < 3; ++i)
            if(input_parameters[i] - step_size[i] < 0.0f)
                return false;

        return true;
    }
};

struct BRDFEvaluator
{
	const Tempest::Vector2* LightDirections;
	uint32_t			    LightCount;
	uint32_t				BounceCount;
	float					Fresnel;
	
	inline EXPORT_EVALUATOR float operator()(uint32_t worker_id, uint32_t idx, const FittingType* input_params)
	{
		uint32_t view_idx = idx / LightCount;
		uint32_t light_idx = idx % LightCount;
		
		OptimizationParameters parameters;

		for(uint32_t param_idx = 0, param_idx_end = TGE_FIXED_ARRAY_SIZE(parameters.ParametersArray);
			param_idx < param_idx_end; ++param_idx)
		{
			parameters.ParametersArray[param_idx] = static_cast<float>(input_params[param_idx]);
		}

        auto sggx_stddev_v2 = Tempest::Vector2Max(parameters.Parameters.StandardDeviation, 0.0f);

        Tempest::Vector2 sggx_stddev{ sggx_stddev_v2.x, sggx_stddev_v2.y };
		
		Tempest::Quaternion quat = Tempest::ToQuaternion(parameters.Parameters.Euler);

        RTSGGXSurfaceProxy sggx_render_material;
		sggx_render_material.SampleCount = 256;
        sggx_render_material.Depth = BounceCount;
        sggx_render_material.Diffuse = Tempest::ToSpectrum(Maxf(parameters.Parameters.Diffuse, 0.0f)); // input_params[6], input_params[7], input_params[8] };
        sggx_render_material.Specular = Tempest::ToSpectrum(Maxf(parameters.Parameters.Specular, 0.0f));
        sggx_render_material.SGGXBasis = quat.V4;
        sggx_render_material.StandardDeviation = sggx_stddev;
		sggx_render_material.SpecularMap = nullptr;
		sggx_render_material.DiffuseMap = nullptr;
		sggx_render_material.BasisMap = nullptr;
		sggx_render_material.StandardDeviationMap = nullptr;
		sggx_render_material.Fresnel = { Fresnel, 0.0f };

        #ifdef __CUDACC__
            using namespace Tempest::Cuda;
        #else
            using namespace Tempest::Cpp;
        #endif

        Tempest::SampleData sample_data{};
		sample_data.Material = reinterpret_cast<Tempest::RTMaterial*>(&sggx_render_material);
		sample_data.IncidentLight = Tempest::ParabolicToCartesianCoordinates(LightDirections[light_idx]);
		sample_data.OutgoingLight = Tempest::ParabolicToCartesianCoordinates(LightDirections[view_idx]);
        sample_data.Tangent =  Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
        sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
        sample_data.Normal =   Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
        sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_stddev.x, sggx_stddev.y, 1.0f }, Tempest::ToMatrix3(quat), sample_data.OutgoingLight);
        
        auto spec = BounceCount != ~0u ?
            SGGXMicroFlakePseudoVolumeBRDF(sample_data) :
            SGGXMicroFlakeSurfaceBRDF(sample_data);

        TGE_ASSERT(std::isfinite(Array(spec)[0]), "Invalid value");

		return Array(spec)[0];
	}
};

struct BRDFEvaluatorSpecular
{
    const Tempest::Vector2* LightDirections;
    uint32_t			    LightCount;
    uint32_t				BounceCount;
    float					Fresnel;

    inline EXPORT_EVALUATOR float operator()(uint32_t worker_id, uint32_t idx, const FittingType* input_params)
    {
        uint32_t view_idx = idx / LightCount;
        uint32_t light_idx = idx % LightCount;

        OptimizationParametersSpecular parameters;

        for(uint32_t param_idx = 0, param_idx_end = TGE_FIXED_ARRAY_SIZE(parameters.ParametersArray);
            param_idx < param_idx_end; ++param_idx)
        {
            parameters.ParametersArray[param_idx] = static_cast<float>(input_params[param_idx]);
        }

        auto sggx_stddev_v2 = Tempest::Vector2Max(parameters.Parameters.StandardDeviation, 0.0f);

        Tempest::Vector2 sggx_stddev{ sggx_stddev_v2.x, sggx_stddev_v2.y };

        Tempest::Quaternion quat = Tempest::ToQuaternion(parameters.Parameters.Euler);

        RTSGGXSurfaceProxy sggx_render_material;
        sggx_render_material.SampleCount = 256;
        sggx_render_material.Depth = BounceCount;
        sggx_render_material.Diffuse = {};
        sggx_render_material.Specular = Tempest::ToSpectrum(Maxf(parameters.Parameters.Specular, 0.0f));
        sggx_render_material.SGGXBasis = reinterpret_cast<Tempest::Vector4&>(quat);
        sggx_render_material.StandardDeviation = sggx_stddev;
        sggx_render_material.SpecularMap = nullptr;
        sggx_render_material.DiffuseMap = nullptr;
        sggx_render_material.BasisMap = nullptr;
        sggx_render_material.StandardDeviationMap = nullptr;
        sggx_render_material.Fresnel = { Fresnel, 0.0f };

        #ifdef __CUDACC__
            using namespace Tempest::Cuda;
        #else
            using namespace Tempest::Cpp;
        #endif

        Tempest::SampleData sample_data{};
        sample_data.Material = reinterpret_cast<Tempest::RTMaterial*>(&sggx_render_material);
        sample_data.IncidentLight = Tempest::ParabolicToCartesianCoordinates(LightDirections[light_idx]);
        sample_data.OutgoingLight = Tempest::ParabolicToCartesianCoordinates(LightDirections[view_idx]);
        sample_data.Tangent = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
        sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
        sample_data.Normal = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };
        sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_stddev.x, sggx_stddev.y, 1.0f }, Tempest::ToMatrix3(quat), sample_data.OutgoingLight);

        auto spec = BounceCount != ~0u ?
            SGGXMicroFlakePseudoVolumeBRDF(sample_data) :
            SGGXMicroFlakeSurfaceBRDF(sample_data);

        TGE_ASSERT(std::isfinite(Array(spec)[0]), "Invalid value");

        return Array(spec)[0];
    }
};

namespace Tempest
{
template<class TFloat> struct FitStatistics;
template<class TFloat> struct LevenbergMarquardtSettings;
}

void LeastSquaresFitSGGXCuda(const Tempest::Vector2* sampled_lights, uint32_t sampled_light_count, float* sampled_lv_lum_slice, const LeastSquaresFitOptions& opts,
                             const FittingType* parameters, uint32_t parameters_count, FittingType** opt_parameters, Tempest::FitStatistics<FittingType>* stats,
							 Tempest::LevenbergMarquardtSettings<FittingType>* settings);

#endif //  _SGGX_BRDF_EVALUATOR_HH_
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

#include "tempest/utils/parse-command-line.hh"
#include "tempest/image/btf.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/image/image.hh"
#include "tempest/graphics/ray-tracing/illumination-models.hh"

template<class TData>
struct ComputeValues
{
    TData SquaredSum,
          MaxValue;
};

void BTFGenericSliceExtractor(const Tempest::BTF* btf_cpu, uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, float** out_luminance_slice)
{
    BTFParallelExtractLuminanceSlice(btf_cpu, id, pool, btf_x, btf_y, out_luminance_slice);
}

void BTFGenericSliceExtractor(const Tempest::BTF* btf_cpu, uint32_t id, Tempest::ThreadPool& pool, uint32_t btf_x, uint32_t btf_y, Tempest::Vector3** out_spectrum_slice)
{
    BTFParallelExtractRGBSlice(btf_cpu, id, pool, btf_x, btf_y, out_spectrum_slice);
}

Tempest::Vector3 ComputeResidual(const Tempest::Vector3& btf_spec, const Tempest::Spectrum& sggx_spec)
{
    return btf_spec - Tempest::SpectrumToRGB(sggx_spec);
}

float ComputeResidual(float btf_spec, const Tempest::Spectrum& sggx_spec)
{
    return btf_spec - Tempest::RGBToLuminance(Tempest::SpectrumToRGB(sggx_spec));
}

enum class ErrorComputationMethod
{
    Pairwise,
    MaxView,
};

void SelectMaximum(const ComputeValues<float>& lhs, ComputeValues<float>* rhs)
{
    if(lhs.SquaredSum > rhs->SquaredSum)
    {
        rhs->SquaredSum = lhs.SquaredSum;
        rhs->MaxValue = lhs.MaxValue;
    }
}

template<class TData>
void SelectMaximum(const ComputeValues<TData>& lhs, ComputeValues<TData>* rhs)
{
    for(size_t idx = 0; idx < TGE_FIXED_ARRAY_SIZE(lhs.MaxValue.Components); ++idx)
    {
        if(Array(lhs.SquaredSum)[idx] > Array(rhs->SquaredSum)[idx])
        {
            Array(rhs->SquaredSum)[idx] = Array(lhs.SquaredSum)[idx];
            Array(rhs->MaxValue)[idx] = Array(lhs.MaxValue)[idx];
        }
    }
}

template<class TData>
void ComputeError(uint32_t id, Tempest::ThreadPool& pool, ErrorComputationMethod method, const Tempest::BTF* btf, uint32_t x_start, uint32_t x_end, uint32_t y_start, uint32_t y_end,
                  const Tempest::Texture* diffuse_tex, const Tempest::Texture* specular_tex, const Tempest::Texture* basis_tex, const Tempest::Texture* stddev_tex,
                  TData* lv_slice, bool print_values, TData* out_rmse_array, TData* out_psnr_array)
{
    auto thread_count = pool.getThreadCount();
    std::unique_ptr<ComputeValues<TData>[]> partial_compute(new ComputeValues<TData>[thread_count]);

    auto& stddev_hdr = stddev_tex->getHeader();

    bool two_comp = Tempest::DataFormatChannels(stddev_hdr.Format) == 2;

	auto btf_light_count = btf->LightCount,
		 btf_view_count = btf->LightCount;

    auto partial_compute_ptr = partial_compute.get();
    for(uint32_t y = y_start; y < y_end; ++y)
		for(uint32_t x = x_start; x < x_end; ++x)
		{
			BTFGenericSliceExtractor(btf, id, pool, x, y, &lv_slice);

			uint32_t sggx_x = x - x_start,
					 sggx_y = y - y_start;
            
            float fresnel = 1.0f;

            Tempest::RTSGGXSurface sggx_render_material;
            sggx_render_material.SampleCount = 256;
            sggx_render_material.Depth = 0; // BounceCount
            sggx_render_material.Diffuse = Tempest::RGBToSpectrum(diffuse_tex->fetchRGB(sggx_x, sggx_y));
            sggx_render_material.Specular = Tempest::RGBToSpectrum(specular_tex->fetchRGB(sggx_x, sggx_y));
            sggx_render_material.SGGXBasis = basis_tex->fetchRGBA(sggx_x, sggx_y);
            if(two_comp)
            {
                sggx_render_material.StandardDeviation = stddev_tex->fetchRG(sggx_x, sggx_y);
            }
            else
            {
                auto sggx_stddev = stddev_tex->fetchRGB(sggx_x, sggx_y);
                sggx_render_material.StandardDeviation = { sggx_stddev.x/sggx_stddev.z, sggx_stddev.y/sggx_stddev.z };
            }
            sggx_render_material.SpecularMap = nullptr;
            sggx_render_material.DiffuseMap = nullptr;
            sggx_render_material.BasisMap = nullptr;
            sggx_render_material.StandardDeviationMap = nullptr;
            sggx_render_material.Fresnel = { fresnel, 0.0f };

            memset(partial_compute_ptr, 0, thread_count*sizeof(ComputeValues<TData>));

            TData mse, max_value;

            switch(method)
            {
            default: TGE_ASSERT(false, "Unsupported method");
            case ErrorComputationMethod::Pairwise:
            {
                auto compute_sq_error_sum = Tempest::CreateParallelForLoop2D(btf_light_count, btf_view_count, 16,
                    [btf, lv_slice, btf_light_count, &sggx_render_material, partial_compute_ptr](uint32_t worker_id, uint32_t light_idx, uint32_t view_idx)
				    {
					    uint32_t lv_idx = view_idx*btf_light_count + light_idx;
					
					    auto btf_spec = lv_slice[lv_idx];

                        Tempest::SampleData sample_data{};
                        sample_data.Material = reinterpret_cast<Tempest::RTMaterial*>(&sggx_render_material);
                        sample_data.Tangent = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
                        sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
                        sample_data.Normal = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

                        sample_data.IncidentLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
                        sample_data.OutgoingLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);
                        Tempest::Quaternion quat;
                        quat.V4 = sggx_render_material.SGGXBasis;
                        sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_render_material.StandardDeviation.x, sggx_render_material.StandardDeviation.y, 1.0f }, Tempest::ToMatrix3(quat), sample_data.OutgoingLight);
					
					    auto sggx_spec = 
						    //BounceCount != ~0u ?
						    //SGGXMicroFlakePseudoVolumeBRDF(sample_data) :
						    Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);

                        auto& compute_values = partial_compute_ptr[worker_id];

                        compute_values.MaxValue = Tempest::GenericMax(btf_spec, compute_values.MaxValue);

					    auto residual = ComputeResidual(btf_spec, sggx_spec);
					    compute_values.SquaredSum += residual*residual;
				    });

                pool.enqueueTask(&compute_sq_error_sum);
                pool.waitAndHelp(id, &compute_sq_error_sum);

                auto sq_error_sum = partial_compute_ptr->SquaredSum;
                max_value = partial_compute_ptr->MaxValue;
                for(uint32_t idx = 1; idx < thread_count; ++idx)
                {
                    auto& compute_values = partial_compute_ptr[idx];
                    sq_error_sum += compute_values.SquaredSum;
                    max_value = Tempest::GenericMax(max_value, compute_values.MaxValue);
                }

			    mse = sq_error_sum/(float)(btf_light_count*btf_view_count);
            } break;
            case ErrorComputationMethod::MaxView:
            {
                auto compute_sq_error_sum = Tempest::CreateParallelForLoop(btf_view_count, 1,
                    [btf, lv_slice, btf_light_count, &sggx_render_material, partial_compute_ptr](uint32_t worker_id, uint32_t view_idx, uint32_t)
				    {
                        ComputeValues<TData> cur_value{};
                        for(uint32_t light_idx = 0; light_idx < btf_light_count; ++light_idx)
                        {
					        uint32_t lv_idx = view_idx*btf_light_count + light_idx;
					
					        auto btf_spec = lv_slice[lv_idx];

                            Tempest::SampleData sample_data{};
                            sample_data.Material = reinterpret_cast<Tempest::RTMaterial*>(&sggx_render_material);
                            sample_data.Tangent = Tempest::Vector3{ 1.0f, 0.0f, 0.0f };
                            sample_data.Binormal = Tempest::Vector3{ 0.0f, 1.0f, 0.0f };
                            sample_data.Normal = Tempest::Vector3{ 0.0f, 0.0f, 1.0f };

                            sample_data.IncidentLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[light_idx]);
                            sample_data.OutgoingLight = Tempest::ParabolicToCartesianCoordinates(btf->LightsParabolic[view_idx]);
                            Tempest::Quaternion quat;
                            quat.V4 = sggx_render_material.SGGXBasis;
                            sample_data.DirectionalDensity = Tempest::SGGXProjectedArea(Tempest::Vector3{ sggx_render_material.StandardDeviation.x, sggx_render_material.StandardDeviation.y, 1.0f }, Tempest::ToMatrix3(quat), sample_data.OutgoingLight);
					
					        auto sggx_spec = 
						        //BounceCount != ~0u ?
						        //SGGXMicroFlakePseudoVolumeBRDF(sample_data) :
						        Tempest::SGGXMicroFlakeSurfaceBRDF(sample_data);
                            
                            cur_value.MaxValue = Tempest::GenericMax(btf_spec, cur_value.MaxValue);

					        auto residual = ComputeResidual(btf_spec, sggx_spec);
					        cur_value.SquaredSum += residual*residual;
                        }

                        auto& compute_values = partial_compute_ptr[worker_id];
                        SelectMaximum(cur_value, &compute_values);
				    });

                pool.enqueueTask(&compute_sq_error_sum);
                pool.waitAndHelp(id, &compute_sq_error_sum);

                auto cur_value = *partial_compute_ptr;
                for(uint32_t idx = 1; idx < thread_count; ++idx)
                {
                    auto& compute_values = partial_compute_ptr[idx];
                    SelectMaximum(compute_values, &cur_value);
                }

			    mse = cur_value.SquaredSum/(float)btf_light_count;
                max_value = cur_value.MaxValue;
            } break;
            }

            auto rmse = Tempest::GenericSqrt(mse);
            auto psnr = 20.0f*Tempest::GenericLog10(max_value / rmse);

            if(print_values)
            {
                Tempest::Log(Tempest::LogLevel::Info, "BTF (", x, ", ", y, "); RMSE: ", rmse, "; Peak value: ", max_value, "; PSNR: ", psnr);
            }

            uint32_t err_idx = sggx_y*stddev_hdr.Width + sggx_x;

            out_rmse_array[err_idx] = rmse;
            out_psnr_array[err_idx] = psnr;
		}
}

int main(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("svbrdf-error-tool", true);
    parser.createOption('X', "x-btf-start", "Specify start location on the BTF along the X-axis", true, "0");
    parser.createOption('Y', "y-btf-start", "Specify start location on the BTF along the Y-axis", true, "0");
    parser.createOption('p', "print", "Print values while computing", false);
    parser.createOption('m', "method", "Method of computing the error (pair-wise, max-view)", true, "pair-wise");
    parser.createOption('o', "output", "Specify output prefix for the created textures", true);
    parser.createOption('c', "color", "Enable per channel RMSE and PSNR computation", false);

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    auto unassoc = parser.getUnassociatedCount();
    if(unassoc != 2)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid input parameters.\n"
                                                "USAGE:"
                                                "\tsvbrdf-error-tool [ <options> ] <btf-file> <svbrdf-prefix>");
        return EXIT_FAILURE;
    }

    auto btf_infile = parser.getUnassociatedArgument(0);
    Tempest::BTFPtr btf(Tempest::LoadBTF(Tempest::Path(btf_infile)));
    if(!btf)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load BTF: ", btf_infile);
        return EXIT_FAILURE;
    }

    auto sggx_prefix = parser.getUnassociatedArgument(1);
    Tempest::Path diffuse_texname(sggx_prefix + "_albedo.exr");
    std::unique_ptr<Tempest::Texture> diffuse_tex(Tempest::LoadImage(diffuse_texname));
    if(!diffuse_tex)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load albedo texture: ", diffuse_texname);
        return EXIT_FAILURE;
    }

    Tempest::Path specular_texname(sggx_prefix + "_specular.exr");
    std::unique_ptr<Tempest::Texture> specular_tex(Tempest::LoadImage(specular_texname));
    if(!specular_tex)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load specular texture: ", specular_texname);
        return EXIT_FAILURE;
    }

    Tempest::Path stddev_texname(sggx_prefix + "_sggx_scale.exr");
    std::unique_ptr<Tempest::Texture> stddev_tex(Tempest::LoadImage(stddev_texname));
    if(!stddev_tex)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load standard deviation texture: ", stddev_texname);
        return EXIT_FAILURE;
    }

    Tempest::Path basis_texname(sggx_prefix + "_sggx_basis.exr");
    std::unique_ptr<Tempest::Texture> basis_tex(Tempest::LoadImage(basis_texname));
    if(!basis_tex)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load basis texture: ", basis_texname);
        return EXIT_FAILURE;
    }

	auto& albedo_hdr = diffuse_tex->getHeader(),
		& specular_hdr = specular_tex->getHeader(),
		& stddev_hdr = stddev_tex->getHeader(),
		& basis_hdr = basis_tex->getHeader();

    Tempest::RTSGGXSurface sggx_mat;
	sggx_mat.Model = Tempest::IlluminationModel::SGGXSurface;
	sggx_mat.SGGXBasis = { 0.0f, 0.0f, 0.0f };
	sggx_mat.StandardDeviation = { 1.0f, 1.0f };
	sggx_mat.Diffuse = Tempest::ToSpectrum(1.0f);
	sggx_mat.DiffuseMap = diffuse_tex.get();
	sggx_mat.Specular = Tempest::ToSpectrum(1.0f);
	sggx_mat.SpecularMap = specular_tex.get();
    sggx_mat.BasisMapWidth = basis_hdr.Width;
    sggx_mat.BasisMapHeight = basis_hdr.Height;
	sggx_mat.BasisMap = basis_tex.get();
	sggx_mat.StandardDeviationMap = stddev_tex.get();
    sggx_mat.setup();

	auto x_start = parser.extract<uint32_t>("x-btf-start"),
         y_start = parser.extract<uint32_t>("y-btf-start");

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

	auto x_end = x_start + stddev_hdr.Width,
         y_end = y_start + stddev_hdr.Height;

	auto btf_light_count = btf->LightCount,
		 btf_view_count = btf->LightCount;
    


	Tempest::ThreadPool pool;
	auto id = pool.allocateThreadNumber();

    bool print_values = parser.isSet("print");

    uint32_t tex_area = stddev_hdr.Width*stddev_hdr.Height;

    ErrorComputationMethod method;

    auto method_str = parser.extractString("method");
    if(method_str == "pair-wise")
    {
        method = ErrorComputationMethod::Pairwise;
    }
    else if(method_str == "max-view")
    {
        method = ErrorComputationMethod::MaxView;
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "Unsupported method: ", method_str);
        return EXIT_FAILURE;
    }

    if(parser.isSet("color"))
    {
        std::unique_ptr<Tempest::Vector3[]> rmse_array(new Tempest::Vector3[tex_area]),
                                            psnr_array(new Tempest::Vector3[tex_area]);

        std::unique_ptr<Tempest::Vector3[]> rgb_slice(new Tempest::Vector3[btf_light_count*btf_view_count]);

        ComputeError(id, pool, method, btf.get(), x_start, x_end, y_start, y_end,
                     diffuse_tex.get(), specular_tex.get(), basis_tex.get(), stddev_tex.get(),
                     rgb_slice.get(), print_values, rmse_array.get(), psnr_array.get());
    
        Tempest::TextureDescription error_tex_desc;
        error_tex_desc.Width = stddev_hdr.Width;
        error_tex_desc.Height = stddev_hdr.Height;
        error_tex_desc.Format = Tempest::DataFormat::RGB32F;

        auto output_prefix = parser.isSet("output") ? parser.extractString("output") : sggx_prefix;
        Tempest::SaveImage(error_tex_desc, rmse_array.get(), Tempest::Path(output_prefix + "_color_rmse.exr"));
        Tempest::SaveImage(error_tex_desc, psnr_array.get(), Tempest::Path(output_prefix + "_color_psnr.exr"));
    }
    else
    {
        std::unique_ptr<float[]> rmse_array(new float[tex_area]),
                                 psnr_array(new float[tex_area]);

        std::unique_ptr<float[]> luminance_slice(new float[btf_light_count*btf_view_count]);

        ComputeError(id, pool, method, btf.get(), x_start, x_end, y_start, y_end,
                     diffuse_tex.get(), specular_tex.get(), basis_tex.get(), stddev_tex.get(),
                     luminance_slice.get(), print_values, rmse_array.get(), psnr_array.get());

        Tempest::TextureDescription error_tex_desc;
        error_tex_desc.Width = stddev_hdr.Width;
        error_tex_desc.Height = stddev_hdr.Height;
        error_tex_desc.Format = Tempest::DataFormat::R32F;

        auto output_prefix = parser.isSet("output") ? parser.extractString("output") : sggx_prefix;
        Tempest::SaveImage(error_tex_desc, rmse_array.get(), Tempest::Path(output_prefix + "_rmse.exr"));
        Tempest::SaveImage(error_tex_desc, psnr_array.get(), Tempest::Path(output_prefix + "_psnr.exr"));
    }
    
}
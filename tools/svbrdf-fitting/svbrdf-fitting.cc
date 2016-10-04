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
#include "tempest/utils/timer.hh"
#include "tempest/utils/threads.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/image/image.hh"
#include "tempest/math/sampling3.hh"
#include <atomic>

#include "svbrdf-fitting.hh"

struct MainWorkerParameters
{
    uint32_t                BTFStartX,
                            BTFStartY,
                            RangeX,
                            RangeY;

    bool                    Refit;

    Tempest::Vector2*       StandardDeviationMap;
    Tempest::Quaternion*    BasisMap;
    Tempest::Vector3*       SpecularMap;
    Tempest::Vector3*       DiffuseMap;
    float*                  RMSEMap;

    bool                    DisableCurveFit;
    LeastSquaresFitOptions  FitOptions;
    FitPipelineOptions      PipelineOptions;
};

struct PointPusherThread: public Tempest::Task
{
    Tempest::ThreadPool*  Pool;
    MainWorkerParameters* Parameters;
    SVBRDFFitPipeline*      Pipeline;
    Tempest::Point2*      Points;
    std::atomic<uint32_t>* SharedCounter;
    Tempest::TimeQuery    Timer;

    virtual void execute(uint32_t worker_id) override
    {
        for(uint32_t point_idx, range = Parameters->RangeX*Parameters->RangeY; (point_idx = SharedCounter->fetch_add(1)) < range;)
        {
            auto& point = Points[point_idx];

			SGGXParameters fit_parameters;

            auto* timer_ptr = (Parameters->PipelineOptions.Flags & PIPELINE_OPTION_PRINT_PER_PIXEL) ? &Timer : nullptr;
            Pipeline->cache(worker_id, *Pool, point.x, point.y, Parameters->PipelineOptions, timer_ptr);
			Pipeline->fit(worker_id, *Pool, point.x, point.y, Parameters->DisableCurveFit ? nullptr : &Parameters->FitOptions, Parameters->PipelineOptions, timer_ptr, &fit_parameters);

			Parameters->StandardDeviationMap[point_idx] = fit_parameters.StandardDeviation;
			Parameters->BasisMap[point_idx] = fit_parameters.Orientation;
			Parameters->DiffuseMap[point_idx] = fit_parameters.Diffuse;
			Parameters->SpecularMap[point_idx] = fit_parameters.Specular;
            Parameters->RMSEMap[point_idx] = fit_parameters.RMSE;
        }
    }
};

struct PusherThread: public Tempest::Task
{
    Tempest::ThreadPool*  Pool;
    MainWorkerParameters* Parameters;
    SVBRDFFitPipeline*      Pipeline;
    std::atomic<uint32_t>* SharedCounter;
    Tempest::TimeQuery    Timer;

    virtual void execute(uint32_t worker_id) override
    {
        for(uint32_t tex_xy_idx, range = Parameters->RangeX*Parameters->RangeY; (tex_xy_idx = SharedCounter->fetch_add(1)) < range;)
        {
            uint32_t btf_x = Parameters->BTFStartX + (tex_xy_idx % Parameters->RangeX);
            uint32_t btf_y = Parameters->BTFStartY + (tex_xy_idx / Parameters->RangeX);

			SGGXParameters fit_parameters;
            if(Parameters->Refit)
            {
                fit_parameters.StandardDeviation = Parameters->StandardDeviationMap[tex_xy_idx];
				fit_parameters.Orientation = Parameters->BasisMap[tex_xy_idx];
				fit_parameters.Diffuse = Parameters->DiffuseMap[tex_xy_idx];
				fit_parameters.Specular = Parameters->SpecularMap[tex_xy_idx];
            }

            auto* timer_ptr = (Parameters->PipelineOptions.Flags & PIPELINE_OPTION_PRINT_PER_PIXEL) ? &Timer : nullptr;
            Pipeline->cache(worker_id, *Pool, btf_x, btf_y, Parameters->PipelineOptions, timer_ptr);
			Pipeline->fit(worker_id, *Pool, btf_x, btf_y, Parameters->DisableCurveFit ? nullptr : &Parameters->FitOptions, Parameters->PipelineOptions, timer_ptr, &fit_parameters, Parameters->Refit);
                
			Parameters->StandardDeviationMap[tex_xy_idx] = fit_parameters.StandardDeviation;
			Parameters->BasisMap[tex_xy_idx] = fit_parameters.Orientation;
			Parameters->DiffuseMap[tex_xy_idx] = fit_parameters.Diffuse;
			Parameters->SpecularMap[tex_xy_idx] = fit_parameters.Specular;
            Parameters->RMSEMap[tex_xy_idx] = fit_parameters.RMSE;
        }
	}
};

struct NeighborhoodPusherThread: public Tempest::Task
{
    Tempest::ThreadPool*  Pool;
    MainWorkerParameters* Parameters;
    SVBRDFFitPipeline*      Pipeline;
    std::atomic<uint32_t>* SharedCounter;
    Tempest::TimeQuery    Timer;

    virtual void execute(uint32_t worker_id) override
    {
        auto btf_end_x = Parameters->BTFStartX + Parameters->RangeX,
             btf_end_y = Parameters->BTFStartY + Parameters->RangeY;

        for(uint32_t tex_xy_idx, range = Parameters->RangeX*Parameters->RangeY; (tex_xy_idx = SharedCounter->fetch_add(1)) < range;)
        {
            uint32_t btf_x = Parameters->BTFStartX + (tex_xy_idx % Parameters->RangeX);
            uint32_t btf_y = Parameters->BTFStartY + (tex_xy_idx / Parameters->RangeX);
            float best_rmse;
            {
            uint32_t tex_xy_idx = (btf_y - Parameters->BTFStartY)*Parameters->RangeX + btf_x - Parameters->BTFStartX;
            best_rmse = Parameters->RMSEMap[tex_xy_idx];

            Pipeline->cache(worker_id, *Pool, btf_x, btf_y, Parameters->PipelineOptions, &Timer);
            }

            for(uint32_t nb_y = btf_y > Parameters->BTFStartY ? btf_y - 1 : Parameters->BTFStartY,
                            nb_y_end = btf_y + 1 < btf_end_y ? btf_y + 2 : btf_end_y;
                nb_y < nb_y_end; ++nb_y)
                for(uint32_t nb_x = btf_x > Parameters->BTFStartX ? btf_x - 1 : Parameters->BTFStartX,
                            nb_x_end = btf_x + 1 < btf_end_x ? btf_x + 2 : btf_end_x;
                    nb_x < nb_x_end; ++nb_x)
                {
                        uint32_t tex_xy_idx = (nb_y - Parameters->BTFStartY)*Parameters->RangeX + nb_x - Parameters->BTFStartX;

                    SGGXParameters fit_parameters;
                    fit_parameters.StandardDeviation = Parameters->StandardDeviationMap[tex_xy_idx];
				    fit_parameters.Orientation = Parameters->BasisMap[tex_xy_idx];
				    fit_parameters.Diffuse = Parameters->DiffuseMap[tex_xy_idx];
				    fit_parameters.Specular = Parameters->SpecularMap[tex_xy_idx];
                    fit_parameters.RMSE = Parameters->RMSEMap[tex_xy_idx];

                    Pipeline->fit(worker_id, *Pool, nb_x, nb_y, Parameters->DisableCurveFit ? nullptr : &Parameters->FitOptions, Parameters->PipelineOptions, &Timer, &fit_parameters);

                    if(fit_parameters.RMSE < best_rmse)
                    {
                        Parameters->StandardDeviationMap[tex_xy_idx] = fit_parameters.StandardDeviation;
				        Parameters->BasisMap[tex_xy_idx] = fit_parameters.Orientation;
				        Parameters->DiffuseMap[tex_xy_idx] = fit_parameters.Diffuse;
				        Parameters->SpecularMap[tex_xy_idx] = fit_parameters.Specular;
                        Parameters->RMSEMap[tex_xy_idx] = fit_parameters.RMSE;
                        best_rmse = fit_parameters.RMSE;
                    }
                }
        }
    }
};

int main(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("svbrdffit", true);

    parser.createOption('o', "output", "Specify the prefix of the generated files.", true, "sggx");
    parser.createOption('s', "single", "Fit a single pixel from BTF.", false);
    parser.createOption('X', "x-coordinate", "Specify point location along X axis.", true, "0");
    parser.createOption('Y', "y-coordinate", "Specify point location along Y axis.", true, "0");
    parser.createOption('W', "width", "Specify the width of the processed area.", true, "0");
    parser.createOption('H', "height", "Specify the height of the processed area.", true, "0");
    parser.createOption('d', "display", "Display comparison between BTF and SGGX fit. Specify image size as NxM, e.g. 400x400", true);
    parser.createOption('D', "display-map", "Display map of particular samples across the surface. NOTE: Works only with -p enabled.", true);
    parser.createOption('F', "disable-curve-fitting", "Disables curve fitting and relies on PCA to determine optimal direciton", false);
    parser.createOption('n', "refractive-index", "Specify the refractive index (default to 1.5)", true, "1.5");
    parser.createOption('m', "multi-scattering", "Enable multi scattering model fitting", true);
    parser.createOption('L', "log-file", "Specify log file", true);
    parser.createOption('S', "downsample-fitting", "Specify number of light and view samples used for fitting", true);
    parser.createOption('c', "cuda", "Enable CUDA optimizations", false);
    parser.createOption('p', "points", "Specify comma-separated list of points (example: \"1:2, 3:4\")", true);
    parser.createOption('k', "kernel-size", "Specify pre-filtering kernel size", true);
    parser.createOption('K', "filtering-technique", "Specify pre-filtering technique (gaussian)", true, "gaussian");
    parser.createOption('B', "basis-extract", "Specify basis extraction strategy for the initial guess (pca-hemisphere, pca-plane-projection, photometric)", true, "pca-hemisphere");
    parser.createOption('N', "disable-direction-fitting", "Disable direction fitting", false);
    parser.createOption('M', "maximize-normal-projection", "Enable normal projection maximimization procedure", false);
    parser.createOption('f', "downsample-filtering", "Enable filtering when downsampling", false);
    parser.createOption('r', "refit", "Specify file prefix for refitting SV-BRDF", true);
    parser.createOption('j', "optimize-neighborhood", "Try to optimize based on nearby values", false);
    parser.createOption('w', "worker-count", "Specify number of main worker threads", true, "1");
    parser.createOption('T', "top-light-fit", "Perform fits only for top view", false);
    parser.createOption('a', "top-approximate-ndf", "Perform top approximation of the NDF", false);
    parser.createOption('q', "quiet", "Disable printing of per pixel fit information", false);
    parser.createOption('h', "help", "Print help message", false);

#ifndef NDEBUG
	parser.createOption('Z', "null-test", "Specify number of test steps to validate the fitting procedure against synthetic data", true);
#endif
    if(!parser.parse(argc, argv))
    {
        return EXIT_FAILURE;
    }

    if(parser.isSet("help"))
    {
        parser.printHelp(std::cout);
        return EXIT_SUCCESS;
    }

    std::unique_ptr<Tempest::LogFile> log;
    auto log_file = parser.extractString("log-file");
    if(!log_file.empty())
    {
        log = std::unique_ptr<Tempest::LogFile>(new Tempest::LogFile(log_file));
    }

    auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "input BTF file is not specified\n\n"
                                               "USAGE:\n"
                                               "\tsvbrdffit [ -osxywhdFnmLDc ] <input-file>");
        return EXIT_FAILURE;
    }
    else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "too many input files specified\n\n"
                                               "USAGE:\n"
                                               "\tsvbrdffit [ -osxywhdFnmLDc ] <input-file>");
        return EXIT_FAILURE;
    }

    auto input_filename = parser.getUnassociatedArgument(0);

    Tempest::TimeQuery timer;
    auto start = timer.time();

    Tempest::ThreadPool pool;
    auto id = pool.allocateThreadNumber();



    //float fresnel = Tempest::ComputeReflectionCoefficient(parser.extract<float>("refractive-index"));
    MainWorkerParameters main_params;
    main_params.FitOptions.Fresnel = parser.extract<float>("refractive-index");
    main_params.DisableCurveFit = parser.isSet("disable-curve-fitting");
    main_params.FitOptions.MultiScatteringBounceCount = parser.isSet("multi-scattering") ? parser.extract<uint32_t>("multi-scattering") : ~0;
    main_params.FitOptions.DownSampleLightView = parser.isSet("downsample-fitting") ? parser.extract<uint32_t>("downsample-fitting") : ~0u;
    if(parser.isSet("disable-direction-fitting"))
        main_params.FitOptions.Flags |= LSF_OPTION_DISABLE_DIRECTION_FITTING;
    if(parser.isSet("cuda"))
    {
        main_params.PipelineOptions.Flags |= PIPELINE_OPTION_CUDA;
        main_params.FitOptions.Flags |= LSF_OPTION_CUDA;
    }
    if(parser.isSet("downsample-filtering"))
        main_params.FitOptions.Flags |= LSF_OPTION_FILTER_DOWNSAMPLING;
    if(parser.isSet("top-light-fit"))
        main_params.FitOptions.Flags |= LSF_OPTION_FIT_TOP;
    if(parser.isSet("top-approximate-ndf"))
        main_params.PipelineOptions.Flags |= PIPELINE_OPTION_NDF_TOP;

    main_params.FitOptions.Flags |= LSF_OPTON_DIFFUSE;

    if(!parser.isSet("quiet"))
        main_params.PipelineOptions.Flags |= PIPELINE_OPTION_PRINT_PER_PIXEL;

    auto basis_extract = parser.extractString("basis-extract");

    if(basis_extract == "pca-hemisphere")
    {
        main_params.PipelineOptions.BasisExtract = BasisExtractStrategy::PCAHemisphere;
    }
    else if(basis_extract == "pca-plane-projection")
    {
        main_params.PipelineOptions.BasisExtract = BasisExtractStrategy::PCAPlaneProject;
    }
    else if(basis_extract == "photometric")
    {
        main_params.PipelineOptions.BasisExtract = BasisExtractStrategy::PhotometricNormals;
    }
    else
    {
        Tempest::Log(Tempest::LogLevel::Error, "unsupported basis extraction strategy: ", basis_extract);
        return EXIT_FAILURE;
    }

    if(parser.isSet("maximize-normal-projection"))
        main_params.PipelineOptions.Flags |= PIPELINE_OPTION_MAXIMIZE_NORMAL_PROJECTION;

    if(parser.isSet("kernel-size"))
    {
        main_params.PipelineOptions.KernelRadius = parser.extract<uint32_t>("kernel-size");
        
        auto filter_name = parser.extractString("filtering-technique");
        if(filter_name == "gaussian")
        {
            main_params.PipelineOptions.Filter = FilteringTechnique::Gaussian;
        }
        else
        {
            Tempest::Log(Tempest::LogLevel::Error, "unsupported filtering technique: ", filter_name);
            return EXIT_FAILURE;
        }
    }
    else
    {
        main_params.PipelineOptions.KernelRadius = 0;
        main_params.PipelineOptions.Filter = FilteringTechnique::None;
    }

    if(main_params.FitOptions.DownSampleLightView == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "you must specify greater than 0 downsampling sample count");
        return EXIT_FAILURE;
    }

#ifndef NDEBUG
	if(parser.isSet("null-test"))
	{
		unsigned seed = 1;

		uint32_t number_of_steps = parser.extract<uint32_t>("null-test");

		const uint32_t light_count = 256;

		Tempest::BTFPtr null_btf(Tempest::CreateDummyBTF(light_count));
		
		for(uint32_t step = 0; step < number_of_steps; ++step)
		{
            OptimizationParameters sggx_parameters;
		    sggx_parameters.Parameters.StandardDeviation = { Tempest::FastFloatRand(seed)*0.9f + 0.1f, Tempest::FastFloatRand(seed)*0.9f + 0.1f }; // Don't do ridiculously specular stuff because it doesn't work
            if(sggx_parameters.Parameters.StandardDeviation.y < sggx_parameters.Parameters.StandardDeviation.x)
                std::swap(sggx_parameters.Parameters.StandardDeviation.x, sggx_parameters.Parameters.StandardDeviation.y);

            Tempest::Log(Tempest::LogLevel::Info, "Generated SGGX standard deviation: ", sggx_parameters.Parameters.StandardDeviation.x, ", ", sggx_parameters.Parameters.StandardDeviation.y);

            sggx_parameters.Parameters.Specular = Tempest::FastFloatRand(seed)*0.9f + 0.1f;
            Tempest::Log(Tempest::LogLevel::Info, "Generated specular: ", sggx_parameters.Parameters.Specular);

            sggx_parameters.Parameters.Diffuse = Tempest::FastFloatRand(seed)*0.0f;
            Tempest::Log(Tempest::LogLevel::Info, "Generated albedo: ", sggx_parameters.Parameters.Diffuse);

		    auto dir = Tempest::UniformSampleHemisphere(Tempest::FastFloatRand(seed), Tempest::FastFloatRand(seed));
		    Tempest::Matrix3 basis;
		    basis.makeBasis(dir);

            Tempest::Log(Tempest::LogLevel::Info, "Generated tangent: ", basis.tangent());
            Tempest::Log(Tempest::LogLevel::Info, "Generated binormal: ", basis.binormal());
            Tempest::Log(Tempest::LogLevel::Info, "Generated normal: ", basis.normal());

		    Tempest::Vector3 scaling, euler;
		    basis.decompose(&scaling, &euler);

            sggx_parameters.Parameters.Euler = euler;

        #ifdef DISABLE_CUDA
            SVBRDFFitPipeline pipeline(null_btf.get());
        #else
		    SVBRDFFitPipeline pipeline(null_btf.get(), nullptr);
        #endif    
            main_params.FitOptions.Flags &= ~LSF_OPTON_DIFFUSE;
            NullTestFillBTF(main_params.FitOptions, sggx_parameters, null_btf.get());

            // Zero optimization test
            {
            std::unique_ptr<float[]> luminance_slice(new float[light_count*light_count]);
            auto lv_lum_slice = luminance_slice.get();

            Tempest::BTFParallelExtractLuminanceSlice(null_btf.get(), id, pool, 0, 0, &lv_lum_slice);

            OptimizationParameters opt_parameters;

            float rmse;
            LeastSquaresFitSGGX(id, pool, null_btf.get(), lv_lum_slice, main_params.FitOptions, sggx_parameters, &opt_parameters, &rmse);

            for(uint32_t param_idx = 0; param_idx < TGE_FIXED_ARRAY_SIZE(sggx_parameters.ParametersArray); ++param_idx)
            {
                TGE_ASSERT(sggx_parameters.ParametersArray[param_idx] == opt_parameters.ParametersArray[param_idx], "No change should occur - it is already optimal");
            }
            }

			SGGXParameters fit_parameters;
			pipeline.fit(id, pool, 0, 0, main_params.DisableCurveFit ? nullptr : &main_params.FitOptions, main_params.PipelineOptions, &timer, &fit_parameters);

            TGE_ASSERT(main_params.FitOptions.Fresnel == 1.0f, "Fresnel fitting is awful");

			TGE_ASSERT(sggx_parameters.Parameters.Diffuse == Array(fit_parameters.Diffuse)[0], "Failed optimization");
			TGE_ASSERT(Tempest::ApproxEqual(sggx_parameters.Parameters.Specular, Array(fit_parameters.Specular)[0], 1e-2f), "Failed optimization");
			auto quat = Tempest::ToQuaternion(sggx_parameters.Parameters.Euler);

            auto fit_basis = Tempest::ToMatrix3(quat);

            if(!Tempest::ApproxEqual(sggx_parameters.Parameters.StandardDeviation.x, sggx_parameters.Parameters.StandardDeviation.y, 1e-2f))
            {
                TGE_ASSERT(fabsf(Tempest::Dot(basis.tangent(), fit_basis.tangent())) > 0.9f, "Failed optimization");
                TGE_ASSERT(fabsf(Tempest::Dot(basis.binormal(), fit_basis.binormal())) > 0.9f, "Failed optimization");
                TGE_ASSERT(fabsf(Tempest::Dot(basis.normal(), fit_basis.normal())) > 0.9f, "Failed optimization");
                Tempest::Vector2& sggx_stddev = fit_parameters.StandardDeviation;
                if(!Tempest::ApproxEqual(sggx_parameters.Parameters.StandardDeviation, sggx_stddev, 2e-2f))
                {
                    TGE_ASSERT(!SymmetryTest(null_btf.get(), sggx_parameters), "Should not be symmetric by definition");
                }

                // Currently it gets stuck under these circumstances
                if(basis.normal().z < 0.5f)
                {
                    TGE_ASSERT(Tempest::ApproxEqual(sggx_parameters.Parameters.StandardDeviation, sggx_stddev, 2e-2f), "Failed optimization");
                }
            }
		}

		return EXIT_SUCCESS;
	}
#endif

    Tempest::BTFPtr btf(LoadBTF(Tempest::Path(input_filename)));
    auto btf_ptr = btf.get();
    if(!btf)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to load BTF file used in the fitting process: ", input_filename);
        return EXIT_FAILURE;
    }

    auto btf_height = btf->Height,
         btf_width = btf->Width;

    bool single = parser.isSet("single");
    std::vector<Tempest::Point2> points;
    if(parser.isSet("points"))
    {
        auto points_str = parser.extractString("points");
        bool status = Tempest::ParseCommaSeparatedPoints(points_str.c_str(), &points);
        if(!status)
        {
            return EXIT_FAILURE;
        }
        for(auto& point : points)
        {
            if(point.x >= btf_width || point.y >= btf_height)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Point (", point.x, ", ", point.y, ") is out of range");
                return EXIT_FAILURE;
            }
        }
        main_params.RangeX = static_cast<uint32_t>(points.size());
        main_params.RangeY = 1;
    }
    else
    {
        main_params.BTFStartX = parser.extract<uint32_t>("x-coordinate");
        main_params.BTFStartY = parser.extract<uint32_t>("y-coordinate");
        main_params.RangeX = parser.extract<uint32_t>("width");
        main_params.RangeY = parser.extract<uint32_t>("height");
        
        uint32_t btf_end_x = main_params.BTFStartX + main_params.RangeX,
                 btf_end_y = main_params.BTFStartY + main_params.RangeY;

        if(btf_end_x == 0 || main_params.RangeX == 0)
        {
            btf_end_x = btf_width;
            main_params.RangeX = btf_width - main_params.BTFStartX;
        }

        if(btf_end_y == 0 || main_params.RangeY == 0)
        {
            btf_end_y = btf_height;
            main_params.RangeY = btf_height - main_params.BTFStartY;
        }

        if(single)
        {
            main_params.RangeX = main_params.RangeY = 1;
            btf_end_x = main_params.BTFStartX + 1;
            btf_end_y = main_params.BTFStartY + 1;
        }

        if(main_params.BTFStartX > btf_width || btf_end_x > btf_width ||
           main_params.BTFStartY > btf_height || btf_end_y > btf_height)
        {
            Tempest::Log(Tempest::LogLevel::Error, "the specified BTF coordinates are out-of-bounds: (", main_params.BTFStartX, "--", btf_end_x, "), (", main_params.BTFStartY, "--", btf_end_y, ")");
            return EXIT_FAILURE;
        }
    }

    uint32_t tex_area = main_params.RangeX*main_params.RangeY;
    std::unique_ptr<Tempest::Vector2[]> result_sggx_scale_map;
    std::unique_ptr<Tempest::Quaternion[]> result_sggx_rotation_map;
    std::unique_ptr<Tempest::Vector3[]> result_specular_map;
    std::unique_ptr<Tempest::Vector3[]> result_albedo_map;
    std::unique_ptr<float[]> result_rmse_map(new float[tex_area]);

    main_params.Refit = parser.isSet("refit"); 
    if(main_params.Refit)
    {
        auto refit_prefix = parser.extractString("refit");

        Tempest::Path scale_map_filename(refit_prefix + "_sggx_scale.exr");
        auto scale_map = Tempest::TexturePtr(Tempest::LoadImage(scale_map_filename));
        if(!scale_map)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load standard deviation map for refitting: ", scale_map_filename);
            return EXIT_FAILURE;
        }

        Tempest::Path albedo_map_filename(refit_prefix + "_albedo.exr");
        auto albedo_map = Tempest::TexturePtr(Tempest::LoadImage(albedo_map_filename));
        if(!albedo_map)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load diffuse map for refitting: ", albedo_map_filename);
            return EXIT_FAILURE;
        }
        
        Tempest::Path specular_map_filename(refit_prefix + "_specular.exr");
        auto specular_map = Tempest::TexturePtr(Tempest::LoadImage(specular_map_filename));
        if(!specular_map)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load specular map for refitting: ", specular_map_filename);
            return EXIT_FAILURE;
        }

        Tempest::Path basis_map_filename(refit_prefix + "_sggx_basis.exr");
        auto basis_map = Tempest::TexturePtr(Tempest::LoadImage(basis_map_filename));
        if(!basis_map)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load basis map for refitting: ", basis_map_filename);
            return EXIT_FAILURE;
        }

        auto& scale_hdr = scale_map->getHeader();
        auto& basis_hdr = basis_map->getHeader();
        auto& specular_hdr = specular_map->getHeader();
        auto& albedo_hdr = albedo_map->getHeader();

        if(scale_hdr.Width != main_params.RangeX ||
           basis_hdr.Width != main_params.RangeX ||
           specular_hdr.Width != main_params.RangeX ||
           albedo_hdr.Width != main_params.RangeX ||
           scale_hdr.Height != main_params.RangeY ||
           basis_hdr.Height != main_params.RangeY ||
           specular_hdr.Height != main_params.RangeY ||
           albedo_hdr.Height != main_params.RangeY)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Incompatible sizes between refit texture and specified area: (", main_params.RangeX, ", ", main_params.RangeY, ") compared to (", scale_hdr.Width, ", ", scale_hdr.Height, ")");
            return EXIT_FAILURE;
        }

        result_sggx_scale_map = decltype(result_sggx_scale_map)(reinterpret_cast<Tempest::Vector2*>(scale_map->release()));
        result_sggx_rotation_map = decltype(result_sggx_rotation_map)(reinterpret_cast<Tempest::Quaternion*>(basis_map->release()));
        result_specular_map = decltype(result_specular_map)(reinterpret_cast<Tempest::Vector3*>(specular_map->release()));
        result_albedo_map = decltype(result_albedo_map)(reinterpret_cast<Tempest::Vector3*>(albedo_map->release()));
    }
    else
    {
        result_sggx_scale_map = decltype(result_sggx_scale_map)(new Tempest::Vector2[tex_area]);
        result_sggx_rotation_map = decltype(result_sggx_rotation_map)(new Tempest::Quaternion[tex_area]);
        result_specular_map = decltype(result_specular_map)(new Tempest::Vector3[tex_area]);
        result_albedo_map = decltype(result_albedo_map)(new Tempest::Vector3[tex_area]);
    }

    main_params.StandardDeviationMap = result_sggx_scale_map.get();
    main_params.BasisMap = result_sggx_rotation_map.get();
    main_params.SpecularMap = result_specular_map.get();
    main_params.DiffuseMap = result_albedo_map.get();
    main_params.RMSEMap = result_rmse_map.get();

    struct PipelinesDeleter
    {
        size_t WorkerCount;
        void operator()(SVBRDFFitPipeline** pipeline)
        {
            for(size_t worker_idx = 0; worker_idx < WorkerCount; ++worker_idx)
            {
                delete pipeline[worker_idx];
            }
            delete pipeline;
        }
    };

    size_t worker_count = parser.extract<size_t>("worker-count");
    std::unique_ptr<SVBRDFFitPipeline*[], PipelinesDeleter> pipelines(new SVBRDFFitPipeline*[worker_count], PipelinesDeleter{ worker_count });

    if(worker_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "You must specify at least one main worker thread");
        return EXIT_FAILURE;
    }

#ifdef DISABLE_CUDA
    for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
    {
        pipelines[worker_idx] = new SVBRDFFitPipeline(btf_ptr);
    }
#else
    Tempest::BTFGPUPtr gpu_btf(main_params.PipelineOptions.Flags & PIPELINE_OPTION_CUDA ? Tempest::CreateGPUBTF(btf.get()) : nullptr);
    for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
    {
        pipelines[worker_idx] = new SVBRDFFitPipeline(btf_ptr, gpu_btf.get());
    }
#endif

    bool neighborhood = parser.isSet("optimize-neighborhood");

	if(!points.empty())
	{
		std::unique_ptr<PointPusherThread[]> pusher_threads(new PointPusherThread[worker_count]);

        std::atomic<uint32_t> counter;
        counter.store(0);

        for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
        {
            auto& pusher_thread = pusher_threads[worker_idx];
            pusher_thread.Pool = &pool;
            pusher_thread.Parameters = &main_params;
            pusher_thread.Pipeline = pipelines[worker_idx];
            pusher_thread.Points = &points.front();
            pusher_thread.SharedCounter = &counter;
            pool.enqueueTask(&pusher_thread);
        }

        for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
        {
            auto& pusher_thread = pusher_threads[worker_idx];
            pool.waitAndHelp(id, &pusher_thread);
        }
	}
	else
	{
        {
        std::unique_ptr<PusherThread[]> pusher_threads(new PusherThread[worker_count]);

        std::atomic<uint32_t> counter;
        counter.store(0);

        for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
        {
            auto& pusher_thread = pusher_threads[worker_idx];
            pusher_thread.Pool = &pool;
            pusher_thread.Parameters = &main_params;
            pusher_thread.Pipeline = pipelines[worker_idx];
            pusher_thread.SharedCounter = &counter;
            pool.enqueueTask(&pusher_thread);
        }

        for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
        {
            auto& pusher_thread = pusher_threads[worker_idx];
            pool.waitAndHelp(id, &pusher_thread);
        }
        }


        if(neighborhood)
        {
            std::unique_ptr<NeighborhoodPusherThread[]> pusher_threads(new NeighborhoodPusherThread[worker_count]);

            std::atomic<uint32_t> counter;
            counter.store(0);

            for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
            {
                auto& pusher_thread = pusher_threads[worker_idx];
                pusher_thread.Pool = &pool;
                pusher_thread.Parameters = &main_params;
                pusher_thread.Pipeline = pipelines[worker_idx];
                pusher_thread.SharedCounter = &counter;
                pool.enqueueTask(&pusher_thread);
            }

            for(size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx)
            {
                auto& pusher_thread = pusher_threads[worker_idx];
                pool.waitAndHelp(id, &pusher_thread);
            }
        }
    }
    auto elapsed_time = timer.time() - start;

    std::stringstream ss_out;
    ss_out << parser.extractString("output");

    if(!points.empty())
    {
        ss_out << "_points_";
    }
    else
    {
        if(main_params.BTFStartX != 0 || main_params.BTFStartY != 0 ||
           main_params.BTFStartX + main_params.RangeX != btf_width || main_params.BTFStartY + main_params.RangeY != btf_height)
        {
            ss_out << "_" << main_params.BTFStartX << "_" << main_params.BTFStartY << "_" << main_params.BTFStartX + main_params.RangeX << "_" << main_params.BTFStartY + main_params.RangeY << "_"; 
        }
    }

    auto out_prefix = ss_out.str();
    std::string albedo_texname = out_prefix + "_albedo.exr";

    Tempest::TextureDescription tex_desc;
    tex_desc.Width = main_params.RangeX;
    tex_desc.Height = main_params.RangeY;
    tex_desc.Format = Tempest::DataFormat::RGB32F;
    Tempest::SaveImage(tex_desc, result_albedo_map.get(), Tempest::Path(albedo_texname));

    std::string specular_texname = out_prefix + "_specular.exr";
    Tempest::SaveImage(tex_desc, result_specular_map.get(), Tempest::Path(specular_texname));

    tex_desc.Format = Tempest::DataFormat::RG32F;
    std::string sggx_texname = out_prefix + "_sggx_scale.exr";
    Tempest::SaveImage(tex_desc, result_sggx_scale_map.get(), Tempest::Path(sggx_texname));

    tex_desc.Format = Tempest::DataFormat::RGBA32F;
    std::string rot_texname = out_prefix + "_sggx_basis.exr";
    Tempest::SaveImage(tex_desc, result_sggx_rotation_map.get(), Tempest::Path(rot_texname));

    tex_desc.Format = Tempest::DataFormat::R32F;
    std::string rmse_texname = out_prefix + "_rmse.exr";
    Tempest::SaveImage(tex_desc, result_rmse_map.get(), Tempest::Path(rmse_texname));

    Tempest::Log(Tempest::LogLevel::Info, "Time needed to fit: ", elapsed_time, "us");

    auto display = parser.extractString("display");
    auto display_map = parser.extractString("display-map");

    if(!display.empty() && !display_map.empty())
    {
        Tempest::Log(Tempest::LogLevel::Error, "displaying single distributions cannot be paired with displaying distribution map");
        return EXIT_FAILURE;
    }
    else if(!display.empty() && single)
    {
        if(!single)
        {
            Tempest::Log(Tempest::LogLevel::Error, "displaying fits works only for single pixel");
            return EXIT_FAILURE;
        }

        uint32_t image_width, image_height;
        bool status = Tempest::ParseResolution(display, &image_width, &image_height);
        if(!status)
            EXIT_FAILURE;

        auto quat = result_sggx_rotation_map[0];

        DisplayDistributions(image_width, image_height,
                             result_sggx_scale_map[0], result_albedo_map[0], result_specular_map[0], reinterpret_cast<Tempest::Vector4&>(quat),
                             btf_ptr, pipelines[0]->getLastLuminanceSlice(), main_params.BTFStartX, main_params.BTFStartY, *pipelines[0]->getLastNDFTexture());
    }
    else if(!display_map.empty())
    {
        if(points.empty())
        {
            Tempest::Log(Tempest::LogLevel::Error, "displaying distributions works only with explicitly specified points");
            return EXIT_FAILURE;
        }

        uint32_t image_width, image_height;
        bool status = Tempest::ParseResolution(display_map, &image_width, &image_height);
        if(!status)
            EXIT_FAILURE;

        DisplayDistributionMap(image_width, image_height, &points.front(),
                               result_sggx_scale_map.get(), result_specular_map.get(), result_sggx_rotation_map.get(), tex_area,
                               btf_ptr);
    }

    return EXIT_SUCCESS;
}

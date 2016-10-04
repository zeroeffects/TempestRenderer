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
#include "tempest/image/image.hh"
#include "tempest/math/quaternion.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/system.hh"
#include "tempest/image/image-process.hh"

#include <cstdlib>

#ifndef NDEBUG
#	include "tempest/graphics/ray-tracing/illumination-models.hh"
#endif

struct RegionStats
{
    uint32_t         Coverage = 0;
    Tempest::Vector3 Direction = {};
    bool             PerformFlip = false;
};

int main(int argc, char** argv)
{
	Tempest::CommandLineOptsParser parser("svbrdf-postprocess", true);
    parser.createOption('f', "flip-y", "Flip along Y axis", false);
    parser.createOption('o', "output", "Specify output file (default: modified_<name>)", true);
    parser.createOption('c', "cross-fade", "Specify cross fade interval", true, "0");
    parser.createOption('r', "report-image", "Specify image which would contain marked problem areas after postprocessing", false);
    parser.createOption('t', "tangent", "Specify preferred direction", true);
    parser.createOption('d', "disable-align-region", "Disable region alignment algorithm", false);
    parser.createOption('m', "save-marker-texture", "Specify file to save debug marker texture", true);
    parser.createOption('T', "save-region-tangent-texture", "Specify file to save debug region tangent texture", true);
    parser.createOption('a', "discontinuity-angle", "Specify minimum angle that is considered as discontinuity (must be lower than 180)", true);
    parser.createOption('P', "prefer-tangent", "Disable dominant direction reorientation and fully prefer tangent direction", false);
    parser.createOption('h', "help", "Print help message", false);

	auto status = parser.parse(argc, argv);

    if(!status)
    {
        return EXIT_FAILURE;
    }

    if(parser.isSet("help"))
    {
        parser.printHelp(std::cout);
        return EXIT_SUCCESS;
    }

	auto unassoc_count = parser.getUnassociatedCount();
    if(unassoc_count == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "svbrdf-postprocess: error: input SGGX distribution base name is not specified\n\n"
                                               "USAGE:\n"
                                               "\tsggx-postprocess [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }
	else if(unassoc_count > 1)
    {
        Tempest::Log(Tempest::LogLevel::Error, "svbrdf-postprocess: error: too many input names specified\n\n"
                                               "USAGE:\n"
                                               "\tsggx-postprocess [ <options> ] <input-file>");
        return EXIT_FAILURE;
    }

	auto in_prefix = parser.getUnassociatedArgument(0);

	Tempest::Path stddev_texname(in_prefix + "_sggx_scale.exr");
	std::unique_ptr<Tempest::Texture> in_sggx_stddev_map(Tempest::LoadImage(stddev_texname));
	if(!in_sggx_stddev_map)
	{
		Tempest::Log(Tempest::LogLevel::Error, "failed to load SGGX standard deviation file: ", stddev_texname);
		return EXIT_FAILURE;
	}

	auto& in_stddev_hdr = in_sggx_stddev_map->getHeader();
	if(Tempest::DataFormatChannels(in_stddev_hdr.Format) < 2)
	{
		Tempest::Log(Tempest::LogLevel::Error, "invalid standard deviation file format: ", stddev_texname);
		return EXIT_FAILURE;
	}
	
	auto in_stddev_vec = in_sggx_stddev_map->getData();

    std::unique_ptr<Tempest::Vector2[]> out_stddev_vec(new Tempest::Vector2[in_stddev_hdr.Width*in_stddev_hdr.Height]);

	Tempest::Path basis_texname(in_prefix + "_sggx_basis.exr");
	std::unique_ptr<Tempest::Texture> sggx_basis_map(Tempest::LoadImage(basis_texname));
	if(!sggx_basis_map)
	{
		Tempest::Log(Tempest::LogLevel::Error, "failed to load SGGX basis file: ", basis_texname);
		return EXIT_FAILURE;
	}

	auto& basis_hdr = sggx_basis_map->getHeader();
	if(Tempest::DataFormatChannels(basis_hdr.Format) != 4)
	{
		Tempest::Log(Tempest::LogLevel::Error, "invalid basis file format: ", basis_texname);
		return EXIT_FAILURE;
	}

	Tempest::Quaternion* basis_vec = reinterpret_cast<Tempest::Quaternion*>(sggx_basis_map->getData());

	if(in_stddev_hdr.Width != basis_hdr.Width ||
	   in_stddev_hdr.Height != basis_hdr.Height)
	{
		Tempest::Log(Tempest::LogLevel::Error, "unmatching texture sizes: ", stddev_texname, "; ", basis_texname);
		return EXIT_FAILURE;
	}

    bool flip_y = parser.isSet("flip-y");

    if(flip_y)
    {
        in_sggx_stddev_map->flipY();
        sggx_basis_map->flipY();
    }

    auto stride = Tempest::DataFormatElementSize(in_stddev_hdr.Format);

	for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
		for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
		{
			uint32_t pixel_idx = y*in_stddev_hdr.Width + x;

			auto& in_stddev = *reinterpret_cast<const Tempest::Vector2*>(in_stddev_vec + pixel_idx*stride);
          
            auto& out_stddev = out_stddev_vec[pixel_idx];
            out_stddev.x = in_stddev.x;
            out_stddev.y = in_stddev.y;

			auto& basis = basis_vec[pixel_idx];
			TGE_ASSERT(in_stddev.x <= 1.0f && in_stddev.y <= 1.0f, "invalid space");
			auto mat = Tempest::ToMatrix3(basis); // TODO: do it in less lousy fashion
			TGE_ASSERT(Tempest::Dot(Tempest::Cross(mat.tangent(), mat.binormal()), mat.normal()) > 0.0f, "Invalid basis");
			auto tangent = mat.column(0),
				 binormal = mat.column(1);

			bool redo_basis = false;
			float sign_tangent = 1.0f;
			if(out_stddev.x > out_stddev.y)
			{
				std::swap(tangent, binormal);
				std::swap(out_stddev.x, out_stddev.y);
				sign_tangent = -1.0f;
				redo_basis = true;
			}

			float sign = 1.0f;

			if(redo_basis)
			{
				mat.column(0) = sign_tangent*sign*tangent;
				mat.column(1) = sign*binormal;

				basis = Tempest::Normalize(Tempest::ToQuaternion(mat));
				TGE_ASSERT(Tempest::Dot(Tempest::Cross(mat.tangent(), mat.binormal()), mat.normal()) > 0.0f, "Invalid basis");
			}
		}
    
    if(!parser.isSet("disable-align-region"))
    {
        #define INVALID_MARKER (~0u)
        std::unique_ptr<uint32_t[]> marked_area(new uint32_t[in_stddev_hdr.Width*in_stddev_hdr.Height]);

        float discontinuity_angle = 0.0f;
        if(parser.isSet("discontinuity-angle"))
        {
            float angle = parser.extract<float>("discontinuity-angle");
            if(angle >= 180.0f)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Discontinuity angle should be smaller than 180 degrees");
                return EXIT_FAILURE;
            }

            discontinuity_angle = cosf(Tempest::ToRadians(angle));
        }

        for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
        {
            for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
            {
                auto cur_idx = y*in_stddev_hdr.Width + x;

                bool mark = false;

                auto& cur_basis = basis_vec[cur_idx];
                auto cur_tangent = Tempest::ToTangent(cur_basis);

                if(y > 0)
                {
                    if(x > 0u)
                    {
                        auto& basis = basis_vec[(y - 1)*in_stddev_hdr.Width + x - 1];
                        mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }

                    {
                       auto& basis = basis_vec[(y - 1)*in_stddev_hdr.Width + x];
                       mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }

                    if(x < in_stddev_hdr.Width - 1u)
                    {
                        auto& basis = basis_vec[(y - 1)*in_stddev_hdr.Width + x + 1];
                        mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }
                }

                {
                    if(x > 0)
                    {
                        auto& basis = basis_vec[y*in_stddev_hdr.Width + x - 1];
                        mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }

                    if(x < in_stddev_hdr.Width - 1u)
                    {
                        auto& basis = basis_vec[y*in_stddev_hdr.Width + x + 1];
                       mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }
                }

                if(y < in_stddev_hdr.Height - 1u)
                {
                    if(x > 0)
                    {
                        auto& basis = basis_vec[(y + 1)*in_stddev_hdr.Width + x - 1];
                        mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }

                    {
                       auto& basis = basis_vec[(y + 1)*in_stddev_hdr.Width + x];
                       mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }

                    if(x < in_stddev_hdr.Width - 1u)
                    {
                        auto& basis = basis_vec[(y + 1)*in_stddev_hdr.Width + x + 1];
                        mark |= Tempest::Dot(cur_tangent, Tempest::ToTangent(basis)) < discontinuity_angle;
                    }
                }
                
                if(mark)
                {
                    marked_area[cur_idx] = INVALID_MARKER;
                }
                else
                {
                    marked_area[cur_idx] = 0u;
                }
            }
        }

        // Forward flood
        uint32_t marker_count = 0;
        for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
        {
            for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
            {
                uint32_t proposed_marker = INVALID_MARKER;

                auto cur_idx = y*in_stddev_hdr.Width + x;
                if(marked_area[cur_idx] != INVALID_MARKER)
                {
                    if(y > 0)
                    {
                        if(x > 0)
                        {
                            auto marker = marked_area[(y - 1)*in_stddev_hdr.Width + x - 1];
                            if(marker < proposed_marker)
                                proposed_marker = marker;
                        }

                        auto marker = marked_area[(y - 1)*in_stddev_hdr.Width + x];
                        if(marker < proposed_marker)
                            proposed_marker = marker;

                        if(x < in_stddev_hdr.Width - 1u)
                        {
                            auto marker = marked_area[(y - 1)*in_stddev_hdr.Width + x + 1];
                            if(marker < proposed_marker)
                                proposed_marker = marker;
                        }
                    }

                    if(x > 0)
                    {
                        auto marker = marked_area[y*in_stddev_hdr.Width + x - 1];
                        if(marker < proposed_marker)
                            proposed_marker = marker;
                    }

                    if(proposed_marker == INVALID_MARKER)
                    {
                        proposed_marker = marker_count++;
                    }
                }
                marked_area[cur_idx] = proposed_marker;
            }
        }

        std::unique_ptr<uint32_t[]> connectivity_graph(new uint32_t[marker_count]);
        for(uint32_t idx = 0; idx < marker_count; ++idx)
            connectivity_graph[idx] = idx;

        auto choose_marker = [](uint32_t x, uint32_t y, uint32_t* marked_area, uint32_t width, uint32_t height, uint32_t* connectivity_graph, uint32_t* proposed_marker)
        {
            auto marker = marked_area[y*width + x];
            if(marker == INVALID_MARKER)
                return;

            marker = connectivity_graph[marker];

            if(marker < *proposed_marker)
                *proposed_marker = marker;
        };

        auto connect_regions = [](uint32_t x, uint32_t y, uint32_t* marked_area, uint32_t width, uint32_t height, uint32_t* connectivity_graph, uint32_t proposed_marker)
        {
            TGE_ASSERT(x < width && y < height, "Invalid coordinate");

            auto marker = marked_area[y*width + x];
            if(marker == INVALID_MARKER)
                return;

            connectivity_graph[marker] = proposed_marker;
        };

        for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
        {
            for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
            {
                auto cur_idx = y*in_stddev_hdr.Width + x;
                uint32_t proposed_marker = marked_area[cur_idx];

                if(proposed_marker != INVALID_MARKER)
                {
                    if(y > 0)
                    {
                        /*
                        if(x > 0)
                        {
                            choose_marker(x - 1, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                        }
                        */

                        choose_marker(x, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);

                        /*
                        if(x < in_stddev_hdr.Width - 1u)
                        {
                            choose_marker(x + 1, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                        }
                        */
                    }

                    if(x > 0)
                    {
                        choose_marker(x - 1, y, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                    }

                    if(x < in_stddev_hdr.Width - 1u)
                    {
                        choose_marker(x + 1, y, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                    }

                    if(y < in_stddev_hdr.Height - 1u)
                    {
                      /*
                        if(x > 0)
                        {
                            choose_marker(x - 1, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                        }
                      */
                        choose_marker(x, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);

                      /*  
                        if(x < in_stddev_hdr.Width - 1u)
                        {
                            choose_marker(x + 1, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), &proposed_marker);
                        }
                      */
                    }

                    // And now start replacing
                    if(y > 0)
                    {
                        /*
                        if(x > 0)
                        {
                            connect_regions(x - 1, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                        }
                        */

                        connect_regions(x, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);

                        /*
                        if(x < in_stddev_hdr.Width - 1u)
                        {
                            connect_regions(x + 1, y - 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                        }
                        */
                    }

                    if(x > 0)
                    {
                        connect_regions(x - 1, y, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                    }

                    if(x < in_stddev_hdr.Width - 1u)
                    {
                        connect_regions(x + 1, y, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                    }

                    if(y < in_stddev_hdr.Height - 1u)
                    {
                        /*
                        if(x > 0)
                        {
                            connect_regions(x - 1, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                        }
                        */

                        connect_regions(x, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                        
                        /*
                        if(x < in_stddev_hdr.Width - 1u)
                        {
                            connect_regions(x + 1, y + 1, marked_area.get(), in_stddev_hdr.Width, in_stddev_hdr.Height, connectivity_graph.get(), proposed_marker);
                        }
                        */
                    }
                }

                marked_area[cur_idx] = proposed_marker;
            }
        }

        // Propagate labels
        for(uint32_t marker_idx = 0; marker_idx < marker_count; ++marker_idx)
        {
            auto& orig_label = connectivity_graph[marker_idx];
            uint32_t cur_label = orig_label;
            TGE_ASSERT(cur_label != INVALID_MARKER, "Invalid marker");
            uint32_t next_label;
            while((next_label = connectivity_graph[cur_label]) != cur_label)
            {
                cur_label = next_label;
            }
            orig_label = cur_label;
        }

        std::unique_ptr<uint32_t[]> smallest_subset(new uint32_t[marker_count]);
        std::copy_n(connectivity_graph.get(), marker_count, smallest_subset.get());
        std::sort(smallest_subset.get(), smallest_subset.get() + marker_count);
        auto smallest_end = std::unique(smallest_subset.get(), smallest_subset.get() + marker_count);

        auto smallest_subset_size = smallest_end - smallest_subset.get();

        for(uint32_t marker_idx = 0; marker_idx < marker_count; ++marker_idx)
        {
            auto marker = connectivity_graph[marker_idx];

            auto new_id_iter = std::lower_bound(smallest_subset.get(), smallest_end, marker);
            TGE_ASSERT(new_id_iter != smallest_end && *new_id_iter == marker, "Broken marker subset");

            connectivity_graph[marker_idx] = static_cast<uint32_t>(new_id_iter - smallest_subset.get());
        }

        std::unique_ptr<RegionStats[]> region_stats(new RegionStats[smallest_subset_size]);
        for(uint32_t idx = 0, idx_end = in_stddev_hdr.Width*in_stddev_hdr.Height; idx < idx_end; ++idx)
        {
            auto marker = marked_area[idx];
            if(marker == INVALID_MARKER)
                continue;

            marker = marked_area[idx] = connectivity_graph[marker];
            auto& region = region_stats[marker];

            auto& basis00 = basis_vec[idx];
            auto tangent = Tempest::ToTangent(basis00);
            
            region.Direction += tangent;
            ++region.Coverage;
        }

        auto max_iter = std::max_element(region_stats.get(), region_stats.get() + smallest_subset_size,
                                         [](const RegionStats& lhs, const RegionStats& rhs)
                                         {
                                             return lhs.Coverage < rhs.Coverage;
                                         });

        auto dominant_direction = Normalize(max_iter->Direction);

        if(parser.isSet("save-region-tangent-texture"))
        {
            Tempest::TextureDescription marker_tex_desc;
            marker_tex_desc.Width = in_stddev_hdr.Width;
            marker_tex_desc.Height = in_stddev_hdr.Height;
            marker_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

            std::unique_ptr<uint32_t[]> color_marker_data(new uint32_t[in_stddev_hdr.Width*in_stddev_hdr.Height]);

            for(uint32_t idx = 0, idx_end = in_stddev_hdr.Width*in_stddev_hdr.Height; idx < idx_end; ++idx)
            {
                auto marker = marked_area[idx];
                if(marker == INVALID_MARKER)
                {
                    color_marker_data[idx] = 0;
                    continue;
                }

                auto color = Tempest::Normalize(region_stats[marker].Direction)*0.5f + 0.5f;
                color_marker_data[idx] = Tempest::ToColor(color);
            }

            Tempest::Path marker_path(parser.extractString("save-region-tangent-texture"));
            auto status = Tempest::SaveImage(marker_tex_desc, color_marker_data.get(), marker_path);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Failed to save marker texture: ", marker_path);
                return EXIT_FAILURE;
            }
        }

        bool flip_direction = false;
        Tempest::Vector3 orient_tangent = dominant_direction;
        if(parser.isSet("tangent"))
        {
            auto tangent_str = parser.extractString("tangent");
            if(!Tempest::ParseDirection(tangent_str.c_str(), &orient_tangent))
            {
                return EXIT_FAILURE;
            }
            flip_direction = Tempest::Dot(dominant_direction, orient_tangent) < 0;
        }

        bool prefer_tangent = parser.isSet("prefer-tangent");

        for(uint32_t marker_idx = 0; marker_idx < smallest_subset_size; ++marker_idx)
        {
            auto& region = region_stats[marker_idx];
        
            region.Direction = Tempest::Normalize(region.Direction);

            if(prefer_tangent)
            {
                region.PerformFlip = Dot(region.Direction, orient_tangent) < 0.0f;
            }
            else
            {
                region.PerformFlip = (flip_direction != (Dot(region.Direction, dominant_direction) < 0.0f));
            }
        }
        
        for(uint32_t idx = 0, idx_end = in_stddev_hdr.Width*in_stddev_hdr.Height; idx < idx_end; ++idx)
        {
            auto marker = marked_area[idx];

            auto& basis = basis_vec[idx];
		    auto mat = Tempest::ToMatrix3(basis); // TODO: do it in less lousy fashion
		    TGE_ASSERT(Tempest::Dot(Tempest::Cross(mat.tangent(), mat.binormal()), mat.normal()) > 0.0f, "Invalid basis");
		    auto tangent = mat.column(0),
		         binormal = mat.column(1);

            if(marker != INVALID_MARKER)
            {
                auto& region = region_stats[marker];
                if(!region.PerformFlip)
                    continue;
            }
            else if(prefer_tangent)
            {
                if(Dot(tangent, orient_tangent) < 0.0f)
                    continue;
            }
            else if(flip_direction != (Tempest::Dot(tangent, dominant_direction) >= 0.0f))
                continue;

            TGE_ASSERT(Tempest::Dot(Tempest::Cross(mat.tangent(), mat.binormal()), mat.normal()) > 0.0f, "Invalid basis");

            mat.column(0) = -tangent;
		    mat.column(1) = -binormal;

		    basis = Tempest::Normalize(Tempest::ToQuaternion(mat));
		    TGE_ASSERT(Tempest::Dot(Tempest::Cross(mat.tangent(), mat.binormal()), mat.normal()) > 0.0f, "Invalid basis");
        }

        if(parser.isSet("save-marker-texture"))
        {
            Tempest::TextureDescription marker_tex_desc;
            marker_tex_desc.Width = in_stddev_hdr.Width;
            marker_tex_desc.Height = in_stddev_hdr.Height;
            marker_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

            std::unique_ptr<uint32_t[]> color_marker_data(new uint32_t[in_stddev_hdr.Width*in_stddev_hdr.Height]);

            for(uint32_t idx = 0, idx_end = in_stddev_hdr.Width*in_stddev_hdr.Height; idx < idx_end; ++idx)
            {
                auto color =  Tempest::ColorCodeHSL4ToRGB((float)marked_area[idx]/(smallest_subset_size - 1));
                color_marker_data[idx] = Tempest::ToColor(color);
            }

            Tempest::Path marker_path(parser.extractString("save-marker-texture"));
            auto status = Tempest::SaveImage(marker_tex_desc, color_marker_data.get(), marker_path);
            if(!status)
            {
                Tempest::Log(Tempest::LogLevel::Error, "Failed to save marker texture: ", marker_path);
                return EXIT_FAILURE;
            }
        }

        
    }

    
    Tempest::Path in_path(in_prefix);
	auto subdir = in_path.directoryPath();
	auto prefix_filename = in_path.filename();

	auto output_base = parser.extractString("output");
	if(output_base.empty())
		output_base = subdir + "/modified_" + prefix_filename;

    Tempest::TextureDescription out_stddev_hdr = in_sggx_stddev_map->getHeader();
    out_stddev_hdr.Format = Tempest::DataFormat::RG32F;

    size_t cross_fade = parser.extract<size_t>("cross-fade");
    auto lerp2 = [](const Tempest::Vector2& lhs, const Tempest::Vector2& rhs, float t){ return lhs*(1.0f - t) + rhs*t; };
    auto lerp3 = [](const Tempest::Vector3& lhs, const Tempest::Vector3& rhs, float t){ return lhs*(1.0f - t) + rhs*t; };
    auto slerp = [](const Tempest::Quaternion& lhs, const Tempest::Quaternion& rhs, float t){ return Tempest::Slerp(lhs, rhs, t); };

    Tempest::Texture out_sggx_stddev_map(out_stddev_hdr, reinterpret_cast<uint8_t*>(out_stddev_vec.release()));

	Tempest::Path out_stddev_texname(output_base + "_sggx_scale.exr");
	status = cross_fade ?
                Tempest::CrossFadeSaveImageTyped<Tempest::Vector2>(out_stddev_hdr, out_sggx_stddev_map.getData(), cross_fade, lerp2, out_stddev_texname) :
                Tempest::SaveImage(out_stddev_hdr, out_sggx_stddev_map.getData(), out_stddev_texname);
	if(!status)
	{
		Tempest::Log(Tempest::LogLevel::Error, "failed to save standard deviation: ", out_stddev_texname);
		return EXIT_FAILURE;
	}

	Tempest::Path out_basis_texname(output_base + "_sggx_basis.exr");
	status = cross_fade ?
                Tempest::CrossFadeSaveImageTyped<Tempest::Quaternion>(basis_hdr, sggx_basis_map->getData(), cross_fade, slerp, out_basis_texname) :
                Tempest::SaveImage(basis_hdr, sggx_basis_map->getData(), out_basis_texname);
	if(!status)
	{
		Tempest::Log(Tempest::LogLevel::Error, "failed to save basis: ", out_stddev_texname);
		return EXIT_FAILURE;
	}

    std::string in_albedo_texname(in_prefix + "_albedo.exr");
	std::string out_albedo_texname(output_base + "_albedo.exr");

    std::string in_specular_texname(in_prefix + "_specular.exr");
	std::string out_specular_texname(output_base + "_specular.exr");
    if(flip_y || cross_fade)
    {
        {
        std::unique_ptr<Tempest::Texture> in_albedo_tex(Tempest::LoadImage(Tempest::Path(in_albedo_texname)));
        if(!in_albedo_tex)
        {
            Tempest::Log(Tempest::LogLevel::Error, "Failed to load albedo texture: ", in_albedo_texname);
            return EXIT_FAILURE;
        }

        if(flip_y)
            in_albedo_tex->flipY();

        status = cross_fade ?
                    Tempest::CrossFadeSaveImageTyped<Tempest::Vector3>(in_albedo_tex->getHeader(), in_albedo_tex->getData(), cross_fade, lerp3, Tempest::Path(out_albedo_texname)) :
                    Tempest::SaveImage(in_albedo_tex->getHeader(), in_albedo_tex->getData(), Tempest::Path(out_albedo_texname));
	    if(!status)
	    {
		    Tempest::Log(Tempest::LogLevel::Error, "Failed to save albedo texture: ", out_albedo_texname);
		    return EXIT_FAILURE;
	    }
        }

        {
        std::unique_ptr<Tempest::Texture> in_specular_tex(Tempest::LoadImage(Tempest::Path(in_specular_texname)));
	    if(!in_specular_tex)
	    {
		    Tempest::Log(Tempest::LogLevel::Error, "Failed to load specular texture: ", in_specular_texname);
		    return EXIT_FAILURE;
	    }

        if(flip_y)
            in_specular_tex->flipY();

        status = cross_fade ? 
                    Tempest::CrossFadeSaveImageTyped<Tempest::Vector3>(in_specular_tex->getHeader(), in_specular_tex->getData(), cross_fade, lerp3, Tempest::Path(out_specular_texname)) :
                    Tempest::SaveImage(in_specular_tex->getHeader(), in_specular_tex->getData(), Tempest::Path(out_specular_texname));
	    if(!status)
	    {
		    Tempest::Log(Tempest::LogLevel::Error, "Failed to save specular texture: ", out_specular_texname);
		    return EXIT_FAILURE;
	    }
        }
    }
    else
    {
	    status = Tempest::System::FileCopy(in_albedo_texname, out_albedo_texname);
	    if(!status)
	    {
		    Tempest::Log(Tempest::LogLevel::Error, "Failed to copy albedo texture: ", in_albedo_texname, " -> ", out_albedo_texname);
		    return EXIT_FAILURE;
	    }

	    status = Tempest::System::FileCopy(in_specular_texname, out_specular_texname);
	    if(!status)
	    {
		    Tempest::Log(Tempest::LogLevel::Error, "Failed to copy specular texture: ", in_specular_texname, " -> ", out_specular_texname);
		    return EXIT_FAILURE;
	    }
    }

    if(parser.isSet("report-image"))
    {
        std::string out_error_texname(output_base + "_glitch_detected.png");

        Tempest::TextureDescription report_tex_desc;
        report_tex_desc.Width = in_stddev_hdr.Width;
        report_tex_desc.Height = in_stddev_hdr.Height;
        report_tex_desc.Format = Tempest::DataFormat::RGBA8UNorm;

        uint32_t* report_texture_data = new uint32_t[report_tex_desc.Width*report_tex_desc.Height];
        Tempest::Texture tex(report_tex_desc, reinterpret_cast<uint8_t*>(report_texture_data));

        uint32_t glitches = 0;

        uint8_t color_flip_area = 0;

        for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
        {
            for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
            {
                auto cur_idx = y*in_stddev_hdr.Width + x;

                auto xnext = x < in_stddev_hdr.Width - 1u ? x + 1u : x - 1u;
                auto ynext = y < in_stddev_hdr.Height - 1u ? y + 1u : y - 1u;

                auto& basis00 = basis_vec[cur_idx];
                auto& basis01 = basis_vec[y*in_stddev_hdr.Width + xnext];
                auto& basis10 = basis_vec[ynext*in_stddev_hdr.Width + x];
                auto& basis11 = basis_vec[ynext*in_stddev_hdr.Width + xnext];

                auto tangent00 = Tempest::ToTangent(basis00);
                auto tangent01 = Tempest::ToTangent(basis01);
                auto tangent10 = Tempest::ToTangent(basis10);
                auto tangent11 = Tempest::ToTangent(basis11);

                if(Tempest::Dot(tangent00, tangent01) < 0.0f ||
                    Tempest::Dot(tangent00, tangent10) < 0.0f ||
                    Tempest::Dot(tangent00, tangent11) < 0.0f)
                {
                    ++glitches;
                    report_texture_data[cur_idx] = 255u;
                }
                else
                {
                    report_texture_data[cur_idx] = 0u;
                }
            }
        }

        for(uint32_t y = 0; y < in_stddev_hdr.Height; ++y)
        {
            for(uint32_t x = 0; x < in_stddev_hdr.Width; ++x)
            {
                auto cur_idx = y*in_stddev_hdr.Width + x;

                auto& basis00 = basis_vec[cur_idx];
                auto tangent = Tempest::ToTangent(basis00);

                if(!report_texture_data[cur_idx])
                {
                    report_texture_data[cur_idx] = Tempest::ToColor(tangent*0.5f + 0.5f);
                }
                else
                {
                    report_texture_data[cur_idx] = {};
                }
            }
        }

        if(glitches)
        {
            Tempest::Log(Tempest::LogLevel::Info, "Glitches detected: ", glitches);
            Tempest::SaveImage(report_tex_desc, report_texture_data, Tempest::Path(out_error_texname));
        }
        else
        {
            remove(out_error_texname.c_str());
            Tempest::Log(Tempest::LogLevel::Info, "No glitches detected! Not writing texture!");
        }
    }

	return EXIT_SUCCESS;
}

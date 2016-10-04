#include "tempest/utils/testing.hh"
#include "tempest/graphics/ray-tracing/ray-tracing.hh"
#include "tempest/math/shapes.hh"
#include "tempest/utils/display-image.hh"
#include "tempest/math/vector3.hh"
#include "tempest/math/functions.hh"
#include "tempest/utils/patterns.hh"
#include "tempest/utils/refractive-indices.hh"
#include "tempest/math/curves.hh"

#include <algorithm>
#include <cstring>

const float FiberRadius = 0.05f;
const float FiberLength = 20.0f;
const float AngularSpeed = 0.1f;
const float YarnRadius = 0.1f;
const float FiberMinGap = 0.0f;
const float PatchSize = 10.0f;
const float SegmentLength = 1.0f;
const uint32_t SamplesLong = 1;
const uint32_t SamplesAzimuth = 1;

void ConvertCurveToHair(float radius, Tempest::Vector3* curve, int32_t vert_count, int32_t index_offset, int32_t* indices, Tempest::HairFormat* verts)
{
    const int32_t segment_count = vert_count - 3;

    Tempest::Vector3 orig_coef[4], transformed_coef[4];

    for(int32_t i = 0, vert_idx = 0; i < vert_count - 3; ++i)
    {
        orig_coef[0] = curve[i];
        orig_coef[1] = curve[i + 1];
        orig_coef[2] = curve[i + 2];
        orig_coef[3] = curve[i + 3];

        Tempest::CatmullRomToBezier(orig_coef, transformed_coef);

        int32_t end_vert_idx = vert_idx + 3;
        if(end_vert_idx == 3*segment_count)
            ++end_vert_idx;
        for(int32_t idx = 0; vert_idx < end_vert_idx; ++idx)
        {
            auto& vert = verts[vert_idx++];
            vert.Position = transformed_coef[idx];
            vert.Radius = radius;
        }
    }

    for(int32_t i = 0, j = 0; i < segment_count; ++i)
    {
        indices[i] = i*3 + index_offset;
    }
}

enum class SegmentPointType: uint8_t
{
    PileLeft,
    PileRight,
    Lower,
    LowerToUpper,
    Upper,
    UpperToLower
};

struct SegmentPoint
{
    SegmentPointType Type;
    int8_t  Start, End, Line;  
};

enum class GrammarElement: uint8_t
{
    Invalid,
    Submerge,
    SubmergeCut,
    Raise,
    Cut
};

static const char WeftGrammar[] = "|X-=";
static const char WarpGrammar[] = "-=|X";

GrammarElement TranslateGrammarElement(const char* grammar, char word)
{
    auto* ptr = strchr(grammar, word);
    return ptr ? static_cast<GrammarElement>(ptr - grammar + 1) : GrammarElement::Invalid;
}

void GenerateSegmentFeaturesDescription(const char* grammar, const char* weave_pattern, uint32_t row_count, uint32_t col_count, uint32_t stride, uint32_t row_pitch, std::vector<SegmentPoint>* interm_segments)
{
    for(uint32_t line_idx = 0; line_idx < row_count; ++line_idx)
    {
        const char* line = weave_pattern + line_idx*row_pitch;
        char prev_segment = line[stride*(col_count - 1)];
        int32_t prev_start = 0;
		const char* cur_point = line;
        for(const char* line_end = line + stride*col_count; cur_point < line_end; cur_point += stride)
        {
            if(prev_segment == *cur_point)
                continue;

            switch(TranslateGrammarElement(grammar, prev_segment))
            {
            case GrammarElement::Submerge:
            {
                switch(TranslateGrammarElement(grammar, *cur_point))
                {
                case GrammarElement::Raise:
                {
                    auto cur_pos = static_cast<int32_t>(cur_point - line);
					TGE_CHECK((cur_pos % stride) == 0, "Invalid position");
					cur_pos /= stride;
                    if(prev_start - cur_pos > 1)
                    {
                        interm_segments->push_back(SegmentPoint{ SegmentPointType::Lower, prev_start, cur_pos - 1, line_idx });
                    }
                    interm_segments->push_back(SegmentPoint{ SegmentPointType::LowerToUpper, cur_pos - 1, cur_pos, line_idx });
                    prev_start = cur_pos;
                } break;
                case GrammarElement::Cut:
                {
                    auto cur_pos = static_cast<int32_t>(cur_point - line);
					TGE_CHECK((cur_pos % stride) == 0, "Invalid position");
					cur_pos /= stride;
                    if(cur_pos - prev_start > 1)
                    {
                        interm_segments->push_back(SegmentPoint{ SegmentPointType::Lower, prev_start, cur_pos - 1, line_idx });
                    }
                    interm_segments->push_back(SegmentPoint{ SegmentPointType::PileRight, cur_pos - 1, cur_pos, line_idx });
                    prev_start = cur_pos;
                } break;
				case GrammarElement::SubmergeCut: break;
                default:
                {
                    TGE_CHECK(false, "Invalid weave pattern");
                    return;
                }
                }
            } break;
            case GrammarElement::SubmergeCut:
            {
                switch(TranslateGrammarElement(grammar, *cur_point))
                {
				case GrammarElement::Submerge: break;
                case GrammarElement::Raise: break; // Nothing - it is just extending
                default:
                {
                    TGE_CHECK(false, "Invalid weave pattern");
                    return;
                }
                }
            } break;
            case GrammarElement::Raise:
            {
                switch(TranslateGrammarElement(grammar, *cur_point))
                {
                case GrammarElement::Submerge:
                {
                    auto cur_pos = static_cast<int32_t>(cur_point - line);
					TGE_CHECK((cur_pos % stride) == 0, "Invalid position");
					cur_pos /= stride;
                    if(cur_pos - prev_start > 1)
                    {
                        interm_segments->push_back(SegmentPoint{ SegmentPointType::Upper, prev_start, cur_pos - 1, line_idx });
                    }
                    interm_segments->push_back(SegmentPoint{ SegmentPointType::UpperToLower, cur_pos - 1, cur_pos, line_idx });
                    prev_start = cur_pos;
                } break;
                case GrammarElement::SubmergeCut: break; // Nothing - it is just extending
                default:
                {
                    TGE_CHECK(false, "Invalid weave pattern");
                    return;
                }
                }
            } break;
            case GrammarElement::Cut:
            {
                switch(TranslateGrammarElement(grammar, *cur_point))
                {
                case GrammarElement::Submerge:
                {
					auto cur_pos = static_cast<int32_t>(cur_point - line);
					TGE_CHECK((cur_pos % stride) == 0, "Invalid position");
					cur_pos /= stride;
                    interm_segments->push_back(SegmentPoint{ SegmentPointType::PileLeft, cur_pos - 1, cur_pos, line_idx });
                    prev_start = cur_pos;
                } break;
                default:
                {
                    TGE_CHECK(false, "Invalid weave pattern");
                    return;
                }
                }
            } break;
			default:
				TGE_CHECK(false, "Invalid weave pattern");
            }

            prev_segment = *cur_point;
        }

		int32_t cur_pos = static_cast<int32_t>(cur_point - line)/stride - 1;
		if(cur_pos > prev_start)
		{
			switch(TranslateGrammarElement(grammar, prev_segment))
			{
			case GrammarElement::Raise:
			{
				interm_segments->push_back(SegmentPoint{ SegmentPointType::Upper, prev_start, cur_pos, line_idx });
			} break;
			case GrammarElement::Submerge:
			case GrammarElement::SubmergeCut:
			{
				interm_segments->push_back(SegmentPoint{ SegmentPointType::Lower, prev_start, cur_pos, line_idx });
			} break;
			}
		}
    }
}

void ConvertSegmentDescriptionToHair(const std::vector<SegmentPoint>& interm_segments, std::vector<Tempest::HairFormat>* segments, std::vector<int32_t>* indices)
{
    for(uint32_t seg_idx = 0; seg_idx < interm_segments.size(); ++seg_idx)
    {
        auto& interm_segment = interm_segments[seg_idx];
        float line_offset = interm_segment.Line*SegmentLength;
		float start_offset = interm_segment.Start*SegmentLength;
		float end_offset = interm_segment.End*SegmentLength;
        float half_way = (start_offset + end_offset)*0.5f;
        float fiber_diam = YarnRadius;
        switch(interm_segment.Type)
        {
        case SegmentPointType::PileLeft:
        {
                Tempest::Vector3 curve[6]
                {
                    { half_way + fiber_diam, line_offset, +6.0f*YarnRadius },
                    { half_way + fiber_diam, line_offset, +6.0f*YarnRadius },
                    { half_way + fiber_diam, line_offset, +2.0f*YarnRadius },
                    { start_offset*0.25f + end_offset*0.75f, line_offset, 0.0f },
                    { end_offset, line_offset, -2.0f*YarnRadius },
                    { end_offset + SegmentLength, line_offset, -2.0f*YarnRadius }
                };                                               
                 
                const int32_t segment_count = TGE_FIXED_ARRAY_SIZE(curve) - 3;
                auto start_index_indices = indices->size();
                auto start_index_segments = segments->size();
                indices->resize(start_index_indices + segment_count);
                segments->resize(start_index_segments + 3*segment_count + 1);

                ConvertCurveToHair(YarnRadius, curve, TGE_FIXED_ARRAY_SIZE(curve), static_cast<int32_t>(start_index_segments), &(*indices)[start_index_indices], &(*segments)[start_index_segments]);
        } break;
        case SegmentPointType::PileRight:
        {
                Tempest::Vector3 curve[6]
                {
                    { start_offset - SegmentLength, line_offset, -2.0f*YarnRadius },
                    { start_offset, line_offset, -2.0f*YarnRadius },
                    { start_offset*0.75f + end_offset*0.25f, line_offset, 0.0f },
                    { half_way - fiber_diam, line_offset, +2.0f*YarnRadius },
                    { half_way - fiber_diam, line_offset, +6.0f*YarnRadius },
                    { half_way - fiber_diam, line_offset, +6.0f*YarnRadius }
                };
                                                
                const int32_t segment_count = TGE_FIXED_ARRAY_SIZE(curve) - 3;
                auto start_index_indices = indices->size();
                auto start_index_segments = segments->size();
                indices->resize(start_index_indices + segment_count);
                segments->resize(start_index_segments + 3*segment_count + 1);

                ConvertCurveToHair(YarnRadius, curve, TGE_FIXED_ARRAY_SIZE(curve), static_cast<int32_t>(start_index_segments), &(*indices)[start_index_indices], &(*segments)[start_index_segments]);
        } break;
        case SegmentPointType::Lower:
        {
            indices->push_back(static_cast<int32_t>(segments->size()));
            segments->push_back(Tempest::HairFormat{ { start_offset, line_offset, -2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { start_offset, line_offset, -2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { end_offset, line_offset, -2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { end_offset, line_offset, -2.0f*YarnRadius }, YarnRadius });
        } break;
        case SegmentPointType::LowerToUpper:
        {
            Tempest::Vector3 curve[4]
            {
                { start_offset - SegmentLength, line_offset, -2.0f*YarnRadius },
                { start_offset, line_offset, -2.0f*YarnRadius },
                { end_offset, line_offset, +2.0f*YarnRadius },
                { end_offset + SegmentLength, line_offset, +2.0f*YarnRadius }
            };
                
            const int32_t segment_count = TGE_FIXED_ARRAY_SIZE(curve) - 3;
            auto start_index_indices = indices->size();
            auto start_index_segments = segments->size();
            indices->resize(start_index_indices + segment_count);
            segments->resize(start_index_segments + 3*segment_count + 1);

            ConvertCurveToHair(YarnRadius, curve, TGE_FIXED_ARRAY_SIZE(curve), static_cast<int32_t>(start_index_segments), &(*indices)[start_index_indices], &(*segments)[start_index_segments]);
        } break;
        case SegmentPointType::Upper:
        {
            indices->push_back(static_cast<int32_t>(segments->size()));
            segments->push_back(Tempest::HairFormat{ { start_offset, line_offset, +2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { start_offset, line_offset, +2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { end_offset, line_offset, +2.0f*YarnRadius }, YarnRadius });
            segments->push_back(Tempest::HairFormat{ { end_offset, line_offset, +2.0f*YarnRadius }, YarnRadius });
        } break;
        case SegmentPointType::UpperToLower:
        {
            Tempest::Vector3 curve[4]
            {
                { start_offset - SegmentLength, line_offset, +2.0f*YarnRadius },
                { start_offset, line_offset, +2.0f*YarnRadius },
                { end_offset, line_offset, -2.0f*YarnRadius },
                { end_offset + SegmentLength, line_offset, -2.0f*YarnRadius }
            };

            const int32_t segment_count = TGE_FIXED_ARRAY_SIZE(curve) - 3;
            auto start_index_indices = indices->size();
            auto start_index_segments = segments->size();
            indices->resize(start_index_indices + segment_count);
            segments->resize(start_index_segments + 3*segment_count + 1);

            ConvertCurveToHair(YarnRadius, curve, TGE_FIXED_ARRAY_SIZE(curve), static_cast<int32_t>(start_index_segments), &(*indices)[start_index_indices], &(*segments)[start_index_segments]);
        } break;
        }
    }
}

TGE_TEST("Testing how good am I at weaving stuff with algorithms")
{
	float proj_horiz_span = PatchSize;
	float proj_vert_span = proj_horiz_span;

    uint32_t image_width = 500,
                    image_height = 500;

    Tempest::Matrix4 proj = Tempest::OrthoMatrix(-proj_horiz_span, proj_horiz_span, -proj_vert_span, proj_vert_span, 0.1f, 1000.0f);

    Tempest::Vector3 target{0.0f, 0.0f, 0.0f},
                     origin{0.0f, -15.0f, 0.0f},
                     up{0.0f, 0.0f, 1.0f};

    Tempest::Matrix4 view;
    
    view.identity();
    view.lookAt(origin, target, up);

    Tempest::Matrix4 view_proj = proj * view;

    Tempest::Matrix4 view_proj_inv = view_proj.inverse();

    Tempest::Vector3 target_in_view = view*up;

    Tempest::Matrix4 view_inv;
    view_inv = view.inverse();
    
    std::unique_ptr<Tempest::RayTracerScene> rt_scene(new Tempest::RayTracerScene(image_width, image_height, view_proj_inv));

	/*
    Tempest::SphereAreaLight* area_light = new Tempest::SphereAreaLight;
    area_light->SphereShape.Center = Tempest::Vector3{0.0f, 0.0f, 0.0f};
    area_light->SphereShape.Radius = 0.1f;
    area_light->Radiance = Tempest::Vector3{5000.0f, 5000.0f, 5000.0f};

    rt_scene->addLightSource(area_light);
	*/

	float dir_light_angle = 0.0f;
	float sin_theta, cos_theta;
    Tempest::FastSinCos(Tempest::ToRadians(dir_light_angle), &sin_theta, &cos_theta);

    float dir_light_alt_angle = 0.0f;
    float sin_phi, cos_phi;
    Tempest::FastSinCos(Tempest::ToRadians(dir_light_alt_angle), &sin_phi, &cos_phi);

	Tempest::DirectionalLight* dir_light = new Tempest::DirectionalLight;
	dir_light->Direction = Tempest::Vector3{-sin_theta*cos_phi, -cos_theta*cos_phi, sin_phi};
    NormalizeSelf(&dir_light->Direction);
	dir_light->Radiance = Tempest::RGBToSpectrum(Tempest::Vector3{10.0f, 10.0f, 10.0f});
	rt_scene->addLightSource(dir_light);

	rt_scene->setSamplesCamera(64);
    rt_scene->setSamplesLocalAreaLight(1);
    rt_scene->setSamplesGlobalIllumination(1);
    rt_scene->setMaxRayDepth(0);
    rt_scene->setRussianRoulette(0.9f);

    rt_scene->setBackgroundSpectrum(Tempest::RGBToSpectrum(Tempest::Vector3{0.0f, 0.0f, 0.0f}));

    Tempest::RTMicrofacetMaterial material;
	memset(&material, 0x0, sizeof(material));
	
    Tempest::Spectrum color = Tempest::RGBToSpectrum(Tempest::Vector3{ 1.0f, 1.0f, 1.0f });

    material.Model = Tempest::IlluminationModel::GGXMicrofacetDielectric;
	material.Diffuse = color*1.0f;
    material.Specular = color*0.0f;
    material.SpecularPower = Tempest::Vector2{ 100.0f, 100.0f };
	material.Fresnel = Tempest::CelluloseRefractiveIndex;
    material.setup();

    struct FiberLayer
    {
        Tempest::ShapeVertex* Vertices = nullptr;
        uint32_t              VertexCount = 0;
        int32_t*              Indices = nullptr;
        uint32_t              IndexCount = 0;

        ~FiberLayer()
        {
            delete[] Vertices;
            delete[] Indices;
        }
    };

    Tempest::Vector3 yarn_pos = target;

    // Compared to some literature on pile weaving patterns - this format tends to be non-ambiguous
    // = Cut weft
    // X Cut warp
    // - Weft
    // | Warp
    // =|= is fine; -|- is also; =- is impossible and the same applies in reverse
    // =X= is crazy, but -X- should be fine
    // Cutdown yarns wrap around the endpoints
    const char weave_pattern[3][13] =
    {
        "|======|====",
        "=|====|=====",
        "|-|-|-|-|-|-",
    };

    const uint32_t row_count = TGE_FIXED_ARRAY_SIZE(weave_pattern),
                          col_count = TGE_FIXED_ARRAY_SIZE(weave_pattern[0]) - 1;

    std::vector<Tempest::HairFormat> segments;
    std::vector<int32_t> indices;
	Tempest::RTSubhair fabric;

    if(col_count)
    {
        // Segment the pattern, so it is easy to convert to yarns and fibers
        std::vector<SegmentPoint> interm_segments;

		//*
        GenerateSegmentFeaturesDescription(WeftGrammar, *weave_pattern, row_count, col_count, sizeof(weave_pattern[0][0]), (col_count + 1)*sizeof(weave_pattern[0][0]), &interm_segments);     

        // Merge segments when non-colliding - removes pointless warps and wefts which make sense only then weaving

        // Convert to yarns and fibers
        ConvertSegmentDescriptionToHair(interm_segments, &segments, &indices);
		//*/

		interm_segments.clear();
		size_t warps_start = segments.size();

        //*
        GenerateSegmentFeaturesDescription(WarpGrammar, *weave_pattern, col_count, row_count, sizeof(weave_pattern[0][0])*(col_count + 1), sizeof(weave_pattern[0][0]), &interm_segments);     

        // Merge segments when non-colliding - removes pointless warps and wefts which make sense only then weaving

        // Convert to yarns and fibers
        ConvertSegmentDescriptionToHair(interm_segments, &segments, &indices);
		//*/
        for(size_t i = warps_start; i < segments.size(); ++i)
        {
            auto& segment = segments[i];
            std::swap(segment.Position.x, segment.Position.y);
        }
	}

    fabric.BaseIndex = 0;
    fabric.Material = &material;
    fabric.VertexCount = 4*static_cast<uint32_t>(indices.size());
    fabric.VertexOffset = 0;

    Tempest::Matrix4 mat;
    mat.identity();
    mat.translate(target);

    rt_scene->addHair(mat, 1, &fabric, indices.size(), &indices.front(), segments.size()*sizeof(Tempest::HairFormat), &segments.front(), sizeof(Tempest::HairFormat));
    
    rt_scene->commitScene();

    Tempest::TimeQuery query;
    auto start = query.time();

    rt_scene->initWorkers();

    uint32_t image_size = image_width*image_height;
    
    auto* frame_data = rt_scene->drawOnce();

    auto end = query.time();

    Tempest::Log(Tempest::LogLevel::Info, "Render time (ray tracing): ", end - start, "us");
   
    auto* backbuffer = frame_data->Backbuffer.get();
    Tempest::SaveImage(backbuffer->getHeader(), backbuffer->getData(), Tempest::Path("yarn.tga"));
    Tempest::DisplayImage(backbuffer->getHeader(), backbuffer->getData());
}
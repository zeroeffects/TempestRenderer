#include "tempest/math/matrix4.hh"
#include "tempest/graphics/api-all.hh"
#include "tempest/math/ode.hh"
#include "tempest/utils/timer.hh"

const uint64_t UpdateStep = 1000;

const float GrabRadius = 0.5f;
const float MaxGrabDistance = 0.75f; // in cell fractions

struct Spring
{
    float Mass;
    float Damping;
    float SpringConstant;
};

class Cloth
{
    Spring          m_StructuralSprings,
                    m_ShearingSprings,
                    m_BendingSprings,
					m_AnchorSprings;

    uint32_t        m_SizeX,
                    m_SizeY;

    Tempest::Vector2* m_Points = nullptr;
public:
    Cloth(uint32_t size_x, uint32_t size_y)
        :   m_SizeX(size_x),
            m_SizeY(size_y),
            m_Points(new Tempest::Vector2[2*size_x*size_y])
    {
        m_StructuralSprings = { 20.0f, 100.0f, 50.0f };
        m_ShearingSprings = { 20.0f, 100.0f, 50.0f };
        m_BendingSprings = { 20.0f, 100.0f, 50.0f };
		m_AnchorSprings = { 20.0f, 100.0f, 10.0f };
    }

    ~Cloth()
    {
        delete[] m_Points;
    }

    uint32_t getSizeX() { return m_SizeX; }
    uint32_t getSizeY() { return m_SizeY; }

    Tempest::Vector2* getPoints() { return m_Points; }
    Tempest::Vector2* getVelocities() { return m_Points + m_SizeX*m_SizeY; }

    void simulationStep(float step_size, uint32_t* mask_table, uint32_t* mask_table_end)
    {
        auto* points = getPoints();
        auto* velocities = getVelocities();

        Tempest::Vector2 step_size_v2{ 1.0f/(m_SizeX - 1), 1.0f/(m_SizeY - 1) };

        float diag_size = Tempest::Length(step_size_v2);

        for(uint32_t point_y = 0, point_y_end = m_SizeY; point_y < point_y_end; ++point_y)
        {
            for(uint32_t point_x = 0, point_x_end = m_SizeX; point_x < point_x_end; ++point_x)
            {
                auto idx = point_y*point_x_end + point_x;
                if(mask_table && std::find(mask_table, mask_table_end, idx) != mask_table_end)
                    continue;

                auto& point = points[idx];
                auto& velocity = velocities[idx];

				struct CutOffDeriv
				{
					float Distance;
					float Value;

					CutOffDeriv(float dist, float value)
						:	Distance(dist),
							Value(value) {}

					inline float operator()(const Tempest::Vector2& x) const
					{
						float len = Length(x);
						if(Distance < len)
							return Value;
						else
							return len*Value/Distance; 
					}
				};

				//*
				auto structural_damping_x = m_StructuralSprings.Damping;
				auto structural_sprconst_x = m_StructuralSprings.SpringConstant;

				auto structural_damping_y = m_StructuralSprings.Damping;
				auto structural_sprconst_y = m_StructuralSprings.SpringConstant;

				auto shearing_damping = m_ShearingSprings.Damping;
				auto shearing_sprconst = m_ShearingSprings.SpringConstant;

				auto anchoring_damping = m_AnchorSprings.Damping;
				auto anchoring_sprconst = m_AnchorSprings.SpringConstant;

				#define ODESolver Tempest::SolveSecondOrderLinearODE
				/*/
				auto structural_damping_x = CutOffDeriv(step_size_v2.x*1.15f, m_StructuralSprings.Damping);
				auto structural_sprconst_x = CutOffDeriv(step_size_v2.x*1.15f, m_StructuralSprings.SpringConstant);

				auto structural_damping_y = CutOffDeriv(step_size_v2.y*1.15f, m_StructuralSprings.Damping);
				auto structural_sprconst_y = CutOffDeriv(step_size_v2.y*1.15f, m_StructuralSprings.SpringConstant);

				auto shearing_damping = CutOffDeriv(diag_size*1.15f, m_ShearingSprings.Damping);
				auto shearing_sprconst = CutOffDeriv(diag_size*1.15f, m_ShearingSprings.SpringConstant);

				#define ODESolver Tempest::EulerMethodSolveSecondOrderNonLinearODE
				//*/

				//*
                if(point_x > 0)
                {
                    auto& prev_point = points[point_y*point_x_end + point_x - 1];
                    ODESolver(m_StructuralSprings.Mass, structural_damping_x, structural_sprconst_x, 0.0f, step_size, step_size_v2.x, prev_point, &point, &velocity);
                }

                if(point_x < point_x_end - 1)
                {
                    auto& prev_point = points[point_y*point_x_end + point_x + 1];
                    ODESolver(m_StructuralSprings.Mass, structural_damping_x, structural_sprconst_x, 0.0f, step_size, step_size_v2.x, prev_point, &point, &velocity);
                }

                if(point_y > 0)
                {
                    auto& prev_point = points[(point_y - 1)*point_x_end + point_x];
                    ODESolver(m_StructuralSprings.Mass, structural_damping_y, structural_sprconst_y, 0.0f, step_size, step_size_v2.y, prev_point, &point, &velocity);
                }

                if(point_y < point_y_end - 1)
                {
                    auto& prev_point = points[(point_y + 1)*point_x_end + point_x];
                    ODESolver(m_StructuralSprings.Mass, structural_damping_y, structural_sprconst_y, 0.0f, step_size, step_size_v2.y, prev_point, &point, &velocity);
                }

                if(point_x > 0 && point_y > 0)
                {
                    auto& prev_point = points[(point_y - 1)*point_x_end + point_x - 1];
                    ODESolver(m_ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, diag_size, prev_point, &point, &velocity);
                }

                if(point_x > 0 && point_y < point_y_end - 1)
                {
                    auto& prev_point = points[(point_y + 1)*point_x_end + point_x - 1];
                    ODESolver(m_ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, diag_size, prev_point, &point, &velocity);
                }

                if(point_x < point_x_end - 1 && point_y > 0)
                {
                    auto& prev_point = points[(point_y - 1)*point_x_end + point_x + 1];
                    ODESolver(m_ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, diag_size, prev_point, &point, &velocity);
                }

                if(point_x < point_x_end - 1 && point_y < point_y_end - 1)
                {
                    auto& prev_point = points[(point_y + 1)*point_x_end + point_x + 1];
                    ODESolver(m_ShearingSprings.Mass, shearing_damping, shearing_sprconst, 0.0f, step_size, diag_size, prev_point, &point, &velocity);
                }
				//*/
				ODESolver(m_AnchorSprings.Mass, anchoring_damping, anchoring_sprconst, 0.0f, step_size, 0.0f, Tempest::Vector2{ (float)point_x, (float)point_y }*step_size_v2, &point, &velocity);
            }
        }
    }
};



const uint32_t ClothSizeX = 11;
const uint32_t ClothSizeY = 11;

int TempestMain(int argc, char** argv)
{
    Tempest::WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Test window";
    auto sys_obj = Tempest::CreateSystemAndWindowSimple<Tempest::PreferredSystem>(wdesc);
    TGE_ASSERT(sys_obj, "GL initialization failed");

    Tempest::CommandBufferDescription cmd_buffer_desc;
    cmd_buffer_desc.CommandCount = 16;
    cmd_buffer_desc.ConstantsBufferSize = 1024;

    auto command_buf = Tempest::CreateCommandBuffer(&sys_obj->Backend, cmd_buffer_desc);
    
    uint32_t index_count = 2*((ClothSizeX - 1)*ClothSizeY + // Horizontal
                                     ClothSizeX*(ClothSizeY - 1) + // Vertical
                                     2*(ClothSizeX - 1)*(ClothSizeY - 1)); // Diagonal

    std::unique_ptr<uint16_t[]> idx_arr(new uint16_t[index_count]);

    Cloth cloth(ClothSizeX, ClothSizeY);

    auto* points = cloth.getPoints();
    auto* velocities = cloth.getVelocities();

    size_t index = 0;
    for(uint32_t point_y = 0; point_y < ClothSizeY; ++point_y)
    {
        for(uint32_t point_x = 0; point_x < ClothSizeX; ++point_x)
        {
            auto& point = points[point_y*ClothSizeX + point_x];
            point.x = (float)point_x / (ClothSizeX - 1);
            point.y = (float)point_y / (ClothSizeY - 1);
            auto& velocity = velocities[point_y*ClothSizeX + point_x];
            velocity.x = velocity.y = 0.0f;

            if(point_x != ClothSizeX - 1 && point_y != ClothSizeY - 1)
            {
                idx_arr[index++] = point_y*ClothSizeX + point_x;
                idx_arr[index++] = (point_y + 1)*ClothSizeX + point_x + 1;
                idx_arr[index++] = point_y*ClothSizeX + point_x + 1;
                idx_arr[index++] = (point_y + 1)*ClothSizeX + point_x;
            }

            if(point_x != ClothSizeX - 1)
            {
                idx_arr[index++] = point_y*ClothSizeX + point_x;
                idx_arr[index++] = point_y*ClothSizeX + point_x + 1;
            }

            if(point_y != ClothSizeY - 1)
            {
                idx_arr[index++] = point_y*ClothSizeX + point_x;
                idx_arr[index++] = (point_y + 1)*ClothSizeX + point_x;
            }
        }
    }

    TGE_ASSERT(index == index_count, "Invalid grid generator");

    auto vertex_buf = Tempest::CreateBuffer(&sys_obj->Backend, points, ClothSizeX*ClothSizeY*sizeof(Tempest::Vector2), Tempest:: ResourceBufferType::VertexBuffer);
    auto index_buf = Tempest::CreateBuffer(&sys_obj->Backend, idx_arr.get(), index_count*sizeof(uint16_t), Tempest:: ResourceBufferType::IndexBuffer);
    
    auto shader = Tempest::CreateShader(&sys_obj->ShaderCompiler, CURRENT_SOURCE_DIR "/spring.tfx");
    TGE_ASSERT(shader, "Expecting successful compilation");
    auto res_table = shader->createResourceTable("Globals", 1);
    TGE_ASSERT(res_table, "Expecting valid resource table");
    Tempest::Matrix4 mat;
    mat.identity();
    mat.scale(Tempest::Vector3{ 1.0f/wdesc.Width, 1.0f/wdesc.Height, 1.0f });
    mat.translate(Tempest::Vector2{ -0.5f*wdesc.Width, -0.5f*wdesc.Width });
    mat.scale(Tempest::Vector3{ (float)wdesc.Width, (float)wdesc.Width, 1.0f });
    res_table->setResource("Globals[0].Transform", mat);
    auto baked_table = Tempest::ExtractBakedResourceTable(res_table);
    
    Tempest::Matrix4 mat_inverse = mat.inverse();

    TGE_ASSERT(shader, "Could not create shader file");
    
    Tempest::DataFormat rt_fmt = Tempest::DataFormat::RGBA8UNorm;
    
    auto pipeline_state = Tempest::CreateStateObject(&sys_obj->Backend, &rt_fmt, 1, Tempest::DataFormat::Unknown, shader.get(), Tempest::DrawModes::LineList);
    
    typedef decltype(sys_obj->Backend) BackendType;
    BackendType::CommandBufferType::DrawBatchType batch;
    batch.VertexCount = index_count;
    batch.ResourceTable = baked_table.get();
    batch.IndexBuffer = index_buf.get();
    batch.PipelineState = pipeline_state.get();
    batch.VertexBuffers[0].VertexBuffer = vertex_buf.get();
    batch.VertexBuffers[0].Stride = sizeof(Tempest::Vector2);
    
    command_buf->enqueueBatch(batch);
    command_buf->prepareCommandBuffer();
    
    sys_obj->Window.show();
    
    Tempest::IOCommandBufferDescription io_cmd_buf;
    io_cmd_buf.CommandCount = 1;
    auto io_command_buffer = Tempest::CreateIOCommandBuffer(&sys_obj->Backend, io_cmd_buf);

    auto buffer_cpu_storage = Tempest::CreateStorageBuffer(&sys_obj->Backend, Tempest::StorageMode::BufferWrite, ClothSizeX*ClothSizeY*sizeof(Tempest::Vector2));

    BackendType::IOCommandBufferType::IOCommandType io_cmd;
    io_cmd.CommandType = Tempest::IOCommandMode::CopyStorageToBuffer;
    io_cmd.Source.Storage = buffer_cpu_storage.get();
    io_cmd.Destination.Buffer = vertex_buf.get();
    io_cmd.Width = ClothSizeX*ClothSizeY*sizeof(Tempest::Vector2);
    io_cmd.DestinationOffset = 0;
    io_cmd.SourceOffset = 0;
    io_command_buffer->enqueueCommand(io_cmd);

    auto& window = sys_obj->Window;
    window.setEventMask(Tempest::COLLECT_MOUSE_EVENTS|Tempest::COLLECT_WINDOW_EVENTS);

    Tempest::TimeQuery timer;
    auto update_time = timer.time();

    Tempest::WindowSystemEvent wevent;

    std::vector<uint32_t> captured_points;

    bool active = true;

    Tempest::Vector2 mouse_pos{ (float)window.getMouseX(), (float)window.getMouseY() };
    Tempest::Vector2 prev_pos;

    auto idx_space = Tempest::Vector2{ (float)cloth.getSizeX() - 1, (float)cloth.getSizeY() - 1 };

    while(!sys_obj->Window.isDead())
    {
        auto cur_time = timer.time();

        //*
        bool update_buffer = false;
        for(; cur_time - update_time > UpdateStep; update_time += UpdateStep)
        {
            if(captured_points.empty())
                cloth.simulationStep(UpdateStep*1e-6f, nullptr, nullptr);
            else
                cloth.simulationStep(UpdateStep*1e-6f, &captured_points.front(), &captured_points.front() + captured_points.size());
            update_buffer = true;

            // Constraint legal movement
            //*
            for(uint32_t y = 0, yend = cloth.getSizeY(); y < yend; ++y)
            {
                for(uint32_t x = 0, xend = cloth.getSizeX(); x < xend; ++x)
                {
                    auto idx = y*cloth.getSizeX() + x;
                    auto& point = points[idx];
                    auto init_pos = Tempest::Vector2{ (float)x, (float)y };
                    auto dist_origin_vec = point*idx_space - init_pos;
                    auto dist_origin = Length(dist_origin_vec);

                    if(dist_origin > MaxGrabDistance)
                    {
                        auto norm_dist_orign_vec = dist_origin_vec/dist_origin;
                        point = (init_pos + norm_dist_orign_vec*MaxGrabDistance)/idx_space;

                        // Remove outward velocity
                        auto& velocity = velocities[idx];
                        velocities[idx] = velocity - norm_dist_orign_vec*Dot(velocity, norm_dist_orign_vec);
                    }
                }
            }
            //*/
        }
        //*/

		bool swipe_mode = false;
        
        while(window.getEvent(&wevent))
        {
            switch(wevent.Type)
            {
            case Tempest::WindowEventType::MouseButtonPressed:
            {
                if(wevent.MouseButton != Tempest::MouseButtonId::LeftButton || !active)
                    break;
				
				if(swipe_mode)
				{
				}
				else
				{
					prev_pos = Tempest::ToVector2Trunc(mat_inverse * Tempest::Vector3{ (2.0f*mouse_pos.x/wdesc.Width - 1.0f), (1.0f - 2.0f*mouse_pos.y/wdesc.Height), 0.0f })*idx_space;

					Tempest::Vector2 min_corner = Tempest::Vector2Floor(Tempest::Vector2Clamp(prev_pos - Tempest::ToVector2(MaxGrabDistance + GrabRadius),
																							  Tempest::Vector2{ 0.0f, 0.0f }, Tempest::Vector2{ (float)cloth.getSizeX(), (float)cloth.getSizeY() })),
									 max_corner = Tempest::Vector2Ceil(Tempest::Vector2Clamp(prev_pos + Tempest::ToVector2(MaxGrabDistance + GrabRadius),
																							 Tempest::Vector2{ 0.0f, 0.0f }, Tempest::Vector2{ (float)cloth.getSizeX(), (float)cloth.getSizeY() }));
					for(uint32_t y = (uint32_t)min_corner.y, yend = (uint32_t)max_corner.y; y < yend; ++y)
					{
						for(uint32_t x = (uint32_t)min_corner.x, xend = (uint32_t)max_corner.x; x < xend; ++x)
						{
							auto idx = y*cloth.getSizeX() + x;
							auto& point = points[idx];
							if(Length(point*idx_space - prev_pos) < GrabRadius)
							{
								velocities[idx] = {};
								captured_points.push_back(idx);
							}
						}
					}
				}

				window.captureMouse();
            } break;
            case Tempest::WindowEventType::MouseButtonReleased:
            {
                if(wevent.MouseButton != Tempest::MouseButtonId::LeftButton)
                    break;
				window.releaseMouse();

				if(swipe_mode)
				{
				}
				else
				{
					captured_points.clear();
				}
            } break;
            case Tempest::WindowEventType::MouseMoved:
            {
                mouse_pos = Tempest::Vector2{ (float)wevent.MouseMoved.MouseX, (float)wevent.MouseMoved.MouseY };
                if(captured_points.empty() || !active)
                    break;
                auto cur_pos = Tempest::ToVector2Trunc(mat_inverse * Tempest::Vector3{ (2.0f*mouse_pos.x/wdesc.Width - 1.0f), (1.0f - 2.0f*mouse_pos.y/wdesc.Height), 0.0f })*idx_space;

				if(swipe_mode)
				{
				}
				else
				{
					for(auto captured_point : captured_points)
					{
						auto& point = points[captured_point];
						uint32_t x = captured_point % cloth.getSizeX(),
										y = captured_point / cloth.getSizeX();
						auto init_pos = Tempest::Vector2{ (float)x, (float)y };
						auto new_pos = cur_pos - init_pos;
						auto len = Tempest::Length(new_pos);
						if(len > MaxGrabDistance)
						{
							new_pos *= MaxGrabDistance/len;
						}
						point = (new_pos + init_pos)/idx_space;
					}
				}
				prev_pos = cur_pos;
            } break;
            case Tempest::WindowEventType::Focus:
            {
                active = wevent.Enabled != 0;

				if(swipe_mode)
				{
				}
				else
				{
					if(!active)
						captured_points.clear();
				}
            } break;
            }
        }

        buffer_cpu_storage->storeLinearBuffer(0, ClothSizeX*ClothSizeY*sizeof(Tempest::Vector2), points);

        //if(update_buffer)
        {
            sys_obj->Backend.submitCommandBuffer(io_command_buffer.get());
        }

        sys_obj->Backend.clearColorBuffer(0, Tempest::Vector4{0.0f, 0.0f, 0.0f, 0.0f});

        sys_obj->Backend.submitCommandBuffer(command_buf.get());
        
        sys_obj->Window.swapBuffers();
    }

    return EXIT_SUCCESS;
}

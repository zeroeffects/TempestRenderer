#include "tempest/utils/testing.hh"
#include "tempest/math/ode.hh"

const size_t SimStepCount = 1000;
const float SimStep = 5.0e-3f;
const float StartMag = 10.0f;

const float SpringConstant = 1000.0f;
const float Mass = 2.0f;
const float DampConstant = 1.0f;

TGE_TEST("Testing stability of spring computations")
{
    float dt = 1.0f;
    float t = 0.0f;
    float sim_x = StartMag, rk4_sim_x = StartMag;
    float sim_x_deriv = 0.0f, rk4_sim_x_deriv = 0.0f;
    
    float sim_time = SimStep;

    float damping = -DampConstant/(2.0f*Mass);
    float osc_freq = std::sqrt(4.0f*Mass*SpringConstant - DampConstant*DampConstant)/(2.0f*Mass);

    float sim_step = SimStep;

	float euler_total_error = 0.0f,
	      rk4_total_error = 0.0f;

    for(size_t i = 0; i < SimStepCount; ++i, sim_time += SimStep)
    {
        Tempest::EulerODESolver::solveSecondOrderLinearODE(Mass, DampConstant, SpringConstant, 0.0f, SimStep, &sim_x, &sim_x_deriv);

        float expl_x = StartMag*std::exp(damping*sim_time)*std::cos(osc_freq*sim_time);

		float euler_error = std::fabs(sim_x - expl_x);
        TGE_CHECK(std::fabs(sim_x - expl_x) < StartMag*0.1f, "Really bad quality simulation");

		euler_total_error += euler_error;

		Tempest::RK4ODESolver::solveSecondOrderLinearODE(Mass, DampConstant, SpringConstant, 0.0f, SimStep, &rk4_sim_x, &rk4_sim_x_deriv);

		float rk4_error = std::fabs(rk4_sim_x - expl_x);

		rk4_total_error += rk4_error;

		TGE_CHECK(rk4_error <= StartMag*0.1f, "Really bad quality simulation");
    }

	TGE_CHECK(rk4_total_error <= euler_total_error, "Runge-Kutta is not properly implemented");
}
// ---------------------------------------------------------------------
//
// Copyright (c) 2017 - 2024 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

// Config files
#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for basic libMesh objects
#include <libmesh/boundary_info.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_triangle_interface.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <boost/multi_array.hpp>
#include <set>
#include <string>
#include <limits>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>

inline double
kernel(double x)
{
    x += 4.;
    const double x2 = x * x;
    const double x3 = x * x2;
    const double x4 = x * x3;
    const double x5 = x * x4;
    const double x6 = x * x5;
    const double x7 = x * x6;
    if (x <= 0.)
        return 0.;
    else if (x <= 1.)
        return .1984126984126984e-3 * x7;
    else if (x <= 2.)
        return .1111111111111111e-1 * x6 - .1388888888888889e-2 * x7 - .3333333333333333e-1 * x5 +
               .5555555555555556e-1 * x4 - .5555555555555556e-1 * x3 + .3333333333333333e-1 * x2 -
               .1111111111111111e-1 * x + .1587301587301587e-2;
    else if (x <= 3.)
        return .4333333333333333 * x5 - .6666666666666667e-1 * x6 + .4166666666666667e-2 * x7 - 1.500000000000000 * x4 +
               3.055555555555556 * x3 - 3.700000000000000 * x2 + 2.477777777777778 * x - .7095238095238095;
    else if (x <= 4.)
        return 9. * x4 - 1.666666666666667 * x5 + .1666666666666667 * x6 - .6944444444444444e-2 * x7 -
               28.44444444444444 * x3 + 53. * x2 - 54.22222222222222 * x + 23.59047619047619;
    else if (x <= 5.)
        return 96. * x3 - 22.11111111111111 * x4 + 3. * x5 - .2222222222222222 * x6 + .6944444444444444e-2 * x7 -
               245.6666666666667 * x2 + 344. * x - 203.9650793650794;
    else if (x <= 6.)
        return 483.5000000000000 * x2 - 147.0555555555556 * x3 + 26.50000000000000 * x4 - 2.833333333333333 * x5 +
               .1666666666666667 * x6 - .4166666666666667e-2 * x7 - 871.2777777777778 * x + 664.0904761904762;
    else if (x <= 7.)
        return 943.1222222222222 * x - 423.7000000000000 * x2 + 104.9444444444444 * x3 - 15.50000000000000 * x4 +
               1.366666666666667 * x5 - .6666666666666667e-1 * x6 + .1388888888888889e-2 * x7 - 891.1095238095238;
    else if (x <= 8.)
        return 416.1015873015873 - 364.0888888888889 * x + 136.5333333333333 * x2 - 28.44444444444444 * x3 +
               3.555555555555556 * x4 - .2666666666666667 * x5 + .1111111111111111e-1 * x6 - .1984126984126984e-3 * x7;
    else
        return 0.;
} // kernel

static constexpr double PI_VAL = 3.14159265358979323846;

// Elasticity model data.
namespace ModelData
{
// The tether penalty functions each require some data that is set in the
// input file. This data is passed to each object through the void *ctx
// context data pointer. Here we collect all relevant tether data in a struct:
struct Eel2DData
{
    // elastic / penalty
    double c1_s;
    double kappa_s_body;
    double eta_s_body;
    double kappa_s_surface;
    double eta_s_surface;

    // eel kinematics
    double L;          // body length
    double A;          // legacy amplitude scale kept for backward compatibility
    double s_shift;    // legacy head-offset scale kept for backward compatibility
    double omega;      // angular frequency used by the traveling wave
    double k_wave;     // wave number used by the traveling wave
    double wave_frequency_f;
    double wavelength_lambda;
    double wave_phase0;
    std::string envelope_type;
    double quad_a0;
    double quad_a1;
    double quad_a2;
    double cubic_b0;
    double cubic_b1;
    double cubic_b2;
    double cubic_b3;

    // active bending stress
    bool use_active_stress;
    double active_stress_t0;
    double active_phase0;
    double active_ramp_time;
    double active_xi_start;
    double active_xi_end;
    double active_xi_eps;

    // geometry
    double x_leading;  // reference head x
    double y_center0;  // reference body center y
    int n_geom_pts;
    double xhat_head_ref;
    double xhat_tail_ref;
    double L_mesh;
    double ds_mesh;
    std::vector<double> s_mesh_ref;
    std::vector<double> y_center_ref;
    std::vector<double> half_width_ref;

    // 当前 FE 网格在刚体 body frame 下重建得到的中心线缓存
    std::vector<double> x_center_cur_cache;
    std::vector<double> y_center_cur_cache;
    std::vector<double> curvature_cur_cache;
    std::vector<double> axial_strain_cur_cache;
    double cached_centerline_time;
    double cached_centerline_xcom;
    double cached_centerline_ycom;
    double cached_centerline_theta;
    bool cached_centerline_valid;

    // 自由游动位姿（全局平移 + 转动），每步由当前网格几何更新
    double xcom_cur;
    double ycom_cur;
    double theta_cur;

    // 初始参考位姿：body frame 在该位姿下定义
    double xcom_ref;
    double ycom_ref;
    double theta_ref;

    // 参考 body frame 上用于去除刚体模态的局部积分样本
    std::vector<double> xhat_qp_ref;
    std::vector<double> yhat_qp_ref;
    std::vector<double> s_qp_ref;
    std::vector<double> w_qp_ref;
    double area_ref;
    double polar_moment_ref;

    // 给定 time 时，原始行波形变在参考 body frame 上的刚体模态投影
    double cached_mode_time;
    bool cached_mode_valid;
    double disp_tx;
    double disp_ty;
    double disp_rot;
    double vel_tx;
    double vel_ty;
    double vel_rot;

    // explicitly advanced free-swimming rigid-body state
    double body_density;
    double body_mass;
    double body_inertia;
    double u_com_current;
    double v_com_current;
    double omega_body_current;
    double xcom_observed_prev;
    double ycom_observed_prev;
    double theta_observed_prev;
    double actuation_force_x;
    double actuation_force_y;
    double actuation_torque_z;
    double hydrodynamic_force_x;
    double hydrodynamic_force_y;
    double hydrodynamic_torque_z;
    double steady_force_tol;
    double steady_speed_rel_tol;
    double steady_min_cycles;
    bool rigid_dynamics_initialized;

    // tail stiffness controls (stage-1)
    std::string stiffness_mode;
    double tail_xi_start;
    double tail_stiffness_ratio;
    double tail_stiffness_power;
    double tail_k0;
    double tail_a;
    double tail_b;
    bool use_tail_nonlinear_stiffness;

    // diagnostics accumulators
    double diagnostics_prev_time;
    bool diagnostics_initialized;
    double pout_integral;
    double pin_integral;
    double mean_speed_integral;
    double mean_speed_time;
    int alpha_cycle_index;
    double alpha_cycle_max_current;
    double alpha_cycle_max_last;
    bool alpha_cycle_initialized;
    int tail_cycle_index;
    double tail_q_cycle_min;
    double tail_q_cycle_max;
    double tail_peak_to_peak_last;
    double tail_potential_prev;
    bool tail_cycle_initialized;

    Eel2DData(Pointer<Database> input_db)
      : c1_s(input_db->getDouble("C1_S")),
        kappa_s_body(input_db->getDoubleWithDefault("KAPPA_S_BODY",
                                                    input_db->getDoubleWithDefault("KAPPA_S", 0.0))),
        eta_s_body(input_db->getDoubleWithDefault("ETA_S_BODY",
                                                  input_db->getDoubleWithDefault("ETA_S", 0.0))),
        kappa_s_surface(input_db->getDouble("KAPPA_S_SURFACE")),
        eta_s_surface(input_db->getDouble("ETA_S_SURFACE")),
        L(input_db->getDouble("EEL_LENGTH")),
        A(input_db->getDouble("EEL_A")),
        s_shift(input_db->getDouble("EEL_S_SHIFT")),
        omega([&]()
              {
                  const double f_default =
                      input_db->getDoubleWithDefault("EEL_OMEGA", 2.0 * PI_VAL) / (2.0 * PI_VAL);
                  return 2.0 * PI_VAL * input_db->getDoubleWithDefault("WAVE_FREQUENCY_F", f_default);
              }()),
        k_wave([&]()
               {
                   const double k_default = input_db->getDoubleWithDefault("EEL_KWAVE", 2.0 * PI_VAL);
                   const double lambda_default =
                       (std::abs(k_default) > std::numeric_limits<double>::epsilon()) ? 2.0 * PI_VAL / std::abs(k_default) :
                                                                                       input_db->getDouble("EEL_LENGTH");
                   const double lambda = input_db->getDoubleWithDefault("WAVELENGTH_LAMBDA", lambda_default);
                   return 2.0 * PI_VAL / std::max(lambda, std::numeric_limits<double>::epsilon());
               }()),
        wave_frequency_f([&]()
                         {
                             const double f_default =
                                 input_db->getDoubleWithDefault("EEL_OMEGA", 2.0 * PI_VAL) / (2.0 * PI_VAL);
                             return input_db->getDoubleWithDefault("WAVE_FREQUENCY_F", f_default);
                         }()),
        wavelength_lambda([&]()
                          {
                              const double k_default = input_db->getDoubleWithDefault("EEL_KWAVE", 2.0 * PI_VAL);
                              const double lambda_default =
                                  (std::abs(k_default) > std::numeric_limits<double>::epsilon()) ? 2.0 * PI_VAL / std::abs(k_default) :
                                                                                                  input_db->getDouble("EEL_LENGTH");
                              return input_db->getDoubleWithDefault("WAVELENGTH_LAMBDA", lambda_default);
                          }()),
        wave_phase0(input_db->getDoubleWithDefault("WAVE_PHASE0", 0.0)),
        envelope_type(input_db->getStringWithDefault("ENVELOPE_TYPE", "quadratic")),
        quad_a0([&]()
                {
                    const double denom =
                        std::max(input_db->getDouble("EEL_LENGTH") + input_db->getDouble("EEL_S_SHIFT"),
                                 std::numeric_limits<double>::epsilon());
                    const double a_head = input_db->getDouble("EEL_A") * input_db->getDouble("EEL_S_SHIFT") / denom;
                    return input_db->getDoubleWithDefault("QUAD_A0", a_head);
                }()),
        quad_a1(input_db->getDoubleWithDefault("QUAD_A1", 0.0)),
        quad_a2([&]()
                {
                    const double denom =
                        std::max(input_db->getDouble("EEL_LENGTH") + input_db->getDouble("EEL_S_SHIFT"),
                                 std::numeric_limits<double>::epsilon());
                    const double a_head = input_db->getDouble("EEL_A") * input_db->getDouble("EEL_S_SHIFT") / denom;
                    return input_db->getDoubleWithDefault("QUAD_A2", input_db->getDouble("EEL_A") - a_head);
                }()),
        cubic_b0([&]()
                 {
                     const double denom =
                         std::max(input_db->getDouble("EEL_LENGTH") + input_db->getDouble("EEL_S_SHIFT"),
                                  std::numeric_limits<double>::epsilon());
                     const double a_head = input_db->getDouble("EEL_A") * input_db->getDouble("EEL_S_SHIFT") / denom;
                     return input_db->getDoubleWithDefault("CUBIC_B0", a_head);
                 }()),
        cubic_b1(input_db->getDoubleWithDefault("CUBIC_B1", 0.0)),
        cubic_b2(input_db->getDoubleWithDefault("CUBIC_B2", 0.0)),
        cubic_b3([&]()
                 {
                     const double denom =
                         std::max(input_db->getDouble("EEL_LENGTH") + input_db->getDouble("EEL_S_SHIFT"),
                                  std::numeric_limits<double>::epsilon());
                     const double a_head = input_db->getDouble("EEL_A") * input_db->getDouble("EEL_S_SHIFT") / denom;
                     return input_db->getDoubleWithDefault("CUBIC_B3", input_db->getDouble("EEL_A") - a_head);
                 }()),
        use_active_stress(input_db->getBoolWithDefault("USE_ACTIVE_STRESS", true)),
        active_stress_t0(input_db->getDoubleWithDefault("ACTIVE_STRESS_T0",
                                                        input_db->getDouble("KAPPA_S_BODY") *
                                                            input_db->getDouble("EEL_A"))),
        active_phase0(input_db->getDoubleWithDefault("ACTIVE_PHASE0", 0.0)),
        active_ramp_time(input_db->getDoubleWithDefault("ACTIVE_RAMP_TIME", 0.0)),
        active_xi_start(input_db->getDoubleWithDefault("ACTIVE_XI_START", 0.0)),
        active_xi_end(input_db->getDoubleWithDefault("ACTIVE_XI_END", 1.0)),
        active_xi_eps(input_db->getDoubleWithDefault("ACTIVE_XI_EPS", 0.02)),
        x_leading(input_db->getDouble("EEL_X_LEADING")),
        y_center0(input_db->getDouble("EEL_Y_CENTER0")),
        n_geom_pts(input_db->getInteger("EEL_NUM_S_POINTS")),
        xhat_head_ref(0.0),
        xhat_tail_ref(0.0),
        L_mesh(0.0),
        ds_mesh(0.0),
        cached_centerline_time(0.0), cached_centerline_xcom(0.0), cached_centerline_ycom(0.0),
        cached_centerline_theta(0.0), cached_centerline_valid(false),
        xcom_cur(0.0), ycom_cur(0.0), theta_cur(0.0),
        xcom_ref(0.0), ycom_ref(0.0), theta_ref(0.0),
        area_ref(0.0), polar_moment_ref(0.0),
        cached_mode_time(0.0), cached_mode_valid(false),
        disp_tx(0.0), disp_ty(0.0), disp_rot(0.0),
        vel_tx(0.0), vel_ty(0.0), vel_rot(0.0),
        body_density(input_db->getDoubleWithDefault("BODY_DENSITY",
                                                    input_db->getDoubleWithDefault("RHO", 1.0))),
        body_mass(0.0),
        body_inertia(0.0),
        u_com_current(0.0), v_com_current(0.0), omega_body_current(0.0),
        xcom_observed_prev(0.0), ycom_observed_prev(0.0), theta_observed_prev(0.0),
        actuation_force_x(0.0), actuation_force_y(0.0), actuation_torque_z(0.0),
        hydrodynamic_force_x(0.0), hydrodynamic_force_y(0.0), hydrodynamic_torque_z(0.0),
        steady_force_tol(input_db->getDoubleWithDefault("STEADY_FORCE_TOL", 1.0e-4)),
        steady_speed_rel_tol(input_db->getDoubleWithDefault("STEADY_SPEED_REL_TOL", 5.0e-2)),
        steady_min_cycles(input_db->getDoubleWithDefault("STEADY_MIN_CYCLES", 2.0)),
        rigid_dynamics_initialized(false),
        stiffness_mode(input_db->getStringWithDefault("STIFFNESS_MODE", "uniform_linear")),
        tail_xi_start(input_db->getDoubleWithDefault("TAIL_XI_START", 0.7)),
        tail_stiffness_ratio(input_db->getDoubleWithDefault("TAIL_STIFFNESS_RATIO", 1.0)),
        tail_stiffness_power(input_db->getDoubleWithDefault("TAIL_STIFFNESS_POWER", 1.0)),
        tail_k0(input_db->getDoubleWithDefault("TAIL_NONLINEAR_K0", input_db->getDouble("C1_S"))),
        tail_a(input_db->getDoubleWithDefault("TAIL_NONLINEAR_A", 0.0)),
        tail_b(input_db->getDoubleWithDefault("TAIL_NONLINEAR_B", 0.0)),
        use_tail_nonlinear_stiffness(input_db->getBoolWithDefault("USE_TAIL_NONLINEAR_STIFFNESS", false)),
        diagnostics_prev_time(0.0),
        diagnostics_initialized(false),
        pout_integral(0.0), pin_integral(0.0),
        mean_speed_integral(0.0), mean_speed_time(0.0),
        alpha_cycle_index(0), alpha_cycle_max_current(0.0), alpha_cycle_max_last(0.0),
        alpha_cycle_initialized(false),
        tail_cycle_index(0), tail_q_cycle_min(0.0), tail_q_cycle_max(0.0),
        tail_peak_to_peak_last(0.0), tail_potential_prev(0.0),
        tail_cycle_initialized(false)
    {}
};

static Eel2DData* s_eel_data_state = nullptr;
static EquationSystems* s_equation_systems_state = nullptr;
static MeshBase* s_mesh_state = nullptr;
static std::string s_coords_system_name_state;

void update_current_centerline_cache_from_global_state(double time, Eel2DData& d);
inline double current_centerline_curvature(double s_mesh, const Eel2DData& d);

inline double
clamp_value(double x, double a, double b)
{
    return std::max(a, std::min(x, b));
}

inline double
wrap_angle(double angle)
{
    const double two_pi = 2.0 * PI_VAL;
    while (angle > PI_VAL) angle -= two_pi;
    while (angle < -PI_VAL) angle += two_pi;
    return angle;
}

inline double body_half_width(double s, const Eel2DData& d)
{
    if (d.half_width_ref.empty()) return 0.0;
    if (d.half_width_ref.size() == 1 || d.L_mesh <= std::numeric_limits<double>::epsilon()) return d.half_width_ref.front();

    const double s_mesh = clamp_value(s, 0.0, d.L_mesh);
    const double r = s_mesh / d.ds_mesh;
    const std::size_t i =
        std::min<std::size_t>(d.half_width_ref.size() - 2, static_cast<std::size_t>(std::floor(r)));
    const double a = r - static_cast<double>(i);
    return std::max(0.0, (1.0 - a) * d.half_width_ref[i] + a * d.half_width_ref[i + 1]);
}

inline double
interpolate_body_grid_values(double s_mesh, const std::vector<double>& values, const Eel2DData& d)
{
    if (values.empty()) return 0.0;
    if (values.size() == 1 || d.L_mesh <= std::numeric_limits<double>::epsilon()) return values.front();

    const double s = clamp_value(s_mesh, 0.0, d.L_mesh);
    const double r = s / d.ds_mesh;
    const std::size_t i =
        std::min<std::size_t>(values.size() - 2, static_cast<std::size_t>(std::floor(r)));
    const double a = r - static_cast<double>(i);
    return (1.0 - a) * values[i] + a * values[i + 1];
}

inline double
mesh_centerline_y(double s_mesh, const Eel2DData& d)
{
    return interpolate_body_grid_values(s_mesh, d.y_center_ref, d);
}

inline double
mesh_to_wave_scale(const Eel2DData& d)
{
    if (d.L_mesh <= std::numeric_limits<double>::epsilon()) return 0.0;
    return d.L / d.L_mesh;
}

inline double
mesh_s_to_wave_s(double s_mesh, const Eel2DData& d)
{
    if (d.L_mesh <= std::numeric_limits<double>::epsilon()) return 0.0;
    const double xi = clamp_value(s_mesh / d.L_mesh, 0.0, 1.0);
    return d.L * xi;
}

inline void
compute_body_coordinates(double xhat_ref, double yhat_ref, const Eel2DData& d, double& s_mesh, double& eta_ref)
{
    s_mesh = clamp_value(xhat_ref - d.xhat_head_ref, 0.0, d.L_mesh);
    eta_ref = yhat_ref - mesh_centerline_y(s_mesh, d);
}

inline double
fallback_envelope_head_amplitude(const Eel2DData& d)
{
    const double denom = std::max(d.L + d.s_shift, std::numeric_limits<double>::epsilon());
    return std::max(0.0, d.A * d.s_shift / denom);
}

inline double
fallback_envelope_tail_amplitude(const Eel2DData& d)
{
    return std::max(fallback_envelope_head_amplitude(d) + 1.0e-8, std::max(0.0, d.A));
}

inline double
evaluate_quadratic_envelope(double xi, const Eel2DData& d)
{
    return d.quad_a0 + d.quad_a1 * xi + d.quad_a2 * xi * xi;
}

inline double
evaluate_quadratic_envelope_dxi(double xi, const Eel2DData& d)
{
    return d.quad_a1 + 2.0 * d.quad_a2 * xi;
}

inline double
evaluate_cubic_envelope(double xi, const Eel2DData& d)
{
    return d.cubic_b0 + d.cubic_b1 * xi + d.cubic_b2 * xi * xi + d.cubic_b3 * xi * xi * xi;
}

inline double
evaluate_cubic_envelope_dxi(double xi, const Eel2DData& d)
{
    return d.cubic_b1 + 2.0 * d.cubic_b2 * xi + 3.0 * d.cubic_b3 * xi * xi;
}

inline double
amplitude_envelope(double s, const Eel2DData& d)
{
    // Traveling-wave body kinematics follows the literature form
    // y = A(s) sin(omega t - k s), with the amplitude envelope modeled
    // separately. Quadratic/cubic options follow BCF/thunniform practice.
    const double s_wave = mesh_s_to_wave_s(s, d);
    const double xi =
        (d.L > std::numeric_limits<double>::epsilon()) ? clamp_value(s_wave / d.L, 0.0, 1.0) : 0.0;

    const bool use_cubic = (d.envelope_type == "cubic");
    const double a_raw = use_cubic ? evaluate_cubic_envelope(xi, d) : evaluate_quadratic_envelope(xi, d);
    const double a_head_raw = use_cubic ? evaluate_cubic_envelope(0.0, d) : evaluate_quadratic_envelope(0.0, d);
    const double a_tail_raw = use_cubic ? evaluate_cubic_envelope(1.0, d) : evaluate_quadratic_envelope(1.0, d);

    if (a_tail_raw > a_head_raw + 1.0e-8) return std::max(0.0, a_raw);

    const double a_head = fallback_envelope_head_amplitude(d);
    const double a_tail = fallback_envelope_tail_amplitude(d);
    return a_head + (a_tail - a_head) * xi * xi;
}

inline double
amplitude_envelope_ds(double s, const Eel2DData& d)
{
    const double s_wave = mesh_s_to_wave_s(s, d);
    const double xi =
        (d.L > std::numeric_limits<double>::epsilon()) ? clamp_value(s_wave / d.L, 0.0, 1.0) : 0.0;
    const bool use_cubic = (d.envelope_type == "cubic");
    const double a_head_raw = use_cubic ? evaluate_cubic_envelope(0.0, d) : evaluate_quadratic_envelope(0.0, d);
    const double a_tail_raw = use_cubic ? evaluate_cubic_envelope(1.0, d) : evaluate_quadratic_envelope(1.0, d);

    if (a_tail_raw > a_head_raw + 1.0e-8)
    {
        const double dA_dxi =
            use_cubic ? evaluate_cubic_envelope_dxi(xi, d) : evaluate_quadratic_envelope_dxi(xi, d);
        return dA_dxi / std::max(d.L_mesh, std::numeric_limits<double>::epsilon());
    }

    const double a_head = fallback_envelope_head_amplitude(d);
    const double a_tail = fallback_envelope_tail_amplitude(d);
    return 2.0 * (a_tail - a_head) * xi / std::max(d.L_mesh, std::numeric_limits<double>::epsilon());
}

inline double
body_wave_y(double s, double t, const Eel2DData& d)
{
    // Traveling-wave body kinematics follows the literature form
    // y = A(s) sin(omega t - k s + phase0).
    const double s_wave = mesh_s_to_wave_s(s, d);
    const double phase = d.omega * t - d.k_wave * s_wave + d.wave_phase0;
    return amplitude_envelope(s, d) * std::sin(phase);
}

inline double
body_wave_v(double s, double t, const Eel2DData& d)
{
    // Time derivative of the traveling wave:
    // dy/dt = omega A(s) cos(omega t - k s + phase0).
    const double s_wave = mesh_s_to_wave_s(s, d);
    const double phase = d.omega * t - d.k_wave * s_wave + d.wave_phase0;
    return d.omega * amplitude_envelope(s, d) * std::cos(phase);
}

inline double
body_wave_dy_ds(double s, double t, const Eel2DData& d)
{
    const double s_wave = mesh_s_to_wave_s(s, d);
    const double phase = d.omega * t - d.k_wave * s_wave + d.wave_phase0;
    return amplitude_envelope_ds(s, d) * std::sin(phase) -
           d.k_wave * mesh_to_wave_scale(d) * amplitude_envelope(s, d) * std::cos(phase);
}

inline bool is_tail_region(double s, const Eel2DData& d)
{
    if (d.L_mesh <= std::numeric_limits<double>::epsilon()) return false;
    const double xi = clamp_value(s / d.L_mesh, 0.0, 1.0);
    return xi >= d.tail_xi_start;
}

inline double
tail_nonlinear_restoring_force(double q, double k0, double a, double b)
{
    const double abs_q = std::abs(q);
    return k0 * (q + a * q * abs_q + b * q * q * q);
}

inline double
tail_equivalent_stiffness(double q, double k0, double a, double b)
{
    return std::max(0.0, k0 * (1.0 + 2.0 * a * std::abs(q) + 3.0 * b * q * q));
}

inline double
tail_potential_energy(double q, double k0, double a, double b)
{
    const double abs_q = std::abs(q);
    return 0.5 * k0 * q * q + (k0 * a / 3.0) * abs_q * abs_q * abs_q + 0.25 * k0 * b * q * q * q * q;
}

inline double
stiffness_distribution(double s, const Eel2DData& d)
{
    if (d.stiffness_mode == "tail_gradient")
    {
        if (d.L_mesh <= std::numeric_limits<double>::epsilon()) return 1.0;
        const double xi = clamp_value(s / d.L_mesh, 0.0, 1.0);
        if (xi <= d.tail_xi_start) return 1.0;

        const double denom = std::max(1.0 - d.tail_xi_start, std::numeric_limits<double>::epsilon());
        const double eta = clamp_value((xi - d.tail_xi_start) / denom, 0.0, 1.0);
        return 1.0 + (d.tail_stiffness_ratio - 1.0) * std::pow(eta, std::max(0.0, d.tail_stiffness_power));
    }

    // Stage-1 default: linear uniform stiffness.
    return 1.0;
}

inline double
compute_effective_tail_deformation(double transverse_deflection,
                                   double axial_strain,
                                   double curvature_like,
                                   double s,
                                   const Eel2DData& d)
{
    const double h = std::max(body_half_width(s, d), 0.0);
    const double q_stretch = d.L_mesh * axial_strain;
    const double q_bend = h * h * curvature_like;
    return std::sqrt(transverse_deflection * transverse_deflection + q_stretch * q_stretch + q_bend * q_bend);
}

inline void
compute_actual_deformation_measures(const libMesh::Point& x,
                                    const TensorValue<double>& FF,
                                    double s_ref,
                                    double eta_ref,
                                    double time,
                                    const Eel2DData& d,
                                    double& transverse_deflection,
                                    double& axial_strain,
                                    double& curvature_like)
{
    update_current_centerline_cache_from_global_state(time, const_cast<Eel2DData&>(d));

    const double ct = std::cos(d.theta_cur);
    const double st = std::sin(d.theta_cur);

    const double dx_cur = x(0) - d.xcom_cur;
    const double dy_cur = x(1) - d.ycom_cur;
    const double xhat_cur = ct * dx_cur + st * dy_cur;
    const double yhat_cur = -st * dx_cur + ct * dy_cur;

    double s_cur = 0.0, eta_cur = 0.0;
    compute_body_coordinates(xhat_cur, yhat_cur, d, s_cur, eta_cur);
    transverse_deflection = eta_cur - eta_ref;

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);
    libMesh::VectorValue<double> A0_ref;
    A0_ref(0) = cr;
    A0_ref(1) = sr;
#if (NDIM == 3)
    A0_ref(2) = 0.0;
#endif

    const auto FA0 = FF * A0_ref;
    const double a_body_x = ct * FA0(0) + st * FA0(1);
    const double a_body_y = -st * FA0(0) + ct * FA0(1);
    const double lambda_f = std::sqrt(a_body_x * a_body_x + a_body_y * a_body_y);
    axial_strain = lambda_f - 1.0;

    curvature_like = current_centerline_curvature(s_ref, d);
}

inline double
compute_local_modulus(double s,
                      double transverse_deflection,
                      double axial_strain,
                      double curvature_like,
                      const Eel2DData& d)
{
    double scale = stiffness_distribution(s, d);

    // Modulate the tail stiffness with actual FE deformation measures instead
    // of the prescribed kinematic wave alone.
    if (d.use_tail_nonlinear_stiffness && is_tail_region(s, d))
    {
        const double k_ref = std::max(d.tail_k0, std::numeric_limits<double>::epsilon());
        const double q_eff =
            compute_effective_tail_deformation(transverse_deflection, axial_strain, curvature_like, s, d);
        scale *= tail_equivalent_stiffness(q_eff, d.tail_k0, d.tail_a, d.tail_b) / k_ref;
    }

    return std::max(d.c1_s * scale, 1.0e-12);
}

inline double smooth_step_tanh(double x, double eps)
{
    if (eps <= 0.0) return (x >= 0.0) ? 1.0 : 0.0;
    return 0.5 * (1.0 + std::tanh(x / eps));
}

inline double mask_interval(double xi, double a, double b, double eps)
{
    const double m1 = smooth_step_tanh(xi - a, eps);
    const double m2 = smooth_step_tanh(b - xi, eps);
    return m1 * m2;
}

inline double ramp_factor(double time, double ramp_time)
{
    if (ramp_time <= 0.0) return 1.0;
    return std::min(1.0, std::max(0.0, time / ramp_time));
}

inline void eel_raw_body_frame_kinematics(double s,
                                          double time,
                                          const Eel2DData& d,
                                          double& disp_x,
                                          double& disp_y,
                                          double& vel_x,
                                          double& vel_y)
{
    disp_x = 0.0;
    disp_y = body_wave_y(s, time, d);
    vel_x = 0.0;
    vel_y = body_wave_v(s, time, d);
}

void initialize_reference_projection_data(EquationSystems* equation_systems,
                                          const std::string& coords_system_name,
                                          MeshBase& mesh,
                                          Eel2DData& d)
{
    d.xhat_qp_ref.clear();
    d.yhat_qp_ref.clear();
    d.s_qp_ref.clear();
    d.w_qp_ref.clear();

    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    std::vector<std::vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, SEVENTH);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<vector<double> >& phi = fe->get_phi();

    boost::multi_array<double, 2> X_node;
    VectorValue<double> x;

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);

    double area_local = 0.0;
    double polar_moment_local = 0.0;

    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d_idx = 0; d_idx < NDIM; ++d_idx) dof_map.dof_indices(elem, dof_indices[d_idx], d_idx);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);

            const double dx = x(0) - d.xcom_ref;
            const double dy = x(1) - d.ycom_ref;
            const double xhat = cr * dx + sr * dy;
            const double yhat = -sr * dx + cr * dy;
            double s = 0.0, eta = 0.0;
            compute_body_coordinates(xhat, yhat, d, s, eta);
            const double w = JxW[qp];

            d.xhat_qp_ref.push_back(xhat);
            d.yhat_qp_ref.push_back(yhat);
            d.s_qp_ref.push_back(s);
            d.w_qp_ref.push_back(w);

            area_local += w;
            polar_moment_local += (xhat * xhat + yhat * yhat) * w;
        }
    }

    double global_vals[2] = { area_local, polar_moment_local };
    IBTK_MPI::sumReduction(global_vals, 2);
    d.area_ref = global_vals[0];
    d.polar_moment_ref = global_vals[1];
    d.cached_mode_valid = false;
}

void initialize_reference_body_geometry(MeshBase& mesh, Eel2DData& d)
{
    BoundaryInfo& boundary_info = mesh.get_boundary_info();
    boundary_info.build_node_list_from_side_list();

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);

    double xhat_head_local = std::numeric_limits<double>::max();
    double xhat_tail_local = -std::numeric_limits<double>::max();

    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;

        const double dx = (*node)(0) - d.xcom_ref;
        const double dy = (*node)(1) - d.ycom_ref;
        const double xhat = cr * dx + sr * dy;

        xhat_head_local = std::min(xhat_head_local, xhat);
        xhat_tail_local = std::max(xhat_tail_local, xhat);
    }

    d.xhat_head_ref = xhat_head_local;
    d.xhat_tail_ref = xhat_tail_local;
    IBTK_MPI::minReduction(&d.xhat_head_ref, 1);
    IBTK_MPI::maxReduction(&d.xhat_tail_ref, 1);

    d.L_mesh = std::max(d.xhat_tail_ref - d.xhat_head_ref, std::numeric_limits<double>::epsilon());
    d.n_geom_pts = std::max(8, d.n_geom_pts);
    d.ds_mesh = d.L_mesh / static_cast<double>(d.n_geom_pts - 1);

    d.s_mesh_ref.resize(d.n_geom_pts);
    d.y_center_ref.assign(d.n_geom_pts, 0.0);
    d.half_width_ref.assign(d.n_geom_pts, 0.0);

    std::vector<double> ymax_local(d.n_geom_pts, -std::numeric_limits<double>::max());
    std::vector<double> ymin_local(d.n_geom_pts, std::numeric_limits<double>::max());
    std::vector<double> count_local(d.n_geom_pts, 0.0);

    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;

        const double dx = (*node)(0) - d.xcom_ref;
        const double dy = (*node)(1) - d.ycom_ref;
        const double xhat = cr * dx + sr * dy;
        const double yhat = -sr * dx + cr * dy;

        const double s_mesh = clamp_value(xhat - d.xhat_head_ref, 0.0, d.L_mesh);
        const int k = std::max(
            0, std::min(d.n_geom_pts - 1, static_cast<int>(std::llround(s_mesh / d.ds_mesh))));

        ymax_local[k] = std::max(ymax_local[k], yhat);
        ymin_local[k] = std::min(ymin_local[k], yhat);
        count_local[k] += 1.0;
    }

    IBTK_MPI::maxReduction(ymax_local.data(), d.n_geom_pts);
    IBTK_MPI::minReduction(ymin_local.data(), d.n_geom_pts);
    IBTK_MPI::sumReduction(count_local.data(), d.n_geom_pts);

    for (int k = 0; k < d.n_geom_pts; ++k)
    {
        if (count_local[k] > 0.5) continue;

        int kl = k - 1;
        while (kl >= 0 && count_local[kl] < 0.5) --kl;
        int kr = k + 1;
        while (kr < d.n_geom_pts && count_local[kr] < 0.5) ++kr;

        if (kl >= 0 && kr < d.n_geom_pts)
        {
            const double a = static_cast<double>(k - kl) / static_cast<double>(kr - kl);
            ymax_local[k] = (1.0 - a) * ymax_local[kl] + a * ymax_local[kr];
            ymin_local[k] = (1.0 - a) * ymin_local[kl] + a * ymin_local[kr];
        }
        else if (kl >= 0)
        {
            ymax_local[k] = ymax_local[kl];
            ymin_local[k] = ymin_local[kl];
        }
        else if (kr < d.n_geom_pts)
        {
            ymax_local[k] = ymax_local[kr];
            ymin_local[k] = ymin_local[kr];
        }
    }

    for (int k = 0; k < d.n_geom_pts; ++k)
    {
        d.s_mesh_ref[k] = static_cast<double>(k) * d.ds_mesh;
        d.y_center_ref[k] = 0.5 * (ymax_local[k] + ymin_local[k]);
        d.half_width_ref[k] = std::max(0.0, 0.5 * (ymax_local[k] - ymin_local[k]));
    }

    d.cached_mode_valid = false;
    d.cached_centerline_valid = false;
}

void update_current_centerline_cache(EquationSystems* equation_systems,
                                     const std::string& coords_system_name,
                                     MeshBase& mesh,
                                     double xcom_frame,
                                     double ycom_frame,
                                     double theta_frame,
                                     double time,
                                     Eel2DData& d)
{
    const double tol_time = 1.0e-12 * std::max(1.0, std::abs(time));
    const double tol_pose = 1.0e-12;
    if (d.cached_centerline_valid && std::abs(time - d.cached_centerline_time) <= tol_time &&
        std::abs(xcom_frame - d.cached_centerline_xcom) <= tol_pose &&
        std::abs(ycom_frame - d.cached_centerline_ycom) <= tol_pose &&
        std::abs(wrap_angle(theta_frame - d.cached_centerline_theta)) <= tol_pose)
    {
        return;
    }

    BoundaryInfo& boundary_info = mesh.get_boundary_info();
    boundary_info.build_node_list_from_side_list();

    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    const double cr_ref = std::cos(d.theta_ref);
    const double sr_ref = std::sin(d.theta_ref);
    const double ct = std::cos(theta_frame);
    const double st = std::sin(theta_frame);

    std::vector<double> ymax_local(d.n_geom_pts, -std::numeric_limits<double>::max());
    std::vector<double> ymin_local(d.n_geom_pts, std::numeric_limits<double>::max());
    std::vector<double> xsum_local(d.n_geom_pts, 0.0);
    std::vector<double> count_local(d.n_geom_pts, 0.0);

    std::vector<dof_id_type> dof_idx_x, dof_idx_y;
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;

        const double dx_ref = (*node)(0) - d.xcom_ref;
        const double dy_ref = (*node)(1) - d.ycom_ref;
        const double xhat_ref = cr_ref * dx_ref + sr_ref * dy_ref;
        const double s_mesh = clamp_value(xhat_ref - d.xhat_head_ref, 0.0, d.L_mesh);
        const int k = std::max(0, std::min(d.n_geom_pts - 1, static_cast<int>(std::llround(s_mesh / d.ds_mesh))));

        dof_map.dof_indices(node, dof_idx_x, 0);
        dof_map.dof_indices(node, dof_idx_y, 1);
        const double x_cur = (*X_ghost_vec)(dof_idx_x[0]);
        const double y_cur = (*X_ghost_vec)(dof_idx_y[0]);

        const double dx_cur = x_cur - xcom_frame;
        const double dy_cur = y_cur - ycom_frame;
        const double xhat_cur = ct * dx_cur + st * dy_cur;
        const double yhat_cur = -st * dx_cur + ct * dy_cur;

        ymax_local[k] = std::max(ymax_local[k], yhat_cur);
        ymin_local[k] = std::min(ymin_local[k], yhat_cur);
        xsum_local[k] += xhat_cur;
        count_local[k] += 1.0;
    }

    IBTK_MPI::maxReduction(ymax_local.data(), d.n_geom_pts);
    IBTK_MPI::minReduction(ymin_local.data(), d.n_geom_pts);
    IBTK_MPI::sumReduction(xsum_local.data(), d.n_geom_pts);
    IBTK_MPI::sumReduction(count_local.data(), d.n_geom_pts);

    d.x_center_cur_cache.assign(d.n_geom_pts, 0.0);
    d.y_center_cur_cache.assign(d.n_geom_pts, 0.0);
    for (int k = 0; k < d.n_geom_pts; ++k)
    {
        if (count_local[k] > 0.5)
        {
            d.x_center_cur_cache[k] = xsum_local[k] / count_local[k];
            d.y_center_cur_cache[k] = 0.5 * (ymax_local[k] + ymin_local[k]);
            continue;
        }

        int kl = k - 1;
        while (kl >= 0 && count_local[kl] < 0.5) --kl;
        int kr = k + 1;
        while (kr < d.n_geom_pts && count_local[kr] < 0.5) ++kr;

        if (kl >= 0 && kr < d.n_geom_pts)
        {
            const double a = static_cast<double>(k - kl) / static_cast<double>(kr - kl);
            d.x_center_cur_cache[k] = (1.0 - a) * (xsum_local[kl] / count_local[kl]) + a * (xsum_local[kr] / count_local[kr]);
            const double y_left = 0.5 * (ymax_local[kl] + ymin_local[kl]);
            const double y_right = 0.5 * (ymax_local[kr] + ymin_local[kr]);
            d.y_center_cur_cache[k] = (1.0 - a) * y_left + a * y_right;
        }
        else if (kl >= 0)
        {
            d.x_center_cur_cache[k] = xsum_local[kl] / count_local[kl];
            d.y_center_cur_cache[k] = 0.5 * (ymax_local[kl] + ymin_local[kl]);
        }
        else if (kr < d.n_geom_pts)
        {
            d.x_center_cur_cache[k] = xsum_local[kr] / count_local[kr];
            d.y_center_cur_cache[k] = 0.5 * (ymax_local[kr] + ymin_local[kr]);
        }
    }

    d.curvature_cur_cache.assign(d.n_geom_pts, 0.0);
    d.axial_strain_cur_cache.assign(d.n_geom_pts, 0.0);
    if (d.n_geom_pts >= 2)
    {
        std::vector<double> seg_theta(d.n_geom_pts - 1, 0.0);
        std::vector<double> seg_len(d.n_geom_pts - 1, d.ds_mesh);
        std::vector<double> seg_strain(d.n_geom_pts - 1, 0.0);
        for (int k = 0; k < d.n_geom_pts - 1; ++k)
        {
            const double dx_seg = d.x_center_cur_cache[k + 1] - d.x_center_cur_cache[k];
            const double dy_seg = d.y_center_cur_cache[k + 1] - d.y_center_cur_cache[k];
            seg_len[k] = std::max(std::sqrt(dx_seg * dx_seg + dy_seg * dy_seg), std::numeric_limits<double>::epsilon());
            seg_theta[k] = std::atan2(dy_seg, dx_seg);
            seg_strain[k] = seg_len[k] / std::max(d.ds_mesh, std::numeric_limits<double>::epsilon()) - 1.0;
        }

        d.axial_strain_cur_cache.front() = seg_strain.front();
        d.axial_strain_cur_cache.back() = seg_strain.back();
        for (int k = 1; k < d.n_geom_pts - 1; ++k)
        {
            d.axial_strain_cur_cache[k] = 0.5 * (seg_strain[k - 1] + seg_strain[k]);
        }

        if (d.n_geom_pts >= 3)
        {
            d.curvature_cur_cache.front() = wrap_angle(seg_theta[1] - seg_theta[0]) /
                                            std::max(0.5 * (seg_len[0] + seg_len[1]), std::numeric_limits<double>::epsilon());
            d.curvature_cur_cache.back() = wrap_angle(seg_theta[d.n_geom_pts - 2] - seg_theta[d.n_geom_pts - 3]) /
                                           std::max(0.5 * (seg_len[d.n_geom_pts - 2] + seg_len[d.n_geom_pts - 3]),
                                                    std::numeric_limits<double>::epsilon());
            for (int k = 1; k < d.n_geom_pts - 1; ++k)
            {
                d.curvature_cur_cache[k] = wrap_angle(seg_theta[k] - seg_theta[k - 1]) /
                                           std::max(0.5 * (seg_len[k] + seg_len[k - 1]),
                                                    std::numeric_limits<double>::epsilon());
            }
        }
    }

    d.cached_centerline_time = time;
    d.cached_centerline_xcom = xcom_frame;
    d.cached_centerline_ycom = ycom_frame;
    d.cached_centerline_theta = theta_frame;
    d.cached_centerline_valid = true;
}

void update_current_centerline_cache_from_global_state(double time, Eel2DData& d)
{
    if (!s_equation_systems_state || !s_mesh_state || s_coords_system_name_state.empty()) return;
    update_current_centerline_cache(s_equation_systems_state,
                                    s_coords_system_name_state,
                                    *s_mesh_state,
                                    d.xcom_cur,
                                    d.ycom_cur,
                                    d.theta_cur,
                                    time,
                                    d);
}

inline double
current_centerline_x(double s_mesh, const Eel2DData& d)
{
    return interpolate_body_grid_values(s_mesh, d.x_center_cur_cache, d);
}

inline double
current_centerline_y(double s_mesh, const Eel2DData& d)
{
    return interpolate_body_grid_values(s_mesh, d.y_center_cur_cache, d);
}

inline double
current_centerline_curvature(double s_mesh, const Eel2DData& d)
{
    return interpolate_body_grid_values(s_mesh, d.curvature_cur_cache, d);
}

inline double
current_centerline_axial_strain(double s_mesh, const Eel2DData& d)
{
    return interpolate_body_grid_values(s_mesh, d.axial_strain_cur_cache, d);
}

void update_projected_kinematics_cache(double time, Eel2DData& d)
{
    const double tol = 1.0e-12 * std::max(1.0, std::abs(time));
    if (d.cached_mode_valid && std::abs(time - d.cached_mode_time) <= tol) return;

    double local_sums[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const std::size_t n_qp = d.s_qp_ref.size();
    for (std::size_t k = 0; k < n_qp; ++k)
    {
        double disp_x_raw, disp_y_raw, vel_x_raw, vel_y_raw;
        eel_raw_body_frame_kinematics(d.s_qp_ref[k], time, d, disp_x_raw, disp_y_raw, vel_x_raw, vel_y_raw);

        const double xhat = d.xhat_qp_ref[k];
        const double yhat = d.yhat_qp_ref[k];
        const double w = d.w_qp_ref[k];

        local_sums[0] += disp_x_raw * w;
        local_sums[1] += disp_y_raw * w;
        local_sums[2] += (xhat * disp_y_raw - yhat * disp_x_raw) * w;
        local_sums[3] += vel_x_raw * w;
        local_sums[4] += vel_y_raw * w;
        local_sums[5] += (xhat * vel_y_raw - yhat * vel_x_raw) * w;
    }

    IBTK_MPI::sumReduction(local_sums, 6);

    const double area = std::max(d.area_ref, std::numeric_limits<double>::epsilon());
    const double polar_moment = std::max(d.polar_moment_ref, std::numeric_limits<double>::epsilon());

    d.disp_tx = local_sums[0] / area;
    d.disp_ty = local_sums[1] / area;
    d.disp_rot = local_sums[2] / polar_moment;
    d.vel_tx = local_sums[3] / area;
    d.vel_ty = local_sums[4] / area;
    d.vel_rot = local_sums[5] / polar_moment;

    d.cached_mode_time = time;
    d.cached_mode_valid = true;
}

void initialize_rigid_body_dynamics(Eel2DData& d)
{
    d.body_mass = std::max(d.body_density * d.area_ref, std::numeric_limits<double>::epsilon());
    d.body_inertia = std::max(d.body_density * d.polar_moment_ref, std::numeric_limits<double>::epsilon());
    d.u_com_current = 0.0;
    d.v_com_current = 0.0;
    d.omega_body_current = 0.0;
    d.xcom_observed_prev = d.xcom_cur;
    d.ycom_observed_prev = d.ycom_cur;
    d.theta_observed_prev = d.theta_cur;
    d.actuation_force_x = 0.0;
    d.actuation_force_y = 0.0;
    d.actuation_torque_z = 0.0;
    d.hydrodynamic_force_x = 0.0;
    d.hydrodynamic_force_y = 0.0;
    d.hydrodynamic_torque_z = 0.0;
    d.rigid_dynamics_initialized = true;
}

void compute_hydrodynamic_force_and_torque(double xcom_observed,
                                           double ycom_observed,
                                           double theta_observed,
                                           double actuation_fx,
                                           double actuation_fy,
                                           double actuation_torque_z,
                                           double dt,
                                           Eel2DData& d,
                                           double& fx,
                                           double& fy,
                                           double& torque_z)
{
    fx = 0.0;
    fy = 0.0;
    torque_z = 0.0;

    if (!d.rigid_dynamics_initialized || dt <= std::numeric_limits<double>::epsilon()) return;

    // Infer the fluid reaction from the resolved FE acceleration after removing
    // the explicitly applied target-penalty actuation.
    const double u_observed = (xcom_observed - d.xcom_observed_prev) / dt;
    const double v_observed = (ycom_observed - d.ycom_observed_prev) / dt;
    const double omega_observed = wrap_angle(theta_observed - d.theta_observed_prev) / dt;

    fx = d.body_mass * (u_observed - d.u_com_current) / dt - actuation_fx;
    fy = d.body_mass * (v_observed - d.v_com_current) / dt - actuation_fy;
    torque_z = d.body_inertia * (omega_observed - d.omega_body_current) / dt - actuation_torque_z;
}

void integrate_linear_momentum(double fx, double fy, double dt, Eel2DData& d)
{
    if (d.body_mass <= std::numeric_limits<double>::epsilon()) return;
    d.u_com_current += dt * fx / d.body_mass;
    d.v_com_current += dt * fy / d.body_mass;
}

void integrate_angular_momentum(double torque_z, double dt, Eel2DData& d)
{
    if (d.body_inertia <= std::numeric_limits<double>::epsilon()) return;
    d.omega_body_current += dt * torque_z / d.body_inertia;
}

void update_free_swimming_pose(double dt, Eel2DData& d)
{
    d.xcom_cur += dt * d.u_com_current;
    d.ycom_cur += dt * d.v_com_current;
    d.theta_cur = wrap_angle(d.theta_cur + dt * d.omega_body_current);
}

void update_observed_pose_history(double xcom_observed,
                                  double ycom_observed,
                                  double theta_observed,
                                  Eel2DData& d)
{
    d.xcom_observed_prev = xcom_observed;
    d.ycom_observed_prev = ycom_observed;
    d.theta_observed_prev = theta_observed;
}

bool is_self_propelled_steady_state(const Eel2DData& d)
{
    const double period = (std::abs(d.omega) > std::numeric_limits<double>::epsilon()) ? 2.0 * PI_VAL / std::abs(d.omega) : 0.0;
    if (period <= 0.0 || d.mean_speed_time < d.steady_min_cycles * period) return false;

    const double speed = std::sqrt(d.u_com_current * d.u_com_current + d.v_com_current * d.v_com_current);
    const double mean_speed =
        (d.mean_speed_time > std::numeric_limits<double>::epsilon()) ? d.mean_speed_integral / d.mean_speed_time : speed;
    const double force_norm =
        std::sqrt(d.hydrodynamic_force_x * d.hydrodynamic_force_x + d.hydrodynamic_force_y * d.hydrodynamic_force_y);
    const double speed_scale = std::max(mean_speed, 1.0e-8);

    return force_norm <= d.steady_force_tol &&
           std::abs(speed - mean_speed) / speed_scale <= d.steady_speed_rel_tol;
}

void compute_com_and_orientation(EquationSystems* equation_systems,
                                 const std::string& coords_system_name,
                                 MeshBase& mesh,
                                 double& xcom,
                                 double& ycom,
                                 double& theta,
                                 double theta_prev = std::numeric_limits<double>::quiet_NaN())
{
    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    std::vector<std::vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, THIRD);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<vector<double> >& phi = fe->get_phi();

    boost::multi_array<double, 2> X_node;
    VectorValue<double> x;

    double area = 0.0;
    double mx = 0.0, my = 0.0;
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);
            area += JxW[qp];
            mx += x(0) * JxW[qp];
            my += x(1) * JxW[qp];
        }
    }

    double local_first[3] = { area, mx, my };
    IBTK_MPI::sumReduction(local_first, 3);
    const double area_tot = std::max(local_first[0], std::numeric_limits<double>::epsilon());
    xcom = local_first[1] / area_tot;
    ycom = local_first[2] / area_tot;

    double cxx = 0.0, cyy = 0.0, cxy = 0.0;
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d) dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);
            const double dx = x(0) - xcom;
            const double dy = x(1) - ycom;
            cxx += dx * dx * JxW[qp];
            cxy += dx * dy * JxW[qp];
            cyy += dy * dy * JxW[qp];
        }
    }

    double local_cov[3] = { cxx, cyy, cxy };
    IBTK_MPI::sumReduction(local_cov, 3);
    theta = 0.5 * std::atan2(2.0 * local_cov[2], local_cov[0] - local_cov[1]);

    if (std::isfinite(theta_prev))
    {
        const double pi = 3.14159265358979323846;
        while (theta - theta_prev > 0.5 * pi) theta -= pi;
        while (theta - theta_prev < -0.5 * pi) theta += pi;
    }
}

void compute_com_and_rigid_pose(EquationSystems* equation_systems,
                                const std::string& coords_system_name,
                                MeshBase& mesh,
                                const Eel2DData& d,
                                double& xcom,
                                double& ycom,
                                double& theta,
                                double theta_prev = std::numeric_limits<double>::quiet_NaN())
{
    if (d.area_ref <= std::numeric_limits<double>::epsilon())
    {
        compute_com_and_orientation(equation_systems, coords_system_name, mesh, xcom, ycom, theta, theta_prev);
        return;
    }

    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    std::vector<std::vector<unsigned int> > dof_indices(NDIM);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, SEVENTH);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<libMesh::Point>& q_point = fe->get_xyz();
    const vector<vector<double> >& phi = fe->get_phi();

    boost::multi_array<double, 2> X_node;
    VectorValue<double> x;

    double area = 0.0;
    double mx = 0.0, my = 0.0;
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d_idx = 0; d_idx < NDIM; ++d_idx) dof_map.dof_indices(elem, dof_indices[d_idx], d_idx);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);
            area += JxW[qp];
            mx += x(0) * JxW[qp];
            my += x(1) * JxW[qp];
        }
    }

    double local_first[3] = { area, mx, my };
    IBTK_MPI::sumReduction(local_first, 3);
    const double area_tot = std::max(local_first[0], std::numeric_limits<double>::epsilon());
    xcom = local_first[1] / area_tot;
    ycom = local_first[2] / area_tot;

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);

    double a_local = 0.0;
    double b_local = 0.0;
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d_idx = 0; d_idx < NDIM; ++d_idx) dof_map.dof_indices(elem, dof_indices[d_idx], d_idx);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        const unsigned int n_qp = qrule->n_points();
        for (unsigned int qp = 0; qp < n_qp; ++qp)
        {
            interpolate(x, qp, X_node, phi);

            const double qx = x(0) - xcom;
            const double qy = x(1) - ycom;

            const double dx_ref = q_point[qp](0) - d.xcom_ref;
            const double dy_ref = q_point[qp](1) - d.ycom_ref;
            const double xhat_ref = cr * dx_ref + sr * dy_ref;
            const double yhat_ref = -sr * dx_ref + cr * dy_ref;

            a_local += (xhat_ref * qx + yhat_ref * qy) * JxW[qp];
            b_local += (xhat_ref * qy - yhat_ref * qx) * JxW[qp];
        }
    }

    double local_fit[2] = { a_local, b_local };
    IBTK_MPI::sumReduction(local_fit, 2);
    theta = std::atan2(local_fit[1], local_fit[0]);

    if (std::isfinite(theta_prev))
    {
        const double pi = 3.14159265358979323846;
        while (theta - theta_prev > pi) theta -= 2.0 * pi;
        while (theta - theta_prev < -pi) theta += 2.0 * pi;
    }
}

double compute_tail_y(EquationSystems* equation_systems,
                      const std::string& coords_system_name,
                      MeshBase& mesh,
                      const Eel2DData& d)
{
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);

    // ------------------------------------------------------------
    // Pass 1: 仅遍历本 rank 拥有的 local nodes，找 reference body frame
    //         中最靠后的尾端轴向坐标 xhat_tail
    // ------------------------------------------------------------
    double local_xhat_max = -std::numeric_limits<double>::max();

    for (const Node* node : mesh.local_node_ptr_range())
    {
        if (!node) continue;

        const double dx_ref = (*node)(0) - d.xcom_ref;
        const double dy_ref = (*node)(1) - d.ycom_ref;
        const double xhat_ref = cr * dx_ref + sr * dy_ref;

        local_xhat_max = std::max(local_xhat_max, xhat_ref);
    }

    double xhat_tail = local_xhat_max;
    IBTK_MPI::maxReduction(xhat_tail);

    // ------------------------------------------------------------
    // Pass 2: 再遍历 owner local nodes，挑出 xhat_ref ≈ xhat_tail 的尾端节点，
    //         读取这些节点当前 y 坐标并求平均
    // ------------------------------------------------------------
    const double tol = 1.0e-10 * std::max(1.0, std::abs(xhat_tail));

    double ysum = 0.0;
    double nsum = 0.0;

    for (const Node* node : mesh.local_node_ptr_range())
    {
        if (!node) continue;

        const double dx_ref = (*node)(0) - d.xcom_ref;
        const double dy_ref = (*node)(1) - d.ycom_ref;
        const double xhat_ref = cr * dx_ref + sr * dy_ref;

        if (std::abs(xhat_ref - xhat_tail) > tol) continue;

        std::vector<dof_id_type> dof_idx_y;
        dof_map.dof_indices(node, dof_idx_y, 1); // variable 1 = y
        ysum += (*X_ghost_vec)(dof_idx_y[0]);
        nsum += 1.0;
    }

    double local_vals[2] = { ysum, nsum };
    IBTK_MPI::sumReduction(local_vals, 2);

    if (local_vals[1] < 0.5)
    {
        TBOX_WARNING("compute_tail_y(): no owned tail nodes found.\n");
        return std::numeric_limits<double>::quiet_NaN();
    }

    return local_vals[0] / local_vals[1];
}

void compute_tail_actual_state(EquationSystems* equation_systems,
                               const std::string& coords_system_name,
                               MeshBase& mesh,
                               double time,
                               double xcom_frame,
                               double ycom_frame,
                               double theta_frame,
                               Eel2DData& d,
                               double& tail_y,
                               double& tail_q,
                               double& tail_axial_strain,
                               double& tail_curvature,
                               double& tail_q_eff)
{
    update_current_centerline_cache(equation_systems, coords_system_name, mesh, xcom_frame, ycom_frame, theta_frame, time, d);

    const double s_tail = d.L_mesh;
    const double xhat_tail = current_centerline_x(s_tail, d);
    const double yhat_tail = current_centerline_y(s_tail, d);
    const double ct = std::cos(theta_frame);
    const double st = std::sin(theta_frame);

    tail_y = ycom_frame + st * xhat_tail + ct * yhat_tail;
    tail_q = yhat_tail - mesh_centerline_y(s_tail, d);
    tail_axial_strain = current_centerline_axial_strain(s_tail, d);
    tail_curvature = current_centerline_curvature(s_tail, d);
    tail_q_eff = compute_effective_tail_deformation(tail_q, tail_axial_strain, tail_curvature, s_tail, d);
}

double compute_attack_angle(double vx, double vy, double theta)
{
    if (std::sqrt(vx * vx + vy * vy) <= 1.0e-12) return 0.0;
    return wrap_angle(theta - std::atan2(vy, vx));
}

double compute_tail_attack_angle(double time,
                                 Eel2DData& d,
                                 double& tail_vx,
                                 double& tail_vy,
                                 double& tail_heading,
                                 double& tail_q)
{
    update_projected_kinematics_cache(time, d);

    const double s_tail = d.L_mesh;
    const double xhat_ref = d.xhat_tail_ref;
    const double yhat_ref = mesh_centerline_y(s_tail, d);

    double disp_x_raw = 0.0, disp_y_raw = 0.0, vel_x_raw = 0.0, vel_y_raw = 0.0;
    eel_raw_body_frame_kinematics(s_tail, time, d, disp_x_raw, disp_y_raw, vel_x_raw, vel_y_raw);

    const double disp_x = disp_x_raw - d.disp_tx + d.disp_rot * yhat_ref;
    const double disp_y = disp_y_raw - d.disp_ty - d.disp_rot * xhat_ref;
    const double vel_x = vel_x_raw - d.vel_tx + d.vel_rot * yhat_ref;
    const double vel_y = vel_y_raw - d.vel_ty - d.vel_rot * xhat_ref;

    const double xhat_tail = xhat_ref + disp_x;
    const double yhat_tail = yhat_ref + disp_y;
    const double ct = std::cos(d.theta_cur);
    const double st = std::sin(d.theta_cur);
    const double rx = ct * xhat_tail - st * yhat_tail;
    const double ry = st * xhat_tail + ct * yhat_tail;

    tail_vx = d.u_com_current - d.omega_body_current * ry + ct * vel_x - st * vel_y;
    tail_vy = d.v_com_current + d.omega_body_current * rx + st * vel_x + ct * vel_y;
    tail_heading = d.theta_cur + std::atan(body_wave_dy_ds(s_tail, time, d));
    tail_q = yhat_tail - mesh_centerline_y(s_tail, d);
    return compute_attack_angle(tail_vx, tail_vy, tail_heading);
}

double compute_alpha_max_over_cycle(double alpha, double time, Eel2DData& d)
{
    const double period = (std::abs(d.omega) > std::numeric_limits<double>::epsilon()) ? 2.0 * PI_VAL / std::abs(d.omega) : 0.0;
    if (period <= 0.0)
    {
        d.alpha_cycle_max_current = std::max(d.alpha_cycle_max_current, std::abs(alpha));
        return d.alpha_cycle_max_current;
    }

    const int cycle_index = static_cast<int>(std::floor(time / period + 1.0e-12));
    if (!d.alpha_cycle_initialized)
    {
        d.alpha_cycle_index = cycle_index;
        d.alpha_cycle_max_current = std::abs(alpha);
        d.alpha_cycle_max_last = std::abs(alpha);
        d.alpha_cycle_initialized = true;
        return d.alpha_cycle_max_current;
    }

    if (cycle_index != d.alpha_cycle_index)
    {
        d.alpha_cycle_max_last = d.alpha_cycle_max_current;
        d.alpha_cycle_index = cycle_index;
        d.alpha_cycle_max_current = std::abs(alpha);
    }
    else
    {
        d.alpha_cycle_max_current = std::max(d.alpha_cycle_max_current, std::abs(alpha));
    }
    return d.alpha_cycle_max_current;
}

double compute_output_power(double fx, double fy, double torque_z, double u, double v, double omega_body)
{
    return std::max(0.0, -(fx * u + fy * v + torque_z * omega_body));
}

double compute_input_power(double pout_inst, double tail_elastic_power_rate)
{
    // Stage-1 approximation: estimate the required actuation power as useful
    // output power plus the positive rate of tail elastic energy storage.
    return std::max(0.0, pout_inst) + std::max(0.0, tail_elastic_power_rate);
}

double compute_froude_efficiency(double Pout_avg, double Pin_avg)
{
    if (Pin_avg <= std::numeric_limits<double>::epsilon()) return 0.0;
    return Pout_avg / Pin_avg;
}

double compute_cost_of_transport(double Pin_avg, double mass, double mean_speed)
{
    const double denom = mass * mean_speed;
    if (denom <= std::numeric_limits<double>::epsilon()) return 0.0;
    return Pin_avg / denom;
}

double compute_mean_swimming_speed(double speed_inst, double dt, Eel2DData& d)
{
    if (dt > 0.0)
    {
        d.mean_speed_integral += speed_inst * dt;
        d.mean_speed_time += dt;
    }
    if (d.mean_speed_time <= std::numeric_limits<double>::epsilon()) return speed_inst;
    return d.mean_speed_integral / d.mean_speed_time;
}

double compute_reynolds_number(double U, double L, double nu)
{
    if (nu <= std::numeric_limits<double>::epsilon()) return 0.0;
    return U * L / nu;
}

double compute_strouhal_number(double f, double AF, double U)
{
    if (U <= std::numeric_limits<double>::epsilon()) return 0.0;
    return f * AF / U;
}

double compute_tail_peak_to_peak_amplitude(double tail_q, double time, Eel2DData& d)
{
    const double period = (std::abs(d.omega) > std::numeric_limits<double>::epsilon()) ? 2.0 * PI_VAL / std::abs(d.omega) : 0.0;
    if (period <= 0.0)
    {
        if (!d.tail_cycle_initialized)
        {
            d.tail_q_cycle_min = tail_q;
            d.tail_q_cycle_max = tail_q;
            d.tail_peak_to_peak_last = 0.0;
            d.tail_cycle_initialized = true;
        }
        d.tail_q_cycle_min = std::min(d.tail_q_cycle_min, tail_q);
        d.tail_q_cycle_max = std::max(d.tail_q_cycle_max, tail_q);
        return d.tail_q_cycle_max - d.tail_q_cycle_min;
    }

    const int cycle_index = static_cast<int>(std::floor(time / period + 1.0e-12));
    if (!d.tail_cycle_initialized)
    {
        d.tail_cycle_index = cycle_index;
        d.tail_q_cycle_min = tail_q;
        d.tail_q_cycle_max = tail_q;
        d.tail_peak_to_peak_last = 0.0;
        d.tail_cycle_initialized = true;
        return 0.0;
    }

    if (cycle_index != d.tail_cycle_index)
    {
        d.tail_peak_to_peak_last = d.tail_q_cycle_max - d.tail_q_cycle_min;
        d.tail_cycle_index = cycle_index;
        d.tail_q_cycle_min = tail_q;
        d.tail_q_cycle_max = tail_q;
    }
    else
    {
        d.tail_q_cycle_min = std::min(d.tail_q_cycle_min, tail_q);
        d.tail_q_cycle_max = std::max(d.tail_q_cycle_max, tail_q);
    }

    return d.tail_q_cycle_max - d.tail_q_cycle_min;
}

inline void compute_eel_target(
    const libMesh::Point& X,
    double time,
    Eel2DData& d,
    double& xtar,
    double& ytar,
    double& utar_x,
    double& utar_y);

inline void
compute_eel_target(const libMesh::Point& X,
                   double time,
                   Eel2DData& d,
                   double& xtar,
                   double& ytar,
                   double& utar_x,
                   double& utar_y)
{
    update_projected_kinematics_cache(time, d);

    // ------------------------------------------------------------
    // 1) 参考构型中的当前材料点，先转换到 reference body frame
    //    theta_ref 定义参考 body frame 相对于实验室坐标的姿态。
    // ------------------------------------------------------------
    const double dx_ref = X(0) - d.xcom_ref;
    const double dy_ref = X(1) - d.ycom_ref;

    const double cr = std::cos(d.theta_ref);
    const double sr = std::sin(d.theta_ref);

    // reference body-frame coordinates of the material point
    const double xhat_ref =  cr * dx_ref + sr * dy_ref;
    const double yhat_ref = -sr * dx_ref + cr * dy_ref;

    double s = 0.0, eta = 0.0;
    compute_body_coordinates(xhat_ref, yhat_ref, d, s, eta);

    // ------------------------------------------------------------
    // 3) 在 reference body frame 中施加 eel 的横向行波形变
    // ------------------------------------------------------------
    double disp_x_raw, disp_y_raw, vel_x_raw, vel_y_raw;
    eel_raw_body_frame_kinematics(s, time, d, disp_x_raw, disp_y_raw, vel_x_raw, vel_y_raw);

    // body-frame target point after deformation
    const double disp_x = disp_x_raw - d.disp_tx + d.disp_rot * yhat_ref;
    const double disp_y = disp_y_raw - d.disp_ty - d.disp_rot * xhat_ref;
    const double vel_x = vel_x_raw - d.vel_tx + d.vel_rot * yhat_ref;
    const double vel_y = vel_y_raw - d.vel_ty - d.vel_rot * xhat_ref;

    const double xhat_tar = xhat_ref + disp_x;
    const double yhat_tar = yhat_ref + disp_y;

    // ------------------------------------------------------------
    // 4) 再用当前自由位姿映射回实验室坐标
    // ------------------------------------------------------------
    const double ct = std::cos(d.theta_cur);
    const double st = std::sin(d.theta_cur);
    const double rx = ct * xhat_tar - st * yhat_tar;
    const double ry = st * xhat_tar + ct * yhat_tar;

    xtar = d.xcom_cur + rx;
    ytar = d.ycom_cur + ry;

    // ------------------------------------------------------------
    // 5) 目标速度包含刚体平移、刚体转动以及去除刚体模态后的
    //    body-frame 形变速度。
    // ------------------------------------------------------------
    utar_x = d.u_com_current - d.omega_body_current * ry + ct * vel_x - st * vel_y;
    utar_y = d.v_com_current + d.omega_body_current * rx + st * vel_x + ct * vel_y;
}

// Tether (penalty) stress function.
void
PK1_stress_function(TensorValue<double>& PP,
                    const TensorValue<double>& FF,
                    const libMesh::Point& x,
                    const libMesh::Point& X,
                    Elem* const /*elem*/,
                    const vector<const vector<double>*>& /*var_data*/,
                    const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                    double time,
                    void* ctx)
{
    const Eel2DData* const d = reinterpret_cast<Eel2DData*>(ctx);

    const double cr = std::cos(d->theta_ref);
    const double sr = std::sin(d->theta_ref);

    const double dx_ref = X(0) - d->xcom_ref;
    const double dy_ref = X(1) - d->ycom_ref;
    const double xhat_ref = cr * dx_ref + sr * dy_ref;
    const double yhat_ref = -sr * dx_ref + cr * dy_ref;

    double s = 0.0, eta = 0.0;
    compute_body_coordinates(xhat_ref, yhat_ref, *d, s, eta);

    double transverse_deflection = 0.0, axial_strain = 0.0, curvature_like = 0.0;
    compute_actual_deformation_measures(x, FF, s, eta, time, *d, transverse_deflection, axial_strain, curvature_like);

    const double local_modulus = compute_local_modulus(s, transverse_deflection, axial_strain, curvature_like, *d);
    PP = 2.0 * local_modulus * (FF - tensor_inverse_transpose(FF, NDIM));

    if (!d->use_active_stress) return;

    const double h = body_half_width(s, *d);
    if (h <= 1.0e-12) return;

    const double xi = clamp_value(s / d->L_mesh, 0.0, 1.0);
    const double s_wave = d->L * xi;
    const double m_space = mask_interval(xi, d->active_xi_start, d->active_xi_end, d->active_xi_eps);
    if (m_space <= 0.0) return;

    const double envelope = (s_wave + d->s_shift) / (d->L + d->s_shift);
    const double eta_over_h = std::max(-1.0, std::min(1.0, eta / h));
    const double phase = d->k_wave * s_wave - d->omega * time + d->active_phase0;
    const double Tact =
        d->active_stress_t0 * ramp_factor(time, d->active_ramp_time) * m_space * envelope * eta_over_h * std::sin(phase);

    libMesh::VectorValue<double> A0;
    A0(0) = cr;
    A0(1) = sr;
#if (NDIM == 3)
    A0(2) = 0.0;
#endif

    const auto FA0 = FF * A0;
    TensorValue<double> PP_active;
    outer_product(PP_active, FA0, A0);
    PP += Tact * PP_active;
    return;
} // PK1_stress_function

void
eel_body_force_function(libMesh::VectorValue<double>& F,
                        const libMesh::TensorValue<double>& /*FF*/,
                        const libMesh::Point& x,
                        const libMesh::Point& X,
                        libMesh::Elem* const /*elem*/,
                        const std::vector<const std::vector<double>*>& var_data,
                        const std::vector<const std::vector<libMesh::VectorValue<double> >*>& /*grad_var_data*/,
                        double time,
                        void* ctx)
{
    const Eel2DData* const d = reinterpret_cast<Eel2DData*>(ctx);
    const std::vector<double>& U = *var_data[0];

    double xtar = 0.0, ytar = 0.0, utar_x = 0.0, utar_y = 0.0;
    compute_eel_target(X, time, *const_cast<Eel2DData*>(d), xtar, ytar, utar_x, utar_y);

    F(0) = d->kappa_s_body * (xtar - x(0)) + d->eta_s_body * (utar_x - U[0]);
    F(1) = d->kappa_s_body * (ytar - x(1)) + d->eta_s_body * (utar_y - U[1]);
}

void
eel_surface_force_function(VectorValue<double>& F,
                           const VectorValue<double>& /*n*/,
                           const VectorValue<double>& /*N*/,
                           const TensorValue<double>& /*FF*/,
                           const libMesh::Point& x,
                           const libMesh::Point& X,
                           Elem* const /*elem*/,
                           const unsigned short /*side*/,
                           const vector<const vector<double>*>& var_data,
                           const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                           double time,
                           void* ctx)
{
    const Eel2DData* const d = reinterpret_cast<Eel2DData*>(ctx);

    VectorValue<double> U;
    for (unsigned int k = 0; k < NDIM; ++k) U(k) = (*var_data[0])[k];

    double xtar = 0.0, ytar = 0.0, utar_x = 0.0, utar_y = 0.0;
    compute_eel_target(X, time, *const_cast<Eel2DData*>(d), xtar, ytar, utar_x, utar_y);

    F(0) = d->kappa_s_surface * (xtar - x(0)) + d->eta_s_surface * (utar_x - U(0));
    F(1) = d->kappa_s_surface * (ytar - x(1)) + d->eta_s_surface * (utar_y - U(1));
    return;
} // eel_surface_force_function

void compute_lagrangian_actuation_loads(EquationSystems* equation_systems,
                                        const std::string& coords_system_name,
                                        const std::string& velocity_system_name,
                                        MeshBase& mesh,
                                        bool use_boundary_mesh,
                                        double time,
                                        Eel2DData& eel_data,
                                        double xref,
                                        double yref,
                                        double& fx,
                                        double& fy,
                                        double& torque_z)
{
    fx = 0.0;
    fy = 0.0;
    torque_z = 0.0;

    void* const eel_data_ptr = reinterpret_cast<void*>(&eel_data);
    const unsigned int dim = mesh.mesh_dimension();
    System& X_system = equation_systems->get_system(coords_system_name);
    System& U_system = equation_systems->get_system(velocity_system_name);
    NumericVector<double>* X_vec = X_system.solution.get();
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    copy_and_synch(*X_vec, *X_ghost_vec);
    NumericVector<double>* U_vec = U_system.solution.get();
    NumericVector<double>* U_ghost_vec = U_system.current_local_solution.get();
    copy_and_synch(*U_vec, *U_ghost_vec);
    const DofMap& dof_map = X_system.get_dof_map();
    std::vector<std::vector<unsigned int> > dof_indices(NDIM);

    std::unique_ptr<FEBase> fe(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule = QBase::build(QGAUSS, dim, SEVENTH);
    fe->attach_quadrature_rule(qrule.get());
    const vector<double>& JxW = fe->get_JxW();
    const vector<libMesh::Point>& q_point = fe->get_xyz();
    const vector<vector<double> >& phi = fe->get_phi();
    const vector<vector<VectorValue<double> > >& dphi = fe->get_dphi();

    std::unique_ptr<FEBase> fe_face(FEBase::build(dim, dof_map.variable_type(0)));
    std::unique_ptr<QBase> qrule_face = QBase::build(QGAUSS, dim - 1, SEVENTH);
    fe_face->attach_quadrature_rule(qrule_face.get());
    const vector<double>& JxW_face = fe_face->get_JxW();
    const vector<libMesh::Point>& q_point_face = fe_face->get_xyz();
    const vector<libMesh::Point>& normal_face = fe_face->get_normals();
    const vector<vector<double> >& phi_face = fe_face->get_phi();
    const vector<vector<VectorValue<double> > >& dphi_face = fe_face->get_dphi();

    std::vector<double> U_qp_vec(NDIM);
    std::vector<const std::vector<double>*> var_data(1);
    var_data[0] = &U_qp_vec;
    std::vector<const std::vector<libMesh::VectorValue<double> >*> grad_var_data;

    TensorValue<double> FF, FF_inv_trans;
    boost::multi_array<double, 2> X_node, U_node;
    VectorValue<double> F, N, U, n, x;

    double local_loads[3] = { 0.0, 0.0, 0.0 };
    const auto el_begin = mesh.active_local_elements_begin();
    const auto el_end = mesh.active_local_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        auto elem = *el_it;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            dof_map.dof_indices(elem, dof_indices[d], d);
        }
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, dof_indices);

        if (!use_boundary_mesh)
        {
            const unsigned int n_qp = qrule->n_points();
            for (unsigned int qp = 0; qp < n_qp; ++qp)
            {
                interpolate(x, qp, X_node, phi);
                jacobian(FF, qp, X_node, dphi);
                interpolate(U, qp, U_node, phi);
                for (unsigned int d = 0; d < NDIM; ++d) U_qp_vec[d] = U(d);

                eel_body_force_function(F, FF, x, q_point[qp], elem, var_data, grad_var_data, time, eel_data_ptr);
                local_loads[0] += F(0) * JxW[qp];
                local_loads[1] += F(1) * JxW[qp];
                local_loads[2] += ((x(0) - xref) * F(1) - (x(1) - yref) * F(0)) * JxW[qp];
            }
        }

        for (unsigned short int side = 0; side < elem->n_sides(); ++side)
        {
            if (!use_boundary_mesh && elem->neighbor_ptr(side)) continue;

            fe_face->reinit(elem, side);
            const unsigned int n_qp_face = qrule_face->n_points();
            for (unsigned int qp = 0; qp < n_qp_face; ++qp)
            {
                interpolate(x, qp, X_node, phi_face);
                jacobian(FF, qp, X_node, dphi_face);
                interpolate(U, qp, U_node, phi_face);
                for (unsigned int d = 0; d < NDIM; ++d) U_qp_vec[d] = U(d);
                N = normal_face[qp];
                tensor_inverse_transpose(FF_inv_trans, FF, NDIM);
                n = (FF_inv_trans * N).unit();

                eel_surface_force_function(F,
                                           n,
                                           N,
                                           FF,
                                           x,
                                           q_point_face[qp],
                                           elem,
                                           side,
                                           var_data,
                                           grad_var_data,
                                           time,
                                           eel_data_ptr);
                local_loads[0] += F(0) * JxW_face[qp];
                local_loads[1] += F(1) * JxW_face[qp];
                local_loads[2] += ((x(0) - xref) * F(1) - (x(1) - yref) * F(0)) * JxW_face[qp];
            }
        }
    }

    IBTK_MPI::sumReduction(local_loads, 3);
    fx = local_loads[0];
    fy = local_loads[1];
    torque_z = local_loads[2];
}

} // namespace ModelData
using namespace ModelData;

// Function prototypes
static ofstream drag_stream, lift_stream, U_L1_norm_stream, U_L2_norm_stream, U_max_norm_stream, pose_stream,
    swim_diag_stream;
void postprocess_data(Pointer<Database> input_db,
                      Pointer<PatchHierarchy<NDIM> > patch_hierarchy,
                      Pointer<INSHierarchyIntegrator> navier_stokes_integrator,
                      MeshBase& mesh,
                      EquationSystems* equation_systems,
                      const std::string& coords_system_name,
                      const std::string& velocity_system_name,
                      const int iteration_num,
                      const double loop_time,
                      const string& data_dump_dirname);

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/

int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Setup user-defined kernel function.
        LEInteractor::s_kernel_fcn = &kernel;
        LEInteractor::s_kernel_fcn_stencil_size = 8;

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();
#ifdef LIBMESH_HAVE_EXODUS_API
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
#else
        const bool uses_exodus = false;
        if (!app_initializer->getExodusIIFilename().empty())
        {
            plog << "WARNING: libMesh was compiled without Exodus support, so no "
                 << "Exodus output will be written in this program.\n";
        }
#endif
        const string exodus_filename = app_initializer->getExodusIIFilename();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();
        const string restart_read_dirname = app_initializer->getRestartReadDirectory();
        const int restart_restore_num = app_initializer->getRestartRestoreNumber();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create solid mesh
        Mesh solid_mesh(init.comm(), NDIM);

        const double dx = input_db->getDouble("DX");

        // Read mesh: use a libMesh-supported format, e.g. .msh
        solid_mesh.read("fish2d.msh");
        solid_mesh.prepare_for_use();

        pout << "mesh_dimension=" << solid_mesh.mesh_dimension()
            << ", spatial_dimension=" << solid_mesh.spatial_dimension() << "\n";

        const bool use_boundary_mesh = input_db->getBoolWithDefault("USE_BOUNDARY_MESH", false);
        if (use_boundary_mesh)
        {
            TBOX_ERROR("eel2d_ibfe.cpp first-pass migration requires USE_BOUNDARY_MESH = FALSE\n"
                       << "so that IBFEMethod body+surface target penalty callbacks are both active.");
        }

        MeshBase& mesh = static_cast<MeshBase&>(solid_mesh);

        // Create major algorithm and data objects that comprise the
        // application. These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<INSHierarchyIntegrator> navier_stokes_integrator;
        const string solver_type = app_initializer->getComponentDatabase("Main")->getString("solver_type");
        if (solver_type == "STAGGERED")
        {
            navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
                "INSStaggeredHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
        }
        else if (solver_type == "COLLOCATED")
        {
            navier_stokes_integrator = new INSCollocatedHierarchyIntegrator(
                "INSCollocatedHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSCollocatedHierarchyIntegrator"));
        }
        else
        {
            TBOX_ERROR("Unsupported solver type: " << solver_type << "\n"
                                                   << "Valid options are: COLLOCATED, STAGGERED");
        }
        Pointer<IBStrategy> ib_ops =
            new IBFEMethod("IBFEMethod",
                           app_initializer->getComponentDatabase("IBFEMethod"),
                           &mesh,
                           app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
                           /*register_for_restart*/ true,
                           restart_read_dirname,
                           restart_restore_num);
        Pointer<IBHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_ops,
                                              navier_stokes_integrator);
        Pointer<CartesianGridGeometry<NDIM> > grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM> > load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);

        // Configure the IBFE solver.
        // 自由游动版本：body-frame 规定形变，整体平移/转动由流固耦合自然产生。
        Eel2DData eel_data(input_db);
        s_eel_data_state = &eel_data;
        void* const eel_data_ptr = reinterpret_cast<void*>(&eel_data);
        EquationSystems* equation_systems;
        std::string coords_system_name, velocity_system_name;
        std::vector<int> vars(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) vars[d] = d;
        Pointer<IBFEMethod> ibfe_ops = ib_ops;
        ibfe_ops->initializeFEEquationSystems();
        equation_systems = ibfe_ops->getFEDataManager()->getEquationSystems();
        coords_system_name = ibfe_ops->getCurrentCoordinatesSystemName();
        velocity_system_name = ibfe_ops->getVelocitySystemName();

        vector<SystemData> sys_data(1, SystemData(velocity_system_name, vars));

        IBFEMethod::PK1StressFcnData PK1_stress_data(
            PK1_stress_function, std::vector<IBTK::SystemData>(), eel_data_ptr);
        const std::string pk1_quad_order_name =
            input_db->getStringWithDefault("PK1_QUAD_ORDER", input_db->getStringWithDefault("PK1_DEV_QUAD_ORDER", "THIRD"));
        PK1_stress_data.quad_order = Utility::string_to_enum<libMesh::Order>(pk1_quad_order_name);
        ibfe_ops->registerPK1StressFunction(PK1_stress_data);

        IBFEMethod::LagSurfaceForceFcnData surface_fcn_data(eel_surface_force_function, sys_data, eel_data_ptr);
        ibfe_ops->registerLagSurfaceForceFunction(surface_fcn_data);

        IBFEMethod::LagBodyForceFcnData body_fcn_data(eel_body_force_function, sys_data, eel_data_ptr);
        ibfe_ops->registerLagBodyForceFunction(body_fcn_data);

        if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
        {
            ibfe_ops->registerStressNormalizationPart();
        }

        // Create Eulerian initial condition specification objects.
        if (input_db->keyExists("VelocityInitialConditions"))
        {
            Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
                "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerVelocityInitialConditions(u_init);
        }

        if (input_db->keyExists("PressureInitialConditions"))
        {
            Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
                "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
            navier_stokes_integrator->registerPressureInitialConditions(p_init);
        }

        // Create Eulerian boundary condition specification objects (when necessary).
        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        if (periodic_shift.min() > 0)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u_bc_coefs[d] = nullptr;
            }
        }
        else
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

                const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

                u_bc_coefs[d] = new muParserRobinBcCoefs(
                    bc_coefs_name, app_initializer->getComponentDatabase(bc_coefs_db_name), grid_geometry);
            }
            navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        }

        // Create Eulerian body force function specification objects.
        if (input_db->keyExists("ForcingFunction"))
        {
            Pointer<CartGridFunction> f_fcn = new muParserCartGridFunction(
                "f_fcn", app_initializer->getComponentDatabase("ForcingFunction"), grid_geometry);
            time_integrator->registerBodyForceFunction(f_fcn);
        }

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }
        std::unique_ptr<ExodusII_IO> exodus_io = uses_exodus ? std::make_unique<ExodusII_IO>(mesh) : nullptr;

        // Check to see if this is a restarted run to append current exodus files
        if (uses_exodus)
        {
            const bool from_restart = RestartManager::getManager()->isFromRestart();
            exodus_io->append(from_restart);
        }

        // Initialize hierarchy configuration and data on all patches.
        ibfe_ops->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // 初始化参考位姿（body-frame 相对该位姿定义）。
        compute_com_and_orientation(
            equation_systems, coords_system_name, mesh, eel_data.xcom_cur, eel_data.ycom_cur, eel_data.theta_cur);
        eel_data.xcom_ref = eel_data.xcom_cur;
        eel_data.ycom_ref = eel_data.ycom_cur;
        eel_data.theta_ref = eel_data.theta_cur;
        initialize_reference_body_geometry(mesh, eel_data);
        initialize_reference_projection_data(equation_systems, coords_system_name, mesh, eel_data);
        initialize_rigid_body_dynamics(eel_data);
        s_equation_systems_state = equation_systems;
        s_mesh_state = &mesh;
        s_coords_system_name_state = coords_system_name;

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        if (dump_viz_data)
        {
            pout << "\n\nWriting visualization files...\n\n";
            if (uses_visit)
            {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            }
            if (uses_exodus)
            {
                exodus_io->write_timestep(
                    exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
            }
        }

        // Open streams to save lift and drag coefficients and the norms of the
        // velocity.
        if (IBTK_MPI::getRank() == 0)
        {
            drag_stream.open("C_D.curve", ios_base::out | ios_base::trunc);
            lift_stream.open("C_L.curve", ios_base::out | ios_base::trunc);
            U_L1_norm_stream.open("U_L1.curve", ios_base::out | ios_base::trunc);
            U_L2_norm_stream.open("U_L2.curve", ios_base::out | ios_base::trunc);
            U_max_norm_stream.open("U_max.curve", ios_base::out | ios_base::trunc);
            pose_stream.open("pose.curve", ios_base::out | ios_base::trunc);
            swim_diag_stream.open("swim_diagnostics.curve", ios_base::out | ios_base::trunc);

            drag_stream.precision(10);
            lift_stream.precision(10);
            U_L1_norm_stream.precision(10);
            U_L2_norm_stream.precision(10);
            U_max_norm_stream.precision(10);
            pose_stream.precision(10);
            swim_diag_stream.precision(10);
            pose_stream << "# iter time x_com y_com theta tail_y" << endl;
            swim_diag_stream
                << "# iter time u_com v_com omega_body Fx_act Fy_act Tz_act Fx_h Fy_h Tz_h tail_alpha alpha_max_cycle tail_q AF "
                   "Pout Pin Pout_avg Pin_avg eta_F CoT U_mean Re St k_tail_eq F_tail_restore steady"
                << endl;
        }

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;
        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();

            pout << "Rigid predictor pose(COM/theta): (" << eel_data.xcom_cur << ", " << eel_data.ycom_cur << ", "
                 << eel_data.theta_cur << ")\n";

            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            double xcom_observed = 0.0, ycom_observed = 0.0, theta_observed = 0.0;
            compute_com_and_rigid_pose(equation_systems,
                                       coords_system_name,
                                       mesh,
                                       eel_data,
                                       xcom_observed,
                                       ycom_observed,
                                       theta_observed,
                                       eel_data.theta_observed_prev);

            double actuation_fx = 0.0, actuation_fy = 0.0, actuation_torque = 0.0;
            compute_lagrangian_actuation_loads(equation_systems,
                                               coords_system_name,
                                               velocity_system_name,
                                               mesh,
                                               use_boundary_mesh,
                                               loop_time,
                                               eel_data,
                                               xcom_observed,
                                               ycom_observed,
                                               actuation_fx,
                                               actuation_fy,
                                               actuation_torque);

            double hydrodynamic_fx = 0.0, hydrodynamic_fy = 0.0, hydrodynamic_torque = 0.0;
            compute_hydrodynamic_force_and_torque(xcom_observed,
                                                  ycom_observed,
                                                  theta_observed,
                                                  actuation_fx,
                                                  actuation_fy,
                                                  actuation_torque,
                                                  dt,
                                                  eel_data,
                                                  hydrodynamic_fx,
                                                  hydrodynamic_fy,
                                                  hydrodynamic_torque);
            integrate_linear_momentum(hydrodynamic_fx, hydrodynamic_fy, dt, eel_data);
            integrate_angular_momentum(hydrodynamic_torque, dt, eel_data);
            update_free_swimming_pose(dt, eel_data);
            update_observed_pose_history(xcom_observed, ycom_observed, theta_observed, eel_data);
            eel_data.actuation_force_x = actuation_fx;
            eel_data.actuation_force_y = actuation_fy;
            eel_data.actuation_torque_z = actuation_torque;
            eel_data.hydrodynamic_force_x = hydrodynamic_fx;
            eel_data.hydrodynamic_force_y = hydrodynamic_fy;
            eel_data.hydrodynamic_torque_z = hydrodynamic_torque;

            pout << "Observed pose(COM/theta): (" << xcom_observed << ", " << ycom_observed << ", " << theta_observed
                 << ")\n";
            pout << "Updated rigid pose(COM/theta): (" << eel_data.xcom_cur << ", " << eel_data.ycom_cur << ", "
                 << eel_data.theta_cur << ")\n";
            pout << "Actuation load: (" << actuation_fx << ", " << actuation_fy << ", " << actuation_torque << ")\n";
            pout << "Hydrodynamic load: (" << hydrodynamic_fx << ", " << hydrodynamic_fy << ", "
                 << hydrodynamic_torque << ")\n";
            pout << "Rigid velocity: (" << eel_data.u_com_current << ", " << eel_data.v_com_current << ", "
                 << eel_data.omega_body_current << ")\n";

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    exodus_io->write_timestep(
                        exodus_filename, *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
                }
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
                dynamic_cast<IBFEMethod&>(*ib_ops).writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
            if (dump_postproc_data && (iteration_num % postproc_data_dump_interval == 0 || last_step))
            {
                postprocess_data(input_db,
                                 patch_hierarchy,
                                 navier_stokes_integrator,
                                 mesh,
                                 equation_systems,
                                 coords_system_name,
                                 velocity_system_name,
                                 iteration_num,
                                 loop_time,
                                 postproc_data_dump_dirname);
            }
        }

        // Close the logging streams.
        if (IBTK_MPI::getRank() == 0)
        {
            drag_stream.close();
            lift_stream.close();
            U_L1_norm_stream.close();
            U_L2_norm_stream.close();
            U_max_norm_stream.close();
            pose_stream.close();
            swim_diag_stream.close();
        }

        // Cleanup Eulerian boundary condition specification objects (when
        // necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

    } // cleanup dynamically allocated objects prior to shutdown
} // main

void
postprocess_data(Pointer<Database> input_db,
                 Pointer<PatchHierarchy<NDIM> > /*patch_hierarchy*/,
                 Pointer<INSHierarchyIntegrator> /*navier_stokes_integrator*/,
                 MeshBase& mesh,
                 EquationSystems* equation_systems,
                 const std::string& coords_system_name,
                 const std::string& velocity_system_name,
                 const int iteration_num,
                 const double loop_time,
                 const string& /*data_dump_dirname*/)
{
    Eel2DData eel_data_fallback(input_db);
    Eel2DData* const eel_data = s_eel_data_state ? s_eel_data_state : &eel_data_fallback;
    const bool use_boundary_mesh = input_db->getBoolWithDefault("USE_BOUNDARY_MESH", false);

    static double theta_prev_post = std::numeric_limits<double>::quiet_NaN();
    double xcom = 0.0, ycom = 0.0, theta = 0.0;
    compute_com_and_rigid_pose(equation_systems,
                               coords_system_name,
                               mesh,
                               *eel_data,
                               xcom,
                               ycom,
                               theta,
                               theta_prev_post);
    theta_prev_post = theta;

    double actuation_fx = 0.0, actuation_fy = 0.0, actuation_torque = 0.0;
    compute_lagrangian_actuation_loads(equation_systems,
                                       coords_system_name,
                                       velocity_system_name,
                                       mesh,
                                       use_boundary_mesh,
                                       loop_time,
                                       *eel_data,
                                       xcom,
                                       ycom,
                                       actuation_fx,
                                       actuation_fy,
                                       actuation_torque);

    double tail_y = std::numeric_limits<double>::quiet_NaN();
    double tail_q = 0.0, tail_axial_strain = 0.0, tail_curvature = 0.0, tail_q_eff = 0.0;
    compute_tail_actual_state(equation_systems,
                              coords_system_name,
                              mesh,
                              loop_time,
                              xcom,
                              ycom,
                              theta,
                              *eel_data,
                              tail_y,
                              tail_q,
                              tail_axial_strain,
                              tail_curvature,
                              tail_q_eff);

    double tail_vx = 0.0, tail_vy = 0.0, tail_heading = 0.0, tail_q_kinematic = 0.0;
    const double tail_alpha =
        compute_tail_attack_angle(loop_time, *eel_data, tail_vx, tail_vy, tail_heading, tail_q_kinematic);
    const double alpha_max_cycle = compute_alpha_max_over_cycle(tail_alpha, loop_time, *eel_data);
    const double AF = compute_tail_peak_to_peak_amplitude(tail_q, loop_time, *eel_data);
    const double tail_stiff_scale = stiffness_distribution(eel_data->L_mesh, *eel_data);
    const double tail_k_eq = tail_stiff_scale *
        tail_equivalent_stiffness(tail_q_eff, eel_data->tail_k0, eel_data->tail_a, eel_data->tail_b);
    const double tail_restore = tail_stiff_scale *
        tail_nonlinear_restoring_force(tail_q_eff, eel_data->tail_k0, eel_data->tail_a, eel_data->tail_b);

    const double speed_inst =
        std::sqrt(eel_data->u_com_current * eel_data->u_com_current + eel_data->v_com_current * eel_data->v_com_current);
    const double dt_diag =
        eel_data->diagnostics_initialized ? std::max(0.0, loop_time - eel_data->diagnostics_prev_time) : 0.0;
    const double tail_potential =
        tail_stiff_scale * tail_potential_energy(tail_q_eff, eel_data->tail_k0, eel_data->tail_a, eel_data->tail_b);
    const double tail_potential_rate =
        (dt_diag > std::numeric_limits<double>::epsilon()) ? (tail_potential - eel_data->tail_potential_prev) / dt_diag : 0.0;

    const double pout_inst = compute_output_power(eel_data->hydrodynamic_force_x,
                                                  eel_data->hydrodynamic_force_y,
                                                  eel_data->hydrodynamic_torque_z,
                                                  eel_data->u_com_current,
                                                  eel_data->v_com_current,
                                                  eel_data->omega_body_current);
    const double pin_inst = compute_input_power(pout_inst, tail_potential_rate);
    const double mean_speed = compute_mean_swimming_speed(speed_inst, dt_diag, *eel_data);

    if (dt_diag > 0.0)
    {
        eel_data->pout_integral += pout_inst * dt_diag;
        eel_data->pin_integral += pin_inst * dt_diag;
    }

    const double avg_time = std::max(eel_data->mean_speed_time, std::numeric_limits<double>::epsilon());
    const double pout_avg = eel_data->pout_integral / avg_time;
    const double pin_avg = eel_data->pin_integral / avg_time;
    const double eta_f = compute_froude_efficiency(pout_avg, pin_avg);
    const double cot = compute_cost_of_transport(pin_avg, eel_data->body_mass, mean_speed);
    const double rho = input_db->getDoubleWithDefault("RHO", 1.0);
    const double mu = input_db->getDoubleWithDefault("MU", 1.0);
    const double nu = (rho > std::numeric_limits<double>::epsilon()) ? mu / rho : 0.0;
    const double f_tail = std::abs(eel_data->omega) / (2.0 * PI_VAL);
    const double reynolds = compute_reynolds_number(mean_speed, eel_data->L, nu);
    const double strouhal = compute_strouhal_number(f_tail, AF, mean_speed);
    const int steady_flag = is_self_propelled_steady_state(*eel_data) ? 1 : 0;

    eel_data->diagnostics_prev_time = loop_time;
    eel_data->tail_potential_prev = tail_potential;
    eel_data->diagnostics_initialized = true;

    const double wave_speed_ref = std::abs(eel_data->wave_frequency_f) * std::max(eel_data->wavelength_lambda, eel_data->L);
    const double U_ref = std::max(1.0e-8, std::max(wave_speed_ref, std::max(speed_inst, mean_speed)));
    const double D_ref = std::max(eel_data->L, 1.0e-8);
    const double coeff_denom = 0.5 * rho * U_ref * U_ref * D_ref;
    if (IBTK_MPI::getRank() == 0)
    {
        drag_stream << loop_time << " " << -eel_data->hydrodynamic_force_x / coeff_denom << endl;
        lift_stream << loop_time << " " << -eel_data->hydrodynamic_force_y / coeff_denom << endl;
        pose_stream << iteration_num << " " << loop_time << " " << xcom << " " << ycom << " " << theta << " " << tail_y << endl;
        swim_diag_stream << iteration_num << " " << loop_time << " " << eel_data->u_com_current << " "
                         << eel_data->v_com_current << " " << eel_data->omega_body_current << " "
                         << actuation_fx << " " << actuation_fy << " " << actuation_torque << " "
                         << eel_data->hydrodynamic_force_x << " " << eel_data->hydrodynamic_force_y << " "
                         << eel_data->hydrodynamic_torque_z << " " << tail_alpha << " " << alpha_max_cycle << " "
                         << tail_q << " " << AF << " " << pout_inst << " " << pin_inst << " " << pout_avg << " "
                         << pin_avg << " " << eta_f << " " << cot << " " << mean_speed << " " << reynolds << " "
                         << strouhal << " " << tail_k_eq << " " << tail_restore << " " << steady_flag << endl;
    }
    return;
} // postprocess_data

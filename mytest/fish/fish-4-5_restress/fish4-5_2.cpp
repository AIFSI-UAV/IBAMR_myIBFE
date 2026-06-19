// IBFE hyperelastic continuum fish: B-kappa target active moment mapped by
// FE-normalized stress. Passive stiffness is supplied by raw
// deviatoric/dilatational PK1 stress, with optional continuum damping.

#include <SAMRAI_config.h>
#include <petscsys.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/multi_array.hpp>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/boundary_info.h>
#include <libmesh/const_function.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/dense_vector.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/elem.h>
#include <libmesh/fe.h>
#include <libmesh/fe_base.h>
#include <libmesh/fe_type.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/sparse_matrix.h>
#include <libmesh/mesh.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFECentroidPostProcessor.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <ibamr/app_namespaces.h>

// =========================================================================
// Model parameters
// =========================================================================
namespace ModelData
{

// --- Passive elasticity ---
static double fluid_density = 1.0;
static double fluid_viscosity = 0.0008;
static double c1_s_passive = 0.5;
static double c1_s_passive_anterior = 0.5;
static double c1_s_passive_peduncle = 0.5;
static double c1_s_passive_caudal = 0.5;
static double c1_s_body_transition_s = 0.60;
static double c1_s_body_transition_w = 0.30;
static double c1_s_caudal_transition_s = 0.85;
static double c1_s_caudal_transition_w = 0.10;
static double kappa_vol = 20.0;
static bool   use_continuum_damping = false;
static double continuum_damping_factor = 0.005;
static double continuum_damping_stress_cap_over_c1 = 50.0;

// --- Dedicated B-kappa + FE-normalized active model ---
// B(s) is an active moment scaling reference only. The raw dev/dil passive
// stiffness above is not calibrated from these values. This executable does
// not provide the legacy beta*h^2 or old-stress branches.
static double target_bending_B_body = 1.0e-4;
static double target_bending_B_peduncle = 1.0e-4;
static double target_bending_B_caudal = 1.0e-4;
static double target_bending_B_body_transition_s = 0.60;
static double target_bending_B_body_transition_w = 0.30;
static double target_bending_B_caudal_transition_s = 0.85;
static double target_bending_B_caudal_transition_w = 0.10;
static int    fe_normalized_stress_bins = 128;
static std::string fe_normalized_stress_quad_order = "FIFTH";

// --- Geometry / actuation frequency ---
static double fish_length = 1.00;
static double x_leading   = 1.00;
static double wave_frequency = 1.00;
static double wave_ramp_time = 4.0;
static double wave_omega     = 2.0 * M_PI * 1.00;
static double wave_time_sign = 1.0;

static double active_kappa_amp = 1.0;
enum class ActiveMomentMode { TRAVELING, STATIC };
static ActiveMomentMode active_moment_mode = ActiveMomentMode::TRAVELING;
static double static_moment_m0 = 0.0;
static double initial_bend_amplitude = 0.0;
static double active_wavelength_over_L = 1.00;
static double active_phase0 = 0.0;
static double active_moment_to_stress_sign = -1.0;
enum class ActiveKShapeMode { HALF_BELL, BELL };
static ActiveKShapeMode active_k_shape_mode = ActiveKShapeMode::BELL;
enum class ActiveEnvelopeMode
{
    K_SHAPE,
    POSTERIOR_RAMP,
    POSTERIOR_B_COMPENSATED
};
static ActiveEnvelopeMode active_envelope_mode =
    ActiveEnvelopeMode::K_SHAPE;
static double posterior_g0 = 0.15;
static double posterior_power = 1.75;
static double b_compensation_alpha = 0.5;
static double tail_taper_width = 0.06;
static double active_envelope_cap_safe_over_E = 0.0;
static double active_s_start  = 0.05;
static double active_s_end    = -1.0;
static double active_s_smooth = 0.05;
static double active_band_fraction = 1.0;
static double active_i2_h_power = 3.0;
static double active_end_s_norm =
    std::numeric_limits<double>::quiet_NaN();
static double active_t_act_max_over_c1 = 2.0;
static int    reference_profile_bins = 128;
static std::string wave_head_location = "x_min";
static double reference_backbone_end_x =
    std::numeric_limits<double>::quiet_NaN(); // auto-detect fork root unless set
static bool   use_laplace_reference_parameterization = true;
static bool   use_fe_active_section_data = false;
static double fe_section_i2_floor_ratio = 0.20;
static double laplace_head_bc_width_over_L = 0.05;
static double laplace_tail_bc_width_over_L = 0.05;
static const boundary_id_type LAPLACE_HEAD_BID = 12001;
static const boundary_id_type LAPLACE_TAIL_BID = 12002;

// --- Reference profile extracted from fish2d.msh ---
static double ref_x_min        = -0.10;
static double ref_x_max        = 1.00;
static double ref_body_length  = 1.10;
static double ref_arc_length   = 1.10;
static double ref_h_max        = 0.0;
static double ref_backbone_end_x = 1.00;
static std::vector<double> ref_profile_x;
static std::vector<double> ref_profile_s;
static std::vector<double> ref_centerline_y;
static std::vector<double> ref_halfthickness;
static std::vector<double> ref_halfthickness_raw;   // strict raw upper envelope before smoothing
static std::vector<double> ref_halfthickness_pchip_deriv;
static std::vector<libMesh::Point> ref_centerline_nodes;

struct CenterlineSegment
{
    libMesh::Point X0;
    libMesh::Point X1;
    VectorValue<double> t_hat = VectorValue<double>(1.0, 0.0);
    VectorValue<double> n_hat = VectorValue<double>(0.0, 1.0);
    double len = 0.0;
    double s0  = 0.0;
};

struct ProjectionResult
{
    bool valid = false;
    double s = 0.0;
    double eta = 0.0;
    VectorValue<double> t_hat = VectorValue<double>(1.0, 0.0);
};

static const std::string REF_GEOM_SYSTEM_NAME = "ref_geom";
enum RefGeomVar
{
    REF_GEOM_T_X = 0,
    REF_GEOM_T_Y = 1,
    REF_GEOM_ETA = 2,
    REF_GEOM_S = 3,
    REF_GEOM_N_VARS = 4
};

struct ReferenceGeometrySample
{
    double s = 0.0;
    double eta = 0.0;
    VectorValue<double> t_hat = VectorValue<double>(1.0, 0.0);
};

struct FENormalizedSection
{
    bool valid = false;
    double s_mid = 0.0;
    double g_mean = 0.0;
    double q_scale = 0.0;
    double q_abs_max = 0.0;
    double unit_force = 0.0;
    double unit_moment = 0.0;
};

static std::vector<CenterlineSegment> ref_centerline_segments;
static std::map<dof_id_type, ReferenceGeometrySample> ref_laplace_node_geom;
static bool ref_laplace_parameterization_built = false;
static std::vector<FENormalizedSection> fe_normalized_sections;

struct PhiIsoSectionSample
{
    bool valid = false;
    double s_norm = 0.0;
    double s = 0.0;
    double halfthickness = 0.0;
    libMesh::Point X_mid = libMesh::Point(0.0, 0.0);
    VectorValue<double> t_hat = VectorValue<double>(1.0, 0.0);
    VectorValue<double> n_hat = VectorValue<double>(0.0, 1.0);
};

static std::vector<PhiIsoSectionSample> ref_phi_sections;

// --- Tail diagnostics tracking points on the reference mesh ---
static dof_id_type ref_tail_upper_node_id =
    std::numeric_limits<dof_id_type>::max();
static dof_id_type ref_tail_lower_node_id =
    std::numeric_limits<dof_id_type>::max();
static libMesh::Point ref_tail_tip_center_point = libMesh::Point(1.0, 0.0);

// Fin pitch tracking: boundary nodes at the reference backbone end (fin root).
static dof_id_type ref_fin_root_upper_node_id =
    std::numeric_limits<dof_id_type>::max();
static dof_id_type ref_fin_root_lower_node_id =
    std::numeric_limits<dof_id_type>::max();
static libMesh::Point ref_fin_root_center_ref = libMesh::Point(0.1, 0.0);

// Fin pitch diagnostic state
static bool        s_fin_pitch_diag_enable   = false;
static int         s_fin_pitch_diag_interval = 100;
static std::string s_fin_pitch_diag_filename = "fin_pitch_diag.csv";
static double      s_fin_pitch_prev_y_root   = std::numeric_limits<double>::quiet_NaN();
static double      s_fin_pitch_prev_y_tip    = std::numeric_limits<double>::quiet_NaN();
static double      s_fin_pitch_prev_time     = std::numeric_limits<double>::quiet_NaN();

static double xcom_tracked = 0.0;   // last COM x sample used by diagnostics
static double ycom_tracked = 0.0;   // last COM y sample used by diagnostics
static double xcom_tracked_time =
    std::numeric_limits<double>::quiet_NaN(); // time of the stored COM sample
static double xcom_vel     = 0.0;   // finite-difference COM x-velocity
static double ycom_vel     = 0.0;

// -----------------------------------------------------------------------
// Diagnostics file
// -----------------------------------------------------------------------
static bool        s_curvature_phase_diag_enable = true;
static int         s_curvature_phase_diag_interval = 1;
static std::string s_curvature_phase_diag_filename = "curvature_phase_diag.csv";
static int         s_curvature_phase_diag_stations = 101;
static double      s_curvature_phase_diag_start_time =
    std::numeric_limits<double>::quiet_NaN();
static int         s_progress_print_interval = 10;

static double s_reference_xcom = 0.0;
static double s_reference_ycom = 0.0;
static double s_reference_area = 0.0;
static std::vector<double> s_curvature_cos_accum;
static std::vector<double> s_curvature_sin_accum;
static std::vector<double> s_activation_cos_accum;
static std::vector<double> s_activation_sin_accum;
static std::vector<double> s_prev_curvature_body;
static double s_curvature_phase_accum_time = 0.0;
static double s_curvature_phase_last_time =
    std::numeric_limits<double>::quiet_NaN();
static double s_curvature_phase_positive_work_accum = 0.0;
static double s_curvature_phase_signed_work_accum = 0.0;
static double s_curvature_phase_effective_positive_work_accum = 0.0;
static double s_curvature_phase_effective_signed_work_accum = 0.0;
static double s_curvature_phase_last_power =
    std::numeric_limits<double>::quiet_NaN();
static double s_curvature_phase_last_positive_power =
    std::numeric_limits<double>::quiet_NaN();
static double s_curvature_phase_last_effective_power =
    std::numeric_limits<double>::quiet_NaN();
static double s_curvature_phase_last_effective_positive_power =
    std::numeric_limits<double>::quiet_NaN();
static int s_curvature_phase_samples = 0;

// ── Body displacement amplitude envelope (per curvature station, per beat) ──
// Tracks peak-to-peak y_body at each station; resets every integer beat cycle.
// A_body_norm[k] = (max-min)/2 / Lref from the last completed beat cycle.
static std::vector<double> s_y_body_cyc_max; // running max y_body per station
static std::vector<double> s_y_body_cyc_min; // running min y_body per station
static std::vector<double> s_y_body_amp;     // half-amplitude from last completed cycle
static int s_y_body_cyc_idx = -1;            // integer cycle index for reset detection

// ── Minimal direction diagnostics ─────────────────────────────────────────
static bool        s_direction_debug_enable   = false;
static int         s_direction_debug_interval = 100;
static std::string s_direction_debug_filename = "direction_debug.csv";
static double      s_direction_debug_prev_time =
    std::numeric_limits<double>::quiet_NaN();
static double      s_direction_debug_prev_v_forward =
    std::numeric_limits<double>::quiet_NaN();
// F_IB-based impulse (kept for reference, NOT the true CM impulse):
static double      s_direction_debug_fish_impulse_forward = 0.0;
static double      s_direction_debug_fish_work_forward = 0.0;
// Correct CM-based impulse = integral of (fluid_density * fish_area * a_cm):
static double      s_direction_debug_cm_impulse_fwd = 0.0;
// Sliding window for beat-cycle averaged v_forward (~1.2 cycles at dt_diag=0.05s, f=1.2Hz)
static const int   VFWD_WIN = 20;
static double      s_vfwd_win[VFWD_WIN] = {};
static int         s_vfwd_win_pos   = 0;
static int         s_vfwd_win_count = 0;

// ── FE-consistent active section data: eta_bar(s), I2_c(s) ───────────────
// Precomputed from real FE quadrature so that T_act uses eta_c = eta - eta_bar
// and I2_c = Σ w*eta_c² JxW, ensuring the discrete zero-resultant property.
static bool        s_fe_section_data_built = false;
static int         s_fe_n_bins             = 0;   // set to reference_profile_bins at build time
static std::vector<double> s_fe_sum_w;       // Σ w(eta)*JxW per s-norm bin
static std::vector<double> s_fe_sum_w_eta;   // Σ w(eta)*eta*JxW per s-norm bin
static std::vector<double> s_fe_sum_w_eta2;  // Σ w(eta)*eta²*JxW per s-norm bin
static std::vector<double> s_fe_eta_bar;     // centroid: sum_w_eta/sum_w per bin
static std::vector<double> s_fe_I2_c;        // section I2_c = (sum_w_eta2 - eta_bar*sum_w_eta)/ds_bin
static std::string s_fe_section_diag_filename = "fe_active_section.csv";

// ── Force decomposition diagnostics ───────────────────────────────────────
static bool        s_force_decomp_diag_enable   = true;
static int         s_force_decomp_diag_interval = 100;
static std::string s_force_decomp_diag_filename = "force_decomposition_diag.csv";
static std::string s_force_decomp_quad_order    = "FIFTH";
static double      s_force_decomp_prev_time =
    std::numeric_limits<double>::quiet_NaN();
static double      s_force_decomp_work_dev      = 0.0;
static double      s_force_decomp_work_dil      = 0.0;
static double      s_force_decomp_work_damping  = 0.0;
static double      s_force_decomp_work_active   = 0.0;
static double      s_force_decomp_work_sum      = 0.0;
static double      s_force_decomp_work_weak_dev      = 0.0;
static double      s_force_decomp_work_weak_dil      = 0.0;
static double      s_force_decomp_work_weak_damping  = 0.0;
static double      s_force_decomp_work_weak_active   = 0.0;
static double      s_force_decomp_work_weak_sum      = 0.0;
static double      s_force_decomp_work_ib_on_fluid   = 0.0;

// ── Section moment diagnostics ───────────────────────────────────────────
static bool        s_section_moment_diag_enable   = true;
static int         s_section_moment_diag_interval = 100;
static std::string s_section_moment_diag_filename = "section_moment_decomposition.csv";
static std::string s_section_moment_quad_order    = "FIFTH";
static int         s_section_moment_diag_bins     = 0; // <=0 uses reference_profile_bins

static bool        s_active_moment_profile_debug_enable = false;
static std::string s_active_moment_profile_debug_filename =
    "active_moment_profile_debug.csv";

// ── Geometry conservation diagnostics ────────────────────────────────────
static bool        s_geometry_conservation_diag_enable   = true;
static int         s_geometry_conservation_diag_interval = 100;
static std::string s_geometry_conservation_diag_filename = "geometry_conservation_diag.csv";

// ── Midline history CSV ───────────────────────────────────────────────────────
// Full per-station midline snapshot at regular step intervals. Columns include
// both body-frame deformation (y_body) and propulsion-pattern lateral motion
// (y_prop = y_lab), which preserve recoil/heave for COD and Fourier analysis.
static bool        s_midline_hist_enable      = false;
static int         s_midline_hist_interval    = 10;        // write every N timesteps
static int         s_midline_hist_stations    = 101;
static std::string s_midline_hist_filename    = "midline_history.csv";
static bool        s_midline_hist_header_done = false;

// =========================================================================
// Helper: cosine ramp
// =========================================================================
inline double clamp01(double x)
{
    return std::min(std::max(x, 0.0), 1.0);
}

inline double smoothstep(double x)
{
    x = clamp01(x);
    return x * x * (3.0 - 2.0 * x);
}

inline double smoothstep_cosine(const double s, const double s0, const double w)
{
    const double width = std::max(w, 1.0e-12);
    const double lo = s0 - 0.5 * width;
    const double hi = s0 + 0.5 * width;

    if (s <= lo) return 0.0;
    if (s >= hi) return 1.0;

    const double xi = (s - lo) / width;
    return 0.5 * (1.0 - std::cos(M_PI * xi));
}

inline double get_c1_s_passive_local(const double s_norm)
{
    const double xi = clamp01(s_norm);
    const double w_body =
        smoothstep_cosine(xi, c1_s_body_transition_s, c1_s_body_transition_w);
    const double w_caudal =
        smoothstep_cosine(xi, c1_s_caudal_transition_s, c1_s_caudal_transition_w);
    return c1_s_passive_anterior
           + (c1_s_passive_peduncle - c1_s_passive_anterior) * w_body
           + (c1_s_passive_caudal - c1_s_passive_peduncle) * w_caudal;
}

inline double get_target_bending_B_local(const double s_norm)
{
    const double xi = clamp01(s_norm);
    const double w_body =
        smoothstep_cosine(xi, target_bending_B_body_transition_s,
                          target_bending_B_body_transition_w);
    const double w_caudal =
        smoothstep_cosine(xi, target_bending_B_caudal_transition_s,
                          target_bending_B_caudal_transition_w);
    return target_bending_B_body
           + (target_bending_B_peduncle - target_bending_B_body) * w_body
           + (target_bending_B_caudal - target_bending_B_peduncle) * w_caudal;
}

inline double wave_ramp(double time)
{
    if (wave_ramp_time <= 0.0) return 1.0;
    if (time <= 0.0)           return 0.0;
    if (time >= wave_ramp_time) return 1.0;
    const double xi = time / wave_ramp_time;
    return 0.5 * (1.0 - std::cos(M_PI * xi));
}

// =========================================================================
// Helper: 1D linear interpolation on a monotone grid.
// =========================================================================
double linear_interp_1d(const std::vector<double>& xs, const std::vector<double>& vs, double xq)
{
    if (xs.size() != vs.size() || xs.size() < 2)
    {
        TBOX_ERROR("linear_interp_1d(): invalid lookup table.\n");
    }

    if (xq <= xs.front()) return vs.front();
    if (xq >= xs.back()) return vs.back();

    const auto upper_it = std::lower_bound(xs.begin(), xs.end(), xq);
    const std::size_t i1 = static_cast<std::size_t>(upper_it - xs.begin());
    const std::size_t i0 = i1 - 1;
    const double dx = xs[i1] - xs[i0];
    if (dx <= 0.0) return vs[i0];

    const double alpha = (xq - xs[i0]) / dx;
    return (1.0 - alpha) * vs[i0] + alpha * vs[i1];
}

double linear_interp_monotone_1d(const std::vector<double>& xs,
                                 const std::vector<double>& vs,
                                 const double xq)
{
    if (xs.size() != vs.size() || xs.size() < 2)
    {
        TBOX_ERROR("linear_interp_monotone_1d(): invalid lookup table.\n");
    }

    if (xs.front() <= xs.back()) return linear_interp_1d(xs, vs, xq);

    if (xq >= xs.front()) return vs.front();
    if (xq <= xs.back()) return vs.back();

    for (std::size_t i1 = 1; i1 < xs.size(); ++i1)
    {
        if (xq >= xs[i1])
        {
            const std::size_t i0 = i1 - 1;
            const double dx = xs[i1] - xs[i0];
            if (std::abs(dx) <= 1.0e-24) return vs[i0];

            const double alpha = (xq - xs[i0]) / dx;
            return (1.0 - alpha) * vs[i0] + alpha * vs[i1];
        }
    }

    return vs.back();
}

// =========================================================================
// Reference geometry from the true boundary backbone:
//   1. Extract the closed boundary loop directly from the mesh boundary.
//   2. Split it into upper/lower head-to-tail boundary chains.
//   3. Resample both chains with cubic-Hermite interpolation.
//   4. Reconstruct the backbone as the midpoint curve between the two
//      resampled chains, then rebuild h(s) from point-to-backbone projection.
//
// This removes the x-bin seed entirely: the backbone source is now the real
// boundary geometry instead of vertical x slices through the body mesh.
// =========================================================================
inline double point_distance(const libMesh::Point& X0, const libMesh::Point& X1)
{
    const double dx = X1(0) - X0(0);
    const double dy = X1(1) - X0(1);
    return std::sqrt(dx * dx + dy * dy);
}

inline libMesh::Point midpoint_point(const libMesh::Point& X0, const libMesh::Point& X1)
{
    return libMesh::Point(0.5 * (X0(0) + X1(0)), 0.5 * (X0(1) + X1(1)));
}

std::vector<double> build_polyline_abscissa(const std::vector<libMesh::Point>& pts)
{
    std::vector<double> s(pts.size(), 0.0);
    for (std::size_t k = 1; k < pts.size(); ++k)
    {
        s[k] = s[k - 1] + point_distance(pts[k - 1], pts[k]);
    }
    return s;
}

std::vector<VectorValue<double> > compute_polyline_tangents(const std::vector<libMesh::Point>& pts,
                                                            const std::vector<double>& s)
{
    const std::size_t n = pts.size();
    std::vector<VectorValue<double> > tangents(n, VectorValue<double>(1.0, 0.0));
    if (n < 2) return tangents;

    for (std::size_t k = 0; k < n; ++k)
    {
        std::size_t km = (k == 0 ? 0 : k - 1);
        std::size_t kp = (k + 1 < n ? k + 1 : n - 1);
        if (k == 0) kp = 1;
        if (k + 1 == n) km = n - 2;

        const double ds = std::max(s[kp] - s[km], 1.0e-24);
        tangents[k] = VectorValue<double>((pts[kp](0) - pts[km](0)) / ds,
                                          (pts[kp](1) - pts[km](1)) / ds);
    }

    return tangents;
}

libMesh::Point cubic_hermite_point(const libMesh::Point& X0,
                                   const libMesh::Point& X1,
                                   const VectorValue<double>& M0,
                                   const VectorValue<double>& M1,
                                   const double ds,
                                   const double xi)
{
    const double h00 =  2.0 * xi * xi * xi - 3.0 * xi * xi + 1.0;
    const double h10 =        xi * xi * xi - 2.0 * xi * xi + xi;
    const double h01 = -2.0 * xi * xi * xi + 3.0 * xi * xi;
    const double h11 =        xi * xi * xi -       xi * xi;

    return libMesh::Point(h00 * X0(0) + h10 * ds * M0(0) + h01 * X1(0) + h11 * ds * M1(0),
                          h00 * X0(1) + h10 * ds * M0(1) + h01 * X1(1) + h11 * ds * M1(1));
}

libMesh::Point sample_polyline_cubic_hermite(const std::vector<libMesh::Point>& pts,
                                             const std::vector<double>& s,
                                             const std::vector<VectorValue<double> >& tangents,
                                             const double sq)
{
    const std::size_t n = pts.size();
    if (n == 0) return libMesh::Point();
    if (n == 1) return pts.front();
    if (sq <= s.front()) return pts.front();
    if (sq >= s.back()) return pts.back();

    const auto upper_it = std::lower_bound(s.begin(), s.end(), sq);
    const std::size_t i1 = static_cast<std::size_t>(upper_it - s.begin());
    const std::size_t i0 = i1 - 1;
    const double ds = std::max(s[i1] - s[i0], 1.0e-24);
    const double xi = (sq - s[i0]) / ds;

    return cubic_hermite_point(pts[i0], pts[i1], tangents[i0], tangents[i1], ds, xi);
}

void truncate_centerline_nodes_at_x(const double x_end)
{
    if (ref_centerline_nodes.size() < 2) return;

    const double x_head = ref_centerline_nodes.front()(0);
    const double x_tail_full = ref_centerline_nodes.back()(0);
    const bool increasing_x = x_tail_full >= x_head;
    const double x_tail =
        increasing_x ? std::min(std::max(x_end, x_head), x_tail_full) :
                       std::max(std::min(x_end, x_head), x_tail_full);
    std::vector<libMesh::Point> truncated;
    truncated.reserve(ref_centerline_nodes.size());
    truncated.push_back(ref_centerline_nodes.front());

    for (std::size_t k = 1; k < ref_centerline_nodes.size(); ++k)
    {
        const libMesh::Point& X0 = ref_centerline_nodes[k - 1];
        const libMesh::Point& X1 = ref_centerline_nodes[k];

        const bool before_tail =
            increasing_x ? (X1(0) < x_tail - 1.0e-12) :
                           (X1(0) > x_tail + 1.0e-12);
        if (before_tail)
        {
            truncated.push_back(X1);
            continue;
        }

        if (std::abs(X1(0) - x_tail) <= 1.0e-12)
        {
            truncated.push_back(X1);
        }
        else if ((increasing_x && X0(0) < x_tail && x_tail < X1(0)) ||
                 (!increasing_x && X1(0) < x_tail && x_tail < X0(0)))
        {
            const double dx = X1(0) - X0(0);
            const double alpha = std::abs(dx) > 1.0e-24 ? (x_tail - X0(0)) / dx : 0.0;
            truncated.push_back(libMesh::Point(x_tail,
                                               (1.0 - alpha) * X0(1) + alpha * X1(1)));
        }
        break;
    }

    if (truncated.size() < 2)
    {
        TBOX_ERROR("truncate_centerline_nodes_at_x(): truncated backbone is degenerate.\n");
    }

    ref_centerline_nodes.swap(truncated);
    ref_backbone_end_x = ref_centerline_nodes.back()(0);

    ref_profile_x.assign(ref_centerline_nodes.size(), 0.0);
    ref_centerline_y.assign(ref_centerline_nodes.size(), 0.0);
    for (std::size_t k = 0; k < ref_centerline_nodes.size(); ++k)
    {
        ref_profile_x[k] = ref_centerline_nodes[k](0);
        ref_centerline_y[k] = ref_centerline_nodes[k](1);
    }
}

void refresh_reference_profile_xy_from_centerline()
{
    ref_profile_x.assign(ref_centerline_nodes.size(), 0.0);
    ref_centerline_y.assign(ref_centerline_nodes.size(), 0.0);
    for (std::size_t k = 0; k < ref_centerline_nodes.size(); ++k)
    {
        ref_profile_x[k] = ref_centerline_nodes[k](0);
        ref_centerline_y[k] = ref_centerline_nodes[k](1);
    }
}

void update_head_location_from_x_leading()
{
    const double dist_to_x_min = std::abs(x_leading - ref_x_min);
    const double dist_to_x_max = std::abs(x_leading - ref_x_max);
    const double edge_tol = 1.0e-6 * std::max(1.0, ref_body_length);

    if (std::min(dist_to_x_min, dist_to_x_max) > edge_tol)
    {
        TBOX_WARNING("X_LEADING (" << x_leading
                     << ") is not close to either mesh x edge ["
                     << ref_x_min << ", " << ref_x_max
                     << "]. Using the nearest edge as the fish head.\n");
    }

    wave_head_location = (dist_to_x_min <= dist_to_x_max) ? "x_min" : "x_max";
}

inline bool head_is_at_x_min()
{
    return wave_head_location != "x_max";
}

void extend_reference_centerline_to_tail_tip()
{
    if (ref_centerline_nodes.size() < 2) return;

    const libMesh::Point X0 = ref_centerline_nodes.back();
    const libMesh::Point X1 = ref_tail_tip_center_point;
    const double d = point_distance(X0, X1);
    if (d <= 1.0e-10 * std::max(1.0, ref_body_length)) return;

    const int n_extra = std::max(2, static_cast<int>(
        std::ceil(static_cast<double>(reference_profile_bins) *
                  d / std::max(ref_body_length, 1.0e-12))));
    for (int k = 1; k <= n_extra; ++k)
    {
        const double a = static_cast<double>(k) / static_cast<double>(n_extra);
        ref_centerline_nodes.push_back(
            libMesh::Point((1.0 - a) * X0(0) + a * X1(0),
                           (1.0 - a) * X0(1) + a * X1(1)));
    }
    refresh_reference_profile_xy_from_centerline();
}

inline double reference_x_norm_from_point(const libMesh::Point& X_ref)
{
    const double Lx = std::max(fish_length, 1.0e-12);
    if (head_is_at_x_min())
    {
        return clamp01((X_ref(0) - x_leading) / Lx);
    }
    else
    {
        return clamp01((x_leading - X_ref(0)) / Lx);
    }
}

inline double approximate_s_norm_from_x(const double x_query)
{
    if (ref_profile_x.size() >= 2 && ref_profile_s.size() == ref_profile_x.size())
    {
        if (x_query <= ref_profile_x.front()) return 0.0;
        if (x_query >= ref_profile_x.back()) return clamp01(ref_profile_s.back() / std::max(ref_arc_length, 1.0e-12));
        const auto it = std::lower_bound(ref_profile_x.begin(), ref_profile_x.end(), x_query);
        const std::size_t i1 = static_cast<std::size_t>(std::distance(ref_profile_x.begin(), it));
        const std::size_t i0 = (i1 > 0) ? i1 - 1 : 0;
        const double x0 = ref_profile_x[i0];
        const double x1 = ref_profile_x[i1];
        const double a = (std::abs(x1 - x0) > 1.0e-14) ? (x_query - x0) / (x1 - x0) : 0.0;
        const double s_interp = (1.0 - a) * ref_profile_s[i0] + a * ref_profile_s[i1];
        return clamp01(s_interp / std::max(ref_arc_length, 1.0e-12));
    }
    const double Lx = std::max(ref_x_max - ref_x_min, 1.0e-12);
    return head_is_at_x_min() ? clamp01((x_query - ref_x_min) / Lx) :
                                clamp01((ref_x_max - x_query) / Lx);
}

inline double active_end_s_norm_effective()
{
    if (std::isfinite(active_end_s_norm)) return clamp01(active_end_s_norm);
    if (use_laplace_reference_parameterization)
    {
        TBOX_ERROR("active_end_s_norm_effective(): reference backbone end was not "
                   "mapped through the Laplace reference field.\n");
        return 0.0;
    }
    return approximate_s_norm_from_x(ref_backbone_end_x);
}

inline double active_s_start_norm_effective()
{
    return clamp01(active_s_start);
}

inline double active_s_end_norm_effective()
{
    if (active_s_end < 0.0) return active_end_s_norm_effective();
    return clamp01(active_s_end);
}

inline double active_s_span_norm_effective()
{
    const double s0 = active_s_start_norm_effective();
    const double s1 = active_s_end_norm_effective();
    return std::max(s1 - s0, 0.0);
}

inline double active_xi_from_s_norm(const double s_norm_in)
{
    const double s0 = active_s_start_norm_effective();
    const double span = active_s_span_norm_effective();
    if (span <= 1.0e-12) return 0.0;
    return clamp01((clamp01(s_norm_in) - s0) / span);
}

std::vector<dof_id_type> trace_boundary_path(
    const std::map<dof_id_type, std::vector<dof_id_type> >& adjacency,
    const dof_id_type start_id,
    const dof_id_type next_id,
    const dof_id_type target_id)
{
    std::vector<dof_id_type> path;
    path.push_back(start_id);
    path.push_back(next_id);

    dof_id_type prev_id = start_id;
    dof_id_type cur_id = next_id;

    for (std::size_t iter = 0; iter < adjacency.size() + 2; ++iter)
    {
        if (cur_id == target_id) return path;

        const auto adj_it = adjacency.find(cur_id);
        if (adj_it == adjacency.end() || adj_it->second.size() != 2) return std::vector<dof_id_type>();

        const dof_id_type nbr0 = adj_it->second[0];
        const dof_id_type nbr1 = adj_it->second[1];
        const dof_id_type next_candidate = (nbr0 == prev_id ? nbr1 : nbr0);
        if (next_candidate == start_id) return std::vector<dof_id_type>();

        path.push_back(next_candidate);
        prev_id = cur_id;
        cur_id = next_candidate;
    }

    return std::vector<dof_id_type>();
}

double boundary_path_length(const std::vector<dof_id_type>& path,
                            const std::map<dof_id_type, libMesh::Point>& boundary_points)
{
    double length = 0.0;
    for (std::size_t k = 1; k < path.size(); ++k)
    {
        length += point_distance(boundary_points.at(path[k - 1]), boundary_points.at(path[k]));
    }
    return length;
}

std::vector<libMesh::Point> extract_shortest_boundary_chain(
    const std::map<dof_id_type, std::vector<dof_id_type> >& adjacency,
    const std::map<dof_id_type, libMesh::Point>& boundary_points,
    const dof_id_type start_id,
    const dof_id_type target_id)
{
    const auto start_it = adjacency.find(start_id);
    if (start_it == adjacency.end() || start_it->second.size() != 2)
    {
        TBOX_ERROR("extract_shortest_boundary_chain(): invalid boundary start node.\n");
    }

    std::vector<dof_id_type> best_path;
    double best_length = std::numeric_limits<double>::max();
    for (const dof_id_type next_id : start_it->second)
    {
        const std::vector<dof_id_type> path =
            trace_boundary_path(adjacency, start_id, next_id, target_id);
        if (path.empty() || path.back() != target_id) continue;

        const double length = boundary_path_length(path, boundary_points);
        if (length < best_length)
        {
            best_length = length;
            best_path = path;
        }
    }

    if (best_path.empty())
    {
        TBOX_ERROR("extract_shortest_boundary_chain(): failed to trace boundary chain.\n");
    }

    std::vector<libMesh::Point> chain;
    chain.reserve(best_path.size());
    for (const dof_id_type node_id : best_path)
    {
        chain.push_back(boundary_points.at(node_id));
    }
    return chain;
}

void build_reference_backbone_from_boundary(MeshBase& mesh)
{
    const int n = std::max(reference_profile_bins, 8);

    ref_x_min = std::numeric_limits<double>::max();
    ref_x_max = -std::numeric_limits<double>::max();

    std::map<dof_id_type, libMesh::Point> node_positions;
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        node_positions[node.id()] = libMesh::Point(node(0), node(1));
        ref_x_min = std::min(ref_x_min, node(0));
        ref_x_max = std::max(ref_x_max, node(0));
    }
    IBTK_MPI::minReduction(&ref_x_min, 1);
    IBTK_MPI::maxReduction(&ref_x_max, 1);
    ref_body_length = std::max(ref_x_max - ref_x_min, std::numeric_limits<double>::epsilon());

    std::set<std::pair<dof_id_type, dof_id_type> > unique_edges;
    std::map<dof_id_type, std::vector<dof_id_type> > boundary_adjacency;
    for (auto elem_it = mesh.elements_begin(); elem_it != mesh.elements_end(); ++elem_it)
    {
        Elem* elem = *elem_it;
        if (!elem || !elem->active()) continue;

        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            if (elem->neighbor_ptr(side) != nullptr) continue;

            std::unique_ptr<const Elem> side_elem = elem->build_side_ptr(side);
            if (!side_elem || side_elem->n_nodes() < 2) continue;

            for (unsigned int k = 1; k < side_elem->n_nodes(); ++k)
            {
                const dof_id_type a = side_elem->node_id(k - 1);
                const dof_id_type b = side_elem->node_id(k);
                const std::pair<dof_id_type, dof_id_type> edge = std::minmax(a, b);
                if (!unique_edges.insert(edge).second) continue;

                boundary_adjacency[a].push_back(b);
                boundary_adjacency[b].push_back(a);
            }
        }
    }

    if (boundary_adjacency.empty())
    {
        TBOX_ERROR("build_reference_backbone_from_boundary(): no boundary edges found.\n");
    }

    for (const auto& entry : boundary_adjacency)
    {
        if (entry.second.size() != 2)
        {
            TBOX_ERROR("build_reference_backbone_from_boundary(): boundary is not a simple closed loop.\n");
        }
    }

    update_head_location_from_x_leading();
    const bool head_at_x_min = head_is_at_x_min();

    dof_id_type head_id = std::numeric_limits<dof_id_type>::max();
    dof_id_type tail_upper_id = std::numeric_limits<dof_id_type>::max();
    dof_id_type tail_lower_id = std::numeric_limits<dof_id_type>::max();
    double head_x = head_at_x_min ? std::numeric_limits<double>::max() :
                                    -std::numeric_limits<double>::max();
    double head_y_abs = std::numeric_limits<double>::max();
    double tail_x = head_at_x_min ? -std::numeric_limits<double>::max() :
                                    std::numeric_limits<double>::max();
    double tail_upper_y = -std::numeric_limits<double>::max();
    double tail_lower_y =  std::numeric_limits<double>::max();
    double detected_midline_tail_x = std::numeric_limits<double>::quiet_NaN();
    const double midline_y_tol = 1.0e-8 * std::max(1.0, ref_body_length);

    for (const auto& entry : boundary_adjacency)
    {
        const dof_id_type node_id = entry.first;
        const libMesh::Point& X = node_positions.at(node_id);

        const bool more_headward =
            head_at_x_min ? (X(0) < head_x - 1.0e-14) :
                            (X(0) > head_x + 1.0e-14);
        if (more_headward ||
            (std::abs(X(0) - head_x) <= 1.0e-14 && std::abs(X(1)) < head_y_abs))
        {
            head_id = node_id;
            head_x = X(0);
            head_y_abs = std::abs(X(1));
        }

        const bool more_tailward =
            head_at_x_min ? (X(0) > tail_x + 1.0e-14) :
                            (X(0) < tail_x - 1.0e-14);
        if (more_tailward)
        {
            tail_x = X(0);
            tail_upper_y = X(1);
            tail_lower_y = X(1);
            tail_upper_id = node_id;
            tail_lower_id = node_id;
        }
        else if (std::abs(X(0) - tail_x) <= 1.0e-14)
        {
            if (X(1) > tail_upper_y)
            {
                tail_upper_y = X(1);
                tail_upper_id = node_id;
            }
            if (X(1) < tail_lower_y)
            {
                tail_lower_y = X(1);
                tail_lower_id = node_id;
            }
        }

        if (std::abs(X(1)) <= midline_y_tol)
        {
            if (!std::isfinite(detected_midline_tail_x))
            {
                detected_midline_tail_x = X(0);
            }
            else
            {
                detected_midline_tail_x =
                    head_at_x_min ? std::max(detected_midline_tail_x, X(0)) :
                                    std::min(detected_midline_tail_x, X(0));
            }
        }
    }

    if (head_id == std::numeric_limits<dof_id_type>::max() ||
        tail_upper_id == std::numeric_limits<dof_id_type>::max() ||
        tail_lower_id == std::numeric_limits<dof_id_type>::max())
    {
        TBOX_ERROR("build_reference_backbone_from_boundary(): failed to identify head/tail boundary nodes.\n");
    }
    ref_tail_tip_center_point =
        libMesh::Point(tail_x, 0.5 * (tail_upper_y + tail_lower_y));

    const double backbone_end_x_requested =
        std::isfinite(reference_backbone_end_x) ? reference_backbone_end_x :
        (std::isfinite(detected_midline_tail_x) ? detected_midline_tail_x : tail_x);
    const double x_body_lo = std::min(head_x, tail_x);
    const double x_body_up = std::max(head_x, tail_x);
    const double backbone_end_x =
        std::min(std::max(backbone_end_x_requested, x_body_lo), x_body_up);

    const std::vector<libMesh::Point> upper_chain =
        extract_shortest_boundary_chain(boundary_adjacency, node_positions, head_id, tail_upper_id);
    const std::vector<libMesh::Point> lower_chain =
        extract_shortest_boundary_chain(boundary_adjacency, node_positions, head_id, tail_lower_id);

    const std::vector<double> upper_s = build_polyline_abscissa(upper_chain);
    const std::vector<double> lower_s = build_polyline_abscissa(lower_chain);
    const std::vector<VectorValue<double> > upper_tangents =
        compute_polyline_tangents(upper_chain, upper_s);
    const std::vector<VectorValue<double> > lower_tangents =
        compute_polyline_tangents(lower_chain, lower_s);

    ref_centerline_nodes.assign(std::size_t(n), libMesh::Point());
    ref_profile_x.assign(std::size_t(n), 0.0);
    ref_centerline_y.assign(std::size_t(n), 0.0);

    const double upper_len = std::max(upper_s.back(), 1.0e-24);
    const double lower_len = std::max(lower_s.back(), 1.0e-24);
    for (int k = 0; k < n; ++k)
    {
        const double xi = (n > 1 ? static_cast<double>(k) / static_cast<double>(n - 1) : 0.0);
        const libMesh::Point Xu =
            sample_polyline_cubic_hermite(upper_chain, upper_s, upper_tangents, xi * upper_len);
        const libMesh::Point Xl =
            sample_polyline_cubic_hermite(lower_chain, lower_s, lower_tangents, xi * lower_len);
        const libMesh::Point Xc = midpoint_point(Xu, Xl);

        ref_centerline_nodes[static_cast<std::size_t>(k)] = Xc;
        ref_profile_x[static_cast<std::size_t>(k)] = Xc(0);
        ref_centerline_y[static_cast<std::size_t>(k)] = Xc(1);
    }

    truncate_centerline_nodes_at_x(backbone_end_x);
}

void build_reference_centerline_segments()
{
    const int n = static_cast<int>(ref_centerline_nodes.size());
    if (n < 2)
    {
        TBOX_ERROR("build_reference_centerline_segments(): need at least 2 backbone nodes.\n");
    }

    ref_centerline_segments.clear();
    ref_profile_s.assign(std::size_t(n), 0.0);

    double s_accum = 0.0;
    for (int k = 0; k < n - 1; ++k)
    {
        const libMesh::Point& X0 = ref_centerline_nodes[static_cast<std::size_t>(k)];
        const libMesh::Point& X1 = ref_centerline_nodes[static_cast<std::size_t>(k + 1)];
        const double dx = X1(0) - X0(0);
        const double dy = X1(1) - X0(1);
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len <= 1.0e-12)
        {
            TBOX_ERROR("build_reference_centerline_segments(): degenerate backbone segment.\n");
        }

        CenterlineSegment seg;
        seg.X0 = X0;
        seg.X1 = X1;
        seg.t_hat = VectorValue<double>(dx / len, dy / len);
        seg.n_hat = VectorValue<double>(-seg.t_hat(1), seg.t_hat(0));
        seg.len = len;
        seg.s0 = s_accum;
        ref_centerline_segments.push_back(seg);

        s_accum += len;
        ref_profile_s[static_cast<std::size_t>(k + 1)] = s_accum;
    }

    ref_arc_length = std::max(s_accum, std::numeric_limits<double>::epsilon());
}

void sample_reference_centerline_at_s(const double s_query,
                                      libMesh::Point& X_out,
                                      VectorValue<double>& t_out)
{
    if (ref_centerline_segments.empty())
    {
        TBOX_ERROR("sample_reference_centerline_at_s(): reference centerline has not been built.\n");
    }

    const double s = std::max(0.0, std::min(s_query, ref_arc_length));
    const CenterlineSegment* seg = &ref_centerline_segments.front();
    if (s >= ref_arc_length)
    {
        seg = &ref_centerline_segments.back();
    }
    else
    {
        for (const CenterlineSegment& candidate : ref_centerline_segments)
        {
            if (s <= candidate.s0 + candidate.len + 1.0e-14)
            {
                seg = &candidate;
                break;
            }
        }
    }

    const double alpha =
        std::max(0.0, std::min((s - seg->s0) / std::max(seg->len, 1.0e-24), 1.0));
    X_out = libMesh::Point((1.0 - alpha) * seg->X0(0) + alpha * seg->X1(0),
                           (1.0 - alpha) * seg->X0(1) + alpha * seg->X1(1));
    t_out = seg->t_hat;
}

ProjectionResult project_to_reference_centerline(const libMesh::Point& X)
{
    ProjectionResult best_proj;
    double best_r2 = std::numeric_limits<double>::max();

    for (std::size_t k = 0; k < ref_centerline_segments.size(); ++k)
    {
        const CenterlineSegment& seg = ref_centerline_segments[k];
        const double dx = seg.X1(0) - seg.X0(0);
        const double dy = seg.X1(1) - seg.X0(1);
        const double inv_len2 = 1.0 / std::max(seg.len * seg.len, 1.0e-24);

        const double lambda_raw =
            ((X(0) - seg.X0(0)) * dx + (X(1) - seg.X0(1)) * dy) * inv_len2;
        const double lambda = clamp01(lambda_raw);
        const libMesh::Point P(seg.X0(0) + lambda * dx, seg.X0(1) + lambda * dy);
        const double rx = X(0) - P(0);
        const double ry = X(1) - P(1);
        const double r2 = rx * rx + ry * ry;

        if (r2 + 1.0e-14 < best_r2)
        {
            best_r2 = r2;
            best_proj.valid = true;
            best_proj.s = seg.s0 + lambda * seg.len;
            best_proj.eta = rx * seg.n_hat(0) + ry * seg.n_hat(1);
            best_proj.t_hat = seg.t_hat;
        }
    }

    return best_proj;
}

ReferenceGeometrySample reference_geometry_from_system_data(
    const std::vector<const std::vector<double>*>& system_var_data,
    const std::size_t system_idx = 0)
{
    if (system_var_data.size() <= system_idx ||
        system_var_data[system_idx] == nullptr ||
        system_var_data[system_idx]->size() < REF_GEOM_N_VARS)
    {
        TBOX_ERROR("reference geometry field is unavailable in PK1 callback.\n");
    }

    const std::vector<double>& data = *system_var_data[system_idx];
    ReferenceGeometrySample geom;
    geom.s = std::max(0.0, std::min(data[REF_GEOM_S], ref_arc_length));
    geom.eta = data[REF_GEOM_ETA];

    const VectorValue<double> t_raw(data[REF_GEOM_T_X], data[REF_GEOM_T_Y]);
    const double t_norm = std::sqrt(std::max(t_raw * t_raw, 0.0));
    if (t_norm <= 1.0e-14)
    {
        TBOX_ERROR("reference geometry field has a degenerate tangent.\n");
    }
    geom.t_hat = t_raw / t_norm;
    return geom;
}

void add_reference_geometry_system(EquationSystems* equation_systems)
{
    if (!equation_systems) return;

    ExplicitSystem& ref_geom_sys =
        equation_systems->add_system<ExplicitSystem>(REF_GEOM_SYSTEM_NAME);
    ref_geom_sys.add_variable("t_ref_x", FIRST, LAGRANGE);
    ref_geom_sys.add_variable("t_ref_y", FIRST, LAGRANGE);
    ref_geom_sys.add_variable("eta_ref", FIRST, LAGRANGE);
    ref_geom_sys.add_variable("s_ref", FIRST, LAGRANGE);
}

void fill_reference_geometry_system(MeshBase& mesh, EquationSystems* equation_systems)
{
    if (!equation_systems) return;

    ExplicitSystem& ref_geom_sys =
        equation_systems->get_system<ExplicitSystem>(REF_GEOM_SYSTEM_NAME);
    NumericVector<double>* ref_geom_vec = ref_geom_sys.solution.get();
    const DofMap& dof_map = ref_geom_sys.get_dof_map();

    std::vector<dof_id_type> dof_indices;
    double local_nodes = 0.0;
    double invalid_projection_count = 0.0;

    for (auto node_it = mesh.local_nodes_begin(); node_it != mesh.local_nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        ReferenceGeometrySample geom;
        bool have_geom = false;
        if (use_laplace_reference_parameterization && ref_laplace_parameterization_built)
        {
            const auto geom_it = ref_laplace_node_geom.find(node.id());
            if (geom_it != ref_laplace_node_geom.end())
            {
                geom = geom_it->second;
                have_geom = true;
            }
        }

        if (!have_geom)
        {
            if (use_laplace_reference_parameterization)
            {
                TBOX_ERROR("fill_reference_geometry_system(): missing Laplace reference geometry "
                           << "for node " << node.id() << ".\n");
            }
            else
            {
                const ProjectionResult proj = project_to_reference_centerline(
                    libMesh::Point(node(0), node(1), node(2)));
                if (proj.valid)
                {
                    geom.s = std::max(0.0, std::min(proj.s, ref_arc_length));
                    geom.eta = proj.eta;
                    geom.t_hat = proj.t_hat;
                    have_geom = true;
                }
                else
                {
                    invalid_projection_count += 1.0;
                }
            }
        }
        local_nodes += 1.0;

        dof_map.dof_indices(&node, dof_indices, REF_GEOM_T_X);
        ref_geom_vec->set(dof_indices[0], geom.t_hat(0));
        dof_map.dof_indices(&node, dof_indices, REF_GEOM_T_Y);
        ref_geom_vec->set(dof_indices[0], geom.t_hat(1));
        dof_map.dof_indices(&node, dof_indices, REF_GEOM_ETA);
        ref_geom_vec->set(dof_indices[0], geom.eta);
        dof_map.dof_indices(&node, dof_indices, REF_GEOM_S);
        ref_geom_vec->set(dof_indices[0], geom.s);
    }

    ref_geom_vec->close();
    ref_geom_sys.update();

    double global_counts[2] = { local_nodes, invalid_projection_count };
    IBTK_MPI::sumReduction(global_counts, 2);
    if (global_counts[1] > 0)
    {
        TBOX_WARNING("fill_reference_geometry_system(): "
                     << global_counts[1] << " / " << global_counts[0]
                     << " nodes could not be projected to the reference centerline.\n");
    }
}

inline double get_c1_s_passive_local_from_reference_point(const libMesh::Point& X_ref)
{
    return get_c1_s_passive_local(approximate_s_norm_from_x(X_ref(0)));
}

void add_halfthickness_sample_to_raw_envelope(std::vector<double>& raw_h,
                                              std::vector<double>& count,
                                              const double s,
                                              const double h_sample)
{
    const int n = static_cast<int>(ref_profile_s.size());
    if (n < 2 || raw_h.size() != ref_profile_s.size() || count.size() != ref_profile_s.size()) return;

    const double h = std::max(0.0, h_sample);
    if (s <= ref_profile_s.front())
    {
        raw_h.front() = std::max(raw_h.front(), h);
        count.front() += 1.0;
        return;
    }
    if (s >= ref_profile_s.back())
    {
        raw_h.back() = std::max(raw_h.back(), h);
        count.back() += 1.0;
        return;
    }

    const auto upper_it = std::upper_bound(ref_profile_s.begin(), ref_profile_s.end(), s);
    const int i1 = static_cast<int>(upper_it - ref_profile_s.begin());
    const int i0 = i1 - 1;

    raw_h[static_cast<std::size_t>(i0)] =
        std::max(raw_h[static_cast<std::size_t>(i0)], h);
    raw_h[static_cast<std::size_t>(i1)] =
        std::max(raw_h[static_cast<std::size_t>(i1)], h);
    count[static_cast<std::size_t>(i0)] += 1.0;
    count[static_cast<std::size_t>(i1)] += 1.0;
}

inline double pchip_endpoint_derivative(const double h0,
                                        const double h1,
                                        const double delta0,
                                        const double delta1)
{
    double d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / std::max(h0 + h1, 1.0e-24);
    if (d * delta0 <= 0.0)
    {
        d = 0.0;
    }
    else if (delta0 * delta1 < 0.0 && std::abs(d) > std::abs(3.0 * delta0))
    {
        d = 3.0 * delta0;
    }
    return d;
}

void rebuild_halfthickness_pchip_derivatives()
{
    const std::size_t n = ref_profile_s.size();
    ref_halfthickness_pchip_deriv.assign(n, 0.0);
    if (n != ref_halfthickness.size() || n < 2) return;

    std::vector<double> dx(n - 1, 0.0), delta(n - 1, 0.0);
    for (std::size_t k = 0; k + 1 < n; ++k)
    {
        dx[k] = std::max(ref_profile_s[k + 1] - ref_profile_s[k], 1.0e-24);
        delta[k] = (ref_halfthickness[k + 1] - ref_halfthickness[k]) / dx[k];
    }

    if (n == 2)
    {
        ref_halfthickness_pchip_deriv[0] = delta[0];
        ref_halfthickness_pchip_deriv[1] = delta[0];
        return;
    }

    ref_halfthickness_pchip_deriv[0] =
        pchip_endpoint_derivative(dx[0], dx[1], delta[0], delta[1]);
    for (std::size_t k = 1; k + 1 < n; ++k)
    {
        if (delta[k - 1] * delta[k] <= 0.0)
        {
            ref_halfthickness_pchip_deriv[k] = 0.0;
        }
        else
        {
            const double w1 = 2.0 * dx[k] + dx[k - 1];
            const double w2 = dx[k] + 2.0 * dx[k - 1];
            ref_halfthickness_pchip_deriv[k] =
                (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
        }
    }
    ref_halfthickness_pchip_deriv[n - 1] =
        pchip_endpoint_derivative(dx[n - 2], dx[n - 3], delta[n - 2], delta[n - 3]);
}

double pchip_interpolate_halfthickness(const double s)
{
    const std::size_t n = ref_profile_s.size();
    if (n < 2 || ref_halfthickness.size() != n ||
        ref_halfthickness_pchip_deriv.size() != n)
    {
        return linear_interp_1d(ref_profile_s, ref_halfthickness, s);
    }

    if (s <= ref_profile_s.front()) return ref_halfthickness.front();
    if (s >= ref_profile_s.back()) return ref_halfthickness.back();

    const auto upper_it = std::upper_bound(ref_profile_s.begin(), ref_profile_s.end(), s);
    const std::size_t i1 = static_cast<std::size_t>(upper_it - ref_profile_s.begin());
    const std::size_t i0 = i1 - 1;
    const double dx = std::max(ref_profile_s[i1] - ref_profile_s[i0], 1.0e-24);
    const double t = (s - ref_profile_s[i0]) / dx;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 =  2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 =        t3 - 2.0 * t2 + t;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 =        t3 -       t2;

    const double y0 = ref_halfthickness[i0];
    const double y1 = ref_halfthickness[i1];
    const double value =
        h00 * y0 + h10 * dx * ref_halfthickness_pchip_deriv[i0] +
        h01 * y1 + h11 * dx * ref_halfthickness_pchip_deriv[i1];

    return std::min(std::max(value, std::min(y0, y1)), std::max(y0, y1));
}

void finalize_reference_halfthickness_from_counts(std::vector<double>& count,
                                                  const char* caller_name)
{
    const int n = static_cast<int>(ref_profile_s.size());
    if (n < 2)
    {
        TBOX_ERROR(caller_name << ": invalid s grid.\n");
    }

    bool have_nonempty_bin = false;
    for (int k = 0; k < n; ++k)
    {
        if (count[static_cast<std::size_t>(k)] > 0.5)
        {
            have_nonempty_bin = true;
            continue;
        }

        int kl = k - 1;
        while (kl >= 0 && count[static_cast<std::size_t>(kl)] <= 0.5) --kl;
        int kr = k + 1;
        while (kr < n && count[static_cast<std::size_t>(kr)] <= 0.5) ++kr;

        if (kl >= 0 && kr < n)
        {
            const double alpha = (ref_profile_s[static_cast<std::size_t>(k)] -
                                  ref_profile_s[static_cast<std::size_t>(kl)]) /
                                 std::max(ref_profile_s[static_cast<std::size_t>(kr)] -
                                          ref_profile_s[static_cast<std::size_t>(kl)],
                                          1.0e-24);
            ref_halfthickness[static_cast<std::size_t>(k)] =
                (1.0 - alpha) * ref_halfthickness[static_cast<std::size_t>(kl)] +
                alpha * ref_halfthickness[static_cast<std::size_t>(kr)];
        }
        else if (kl >= 0)
        {
            ref_halfthickness[static_cast<std::size_t>(k)] =
                ref_halfthickness[static_cast<std::size_t>(kl)];
        }
        else if (kr < n)
        {
            ref_halfthickness[static_cast<std::size_t>(k)] =
                ref_halfthickness[static_cast<std::size_t>(kr)];
        }
    }

    if (!have_nonempty_bin)
    {
        TBOX_ERROR(caller_name << ": no occupied s bins.\n");
    }

    // Strict upper envelope of the mesh samples on the profile grid. Empty
    // bins have already been filled from neighboring occupied bins.
    ref_halfthickness_raw = ref_halfthickness;

    if (n > 2)
    {
        // Smooth the envelope, but never let smoothing undercut the raw
        // mesh-derived upper envelope. The active stress contains K(s)/I2(s);
        // this profile feeds h_scale(s) and the K/I2 diagnostics.
        for (int pass = 0; pass < 3; ++pass)
        {
            std::vector<double> smoothed = ref_halfthickness;
            smoothed.front() =
                std::max(smoothed.front(), ref_halfthickness_raw.front());
            smoothed.back() =
                std::max(smoothed.back(), ref_halfthickness_raw.back());
            for (int k = 1; k < n - 1; ++k)
            {
                const double candidate =
                    0.25 * ref_halfthickness[static_cast<std::size_t>(k - 1)] +
                    0.50 * ref_halfthickness[static_cast<std::size_t>(k)] +
                    0.25 * ref_halfthickness[static_cast<std::size_t>(k + 1)];
                smoothed[static_cast<std::size_t>(k)] =
                    std::max(candidate, ref_halfthickness_raw[static_cast<std::size_t>(k)]);
            }
            ref_halfthickness.swap(smoothed);
        }

        for (int k = 0; k < n; ++k)
        {
            ref_halfthickness[static_cast<std::size_t>(k)] =
                std::max(ref_halfthickness[static_cast<std::size_t>(k)],
                         ref_halfthickness_raw[static_cast<std::size_t>(k)]);
        }
    }

    ref_h_max = *std::max_element(ref_halfthickness.begin(), ref_halfthickness.end());
    if (ref_h_max <= 1.0e-12)
    {
        TBOX_ERROR(caller_name << ": h(s) is degenerate.\n");
    }

    rebuild_halfthickness_pchip_derivatives();
}


VectorValue<double> normalize_reference_vector(const VectorValue<double>& v,
                                               const VectorValue<double>& fallback)
{
    const double n = std::sqrt(std::max(v * v, 0.0));
    if (n > 1.0e-14) return v / n;
    const double nf = std::sqrt(std::max(fallback * fallback, 0.0));
    return (nf > 1.0e-14) ? fallback / nf : VectorValue<double>(1.0, 0.0);
}

libMesh::Point lerp_point(const libMesh::Point& A,
                          const libMesh::Point& B,
                          const double alpha)
{
    return libMesh::Point((1.0 - alpha) * A(0) + alpha * B(0),
                          (1.0 - alpha) * A(1) + alpha * B(1));
}

VectorValue<double> lerp_vector(const VectorValue<double>& A,
                                const VectorValue<double>& B,
                                const double alpha)
{
    return VectorValue<double>((1.0 - alpha) * A(0) + alpha * B(0),
                               (1.0 - alpha) * A(1) + alpha * B(1));
}

std::vector<std::pair<dof_id_type, dof_id_type> >
collect_boundary_edges_from_mesh(MeshBase& mesh)
{
    std::set<std::pair<dof_id_type, dof_id_type> > unique_edges;
    std::vector<std::pair<dof_id_type, dof_id_type> > edges;

    for (auto elem_it = mesh.elements_begin(); elem_it != mesh.elements_end(); ++elem_it)
    {
        Elem* elem = *elem_it;
        if (!elem || !elem->active()) continue;

        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            if (elem->neighbor_ptr(side) != nullptr) continue;

            std::unique_ptr<const Elem> side_elem = elem->build_side_ptr(side);
            if (!side_elem || side_elem->n_nodes() < 2) continue;

            for (unsigned int k = 1; k < side_elem->n_nodes(); ++k)
            {
                const dof_id_type a = side_elem->node_id(k - 1);
                const dof_id_type b = side_elem->node_id(k);
                if (a == b) continue;
                const auto edge = std::minmax(a, b);
                if (!unique_edges.insert(edge).second) continue;
                edges.push_back(edge);
            }
        }
    }

    if (edges.empty())
    {
        TBOX_ERROR("collect_boundary_edges_from_mesh(): no boundary edges found.\n");
    }
    return edges;
}

PhiIsoSectionSample interpolate_phi_section_sample(const double s_norm_query)
{
    if (ref_phi_sections.empty())
    {
        TBOX_ERROR("interpolate_phi_section_sample(): phi isocontour section table is empty.\n");
    }

    const double q = clamp01(s_norm_query);
    if (q <= ref_phi_sections.front().s_norm) return ref_phi_sections.front();
    if (q >= ref_phi_sections.back().s_norm) return ref_phi_sections.back();

    auto upper_it = std::upper_bound(
        ref_phi_sections.begin(), ref_phi_sections.end(), q,
        [](const double value, const PhiIsoSectionSample& sample)
        {
            return value < sample.s_norm;
        });
    const std::size_t i1 = static_cast<std::size_t>(upper_it - ref_phi_sections.begin());
    const std::size_t i0 = i1 - 1;
    const PhiIsoSectionSample& A = ref_phi_sections[i0];
    const PhiIsoSectionSample& B = ref_phi_sections[i1];
    const double denom = std::max(B.s_norm - A.s_norm, 1.0e-24);
    const double alpha = clamp01((q - A.s_norm) / denom);

    PhiIsoSectionSample out;
    out.valid = A.valid && B.valid;
    out.s_norm = q;
    out.s = (1.0 - alpha) * A.s + alpha * B.s;
    out.halfthickness = std::max(0.0, (1.0 - alpha) * A.halfthickness + alpha * B.halfthickness);
    out.X_mid = lerp_point(A.X_mid, B.X_mid, alpha);
    out.t_hat = normalize_reference_vector(lerp_vector(A.t_hat, B.t_hat, alpha), A.t_hat);
    out.n_hat = VectorValue<double>(-out.t_hat(1), out.t_hat(0));
    if ((out.n_hat * A.n_hat) < 0.0) out.n_hat *= -1.0;
    return out;
}

void rebuild_reference_centerline_from_phi_sections()
{
    const std::size_t n = ref_phi_sections.size();
    if (n < 2)
    {
        TBOX_ERROR("rebuild_reference_centerline_from_phi_sections(): need at least two sections.\n");
    }

    ref_centerline_nodes.assign(n, libMesh::Point(0.0, 0.0));
    ref_profile_x.assign(n, 0.0);
    ref_centerline_y.assign(n, 0.0);
    ref_profile_s.assign(n, 0.0);
    ref_halfthickness.assign(n, 0.0);
    ref_halfthickness_raw.assign(n, 0.0);
    ref_halfthickness_pchip_deriv.clear();

    for (std::size_t k = 0; k < n; ++k)
    {
        const PhiIsoSectionSample& sec = ref_phi_sections[k];
        ref_centerline_nodes[k] = sec.X_mid;
        ref_profile_x[k] = sec.X_mid(0);
        ref_centerline_y[k] = sec.X_mid(1);
        ref_profile_s[k] = sec.s;
        ref_halfthickness[k] = sec.halfthickness;
        ref_halfthickness_raw[k] = sec.halfthickness;
    }

    ref_arc_length = std::max(ref_profile_s.back() - ref_profile_s.front(), 1.0e-12);
    ref_h_max = *std::max_element(ref_halfthickness.begin(), ref_halfthickness.end());
    if (ref_h_max <= 1.0e-12)
    {
        TBOX_ERROR("rebuild_reference_centerline_from_phi_sections(): h(phi) is degenerate.\n");
    }

    ref_centerline_segments.clear();
    for (std::size_t k = 0; k + 1 < n; ++k)
    {
        CenterlineSegment seg;
        seg.X0 = ref_phi_sections[k].X_mid;
        seg.X1 = ref_phi_sections[k + 1].X_mid;
        seg.s0 = ref_phi_sections[k].s;
        seg.len = std::max(ref_phi_sections[k + 1].s - ref_phi_sections[k].s, 1.0e-12);
        seg.t_hat = normalize_reference_vector(
            lerp_vector(ref_phi_sections[k].t_hat, ref_phi_sections[k + 1].t_hat, 0.5),
            VectorValue<double>(seg.X1(0) - seg.X0(0), seg.X1(1) - seg.X0(1)));
        seg.n_hat = VectorValue<double>(-seg.t_hat(1), seg.t_hat(0));
        ref_centerline_segments.push_back(seg);
    }

    rebuild_halfthickness_pchip_derivatives();
}

void build_phi_isocontour_section_table(
    MeshBase& mesh,
    const std::map<dof_id_type, const Node*>& id_to_node,
    const std::map<dof_id_type, double>& phi_node,
    const std::map<dof_id_type, VectorValue<double> >& grad_phi_node)
{
    const int n_sections = std::max(reference_profile_bins, 8);
    const auto boundary_edges = collect_boundary_edges_from_mesh(mesh);
    std::vector<PhiIsoSectionSample> sections(static_cast<std::size_t>(n_sections));

    struct BoundaryIntersection
    {
        libMesh::Point X;
        VectorValue<double> grad;
    };

    const double phi_tol = 1.0e-12;
    const double dup_tol2 = 1.0e-20 * std::max(ref_body_length * ref_body_length, 1.0);

    for (int k = 0; k < n_sections; ++k)
    {
        const double s_norm = (n_sections > 1) ?
            static_cast<double>(k) / static_cast<double>(n_sections - 1) : 0.0;
        std::vector<BoundaryIntersection> intersections;

        for (const auto& edge : boundary_edges)
        {
            const dof_id_type id_a = edge.first;
            const dof_id_type id_b = edge.second;
            const auto node_a_it = id_to_node.find(id_a);
            const auto node_b_it = id_to_node.find(id_b);
            const auto phi_a_it = phi_node.find(id_a);
            const auto phi_b_it = phi_node.find(id_b);
            if (node_a_it == id_to_node.end() || node_b_it == id_to_node.end() ||
                phi_a_it == phi_node.end() || phi_b_it == phi_node.end())
            {
                continue;
            }

            const Node* node_a = node_a_it->second;
            const Node* node_b = node_b_it->second;
            const double phi_a = phi_a_it->second;
            const double phi_b = phi_b_it->second;
            const double phi_min = std::min(phi_a, phi_b);
            const double phi_max = std::max(phi_a, phi_b);

            if (s_norm < phi_min - phi_tol || s_norm > phi_max + phi_tol) continue;

            double alpha = 0.0;
            if (std::abs(phi_b - phi_a) > phi_tol)
            {
                alpha = clamp01((s_norm - phi_a) / (phi_b - phi_a));
            }
            else
            {
                if (std::abs(s_norm - phi_a) > phi_tol) continue;
                alpha = 0.5;
            }

            const libMesh::Point A((*node_a)(0), (*node_a)(1));
            const libMesh::Point B((*node_b)(0), (*node_b)(1));
            BoundaryIntersection bi;
            bi.X = lerp_point(A, B, alpha);
            VectorValue<double> ga(1.0, 0.0), gb(1.0, 0.0);
            const auto ga_it = grad_phi_node.find(id_a);
            const auto gb_it = grad_phi_node.find(id_b);
            if (ga_it != grad_phi_node.end()) ga = ga_it->second;
            if (gb_it != grad_phi_node.end()) gb = gb_it->second;
            bi.grad = normalize_reference_vector(lerp_vector(ga, gb, alpha), VectorValue<double>(1.0, 0.0));

            bool duplicate = false;
            for (const BoundaryIntersection& prev : intersections)
            {
                const double dx = bi.X(0) - prev.X(0);
                const double dy = bi.X(1) - prev.X(1);
                if (dx * dx + dy * dy <= dup_tol2)
                {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) intersections.push_back(bi);
        }

        PhiIsoSectionSample section;
        section.s_norm = s_norm;
        if (intersections.size() >= 2)
        {
            VectorValue<double> t_sum(0.0, 0.0);
            for (const BoundaryIntersection& bi : intersections) t_sum += bi.grad;
            VectorValue<double> t_hat = normalize_reference_vector(t_sum, VectorValue<double>(head_is_at_x_min() ? 1.0 : -1.0, 0.0));
            VectorValue<double> n_hat(-t_hat(1), t_hat(0));

            double p_min = std::numeric_limits<double>::max();
            double p_max = -std::numeric_limits<double>::max();
            libMesh::Point X_min, X_max;
            for (const BoundaryIntersection& bi : intersections)
            {
                const double p = bi.X(0) * n_hat(0) + bi.X(1) * n_hat(1);
                if (p < p_min)
                {
                    p_min = p;
                    X_min = bi.X;
                }
                if (p > p_max)
                {
                    p_max = p;
                    X_max = bi.X;
                }
            }

            const double h = 0.5 * std::max(p_max - p_min, 0.0);
            if (h > 1.0e-12)
            {
                section.valid = true;
                section.halfthickness = h;
                section.X_mid = libMesh::Point(0.5 * (X_min(0) + X_max(0)),
                                               0.5 * (X_min(1) + X_max(1)));
                section.t_hat = t_hat;
                section.n_hat = n_hat;
            }
        }
        sections[static_cast<std::size_t>(k)] = section;
    }

    int first_valid = -1;
    int last_valid = -1;
    for (int k = 0; k < n_sections; ++k)
    {
        if (sections[static_cast<std::size_t>(k)].valid)
        {
            if (first_valid < 0) first_valid = k;
            last_valid = k;
        }
    }
    if (first_valid < 0 || last_valid < 0)
    {
        TBOX_ERROR("build_phi_isocontour_section_table(): no valid phi-isocontour boundary sections.\n");
    }

    for (int k = first_valid - 1; k >= 0; --k)
    {
        sections[static_cast<std::size_t>(k)] = sections[static_cast<std::size_t>(first_valid)];
        sections[static_cast<std::size_t>(k)].s_norm =
            static_cast<double>(k) / static_cast<double>(n_sections - 1);
    }
    for (int k = last_valid + 1; k < n_sections; ++k)
    {
        sections[static_cast<std::size_t>(k)] = sections[static_cast<std::size_t>(last_valid)];
        sections[static_cast<std::size_t>(k)].s_norm =
            static_cast<double>(k) / static_cast<double>(n_sections - 1);
    }
    for (int k = first_valid; k <= last_valid; ++k)
    {
        if (sections[static_cast<std::size_t>(k)].valid) continue;
        int kl = k - 1;
        while (kl >= first_valid && !sections[static_cast<std::size_t>(kl)].valid) --kl;
        int kr = k + 1;
        while (kr <= last_valid && !sections[static_cast<std::size_t>(kr)].valid) ++kr;
        if (kl < first_valid || kr > last_valid)
        {
            TBOX_ERROR("build_phi_isocontour_section_table(): failed to fill invalid phi-section.\n");
        }
        const double s_norm_k = static_cast<double>(k) / static_cast<double>(n_sections - 1);
        const double s_norm_l = sections[static_cast<std::size_t>(kl)].s_norm;
        const double s_norm_r = sections[static_cast<std::size_t>(kr)].s_norm;
        const double a = clamp01((s_norm_k - s_norm_l) / std::max(s_norm_r - s_norm_l, 1.0e-24));
        PhiIsoSectionSample filled;
        filled.valid = true;
        filled.s_norm = s_norm_k;
        filled.halfthickness =
            (1.0 - a) * sections[static_cast<std::size_t>(kl)].halfthickness +
            a * sections[static_cast<std::size_t>(kr)].halfthickness;
        filled.X_mid = lerp_point(sections[static_cast<std::size_t>(kl)].X_mid,
                                  sections[static_cast<std::size_t>(kr)].X_mid, a);
        filled.t_hat = normalize_reference_vector(
            lerp_vector(sections[static_cast<std::size_t>(kl)].t_hat,
                        sections[static_cast<std::size_t>(kr)].t_hat, a),
            sections[static_cast<std::size_t>(kl)].t_hat);
        filled.n_hat = VectorValue<double>(-filled.t_hat(1), filled.t_hat(0));
        if ((filled.n_hat * sections[static_cast<std::size_t>(kl)].n_hat) < 0.0) filled.n_hat *= -1.0;
        sections[static_cast<std::size_t>(k)] = filled;
    }

    double midline_length = 0.0;
    for (int k = 0; k + 1 < n_sections; ++k)
    {
        const libMesh::Point& A = sections[static_cast<std::size_t>(k)].X_mid;
        const libMesh::Point& B = sections[static_cast<std::size_t>(k + 1)].X_mid;
        const double dx = B(0) - A(0);
        const double dy = B(1) - A(1);
        midline_length += std::sqrt(dx * dx + dy * dy);
    }
    midline_length = std::max(midline_length, std::max(ref_body_length, 1.0e-12));

    for (int k = 0; k < n_sections; ++k)
    {
        const double s_norm = static_cast<double>(k) / static_cast<double>(n_sections - 1);
        sections[static_cast<std::size_t>(k)].valid = true;
        sections[static_cast<std::size_t>(k)].s_norm = s_norm;
        sections[static_cast<std::size_t>(k)].s = s_norm * midline_length;
    }

    ref_phi_sections.swap(sections);
    rebuild_reference_centerline_from_phi_sections();

    // The strict literature-style phi parameterization does not smooth h(phi):
    // the section half-thickness is defined by the boundary intersection of
    // each isocontour {phi = const}.
    pout << "  Phi-isocontour section table built: n_sections = " << n_sections
         << ", L_phi_midline = " << ref_arc_length
         << ", h_max = " << ref_h_max << "\n";
}

void rebuild_reference_halfthickness_from_projection(MeshBase& mesh)
{
    const int n = static_cast<int>(ref_profile_s.size());
    if (n < 2)
    {
        TBOX_ERROR("rebuild_reference_halfthickness_from_projection(): invalid s grid.\n");
    }

    ref_halfthickness.assign(std::size_t(n), 0.0);
    ref_halfthickness_pchip_deriv.clear();
    std::vector<double> count(std::size_t(n), 0.0);

    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node& node = **node_it;
        const double x_tol = 1.0e-10 * std::max(1.0, ref_body_length);
        if (head_is_at_x_min())
        {
            if (node(0) > ref_backbone_end_x + x_tol) continue;
        }
        else
        {
            if (node(0) < ref_backbone_end_x - x_tol) continue;
        }

        const ProjectionResult proj = project_to_reference_centerline(node);
        if (!proj.valid) continue;

        // A sample constrains both ends of its s-interval. With the
        // shape-preserving interpolant below this keeps h(s) above all
        // projected mesh samples instead of only above nearest-bin maxima.
        add_halfthickness_sample_to_raw_envelope(ref_halfthickness, count, proj.s, std::abs(proj.eta));
    }

    IBTK_MPI::maxReduction(ref_halfthickness.data(), n);
    IBTK_MPI::sumReduction(count.data(), n);
    finalize_reference_halfthickness_from_counts(
        count, "rebuild_reference_halfthickness_from_projection()");
}

void build_reference_laplace_parameterization(MeshBase& mesh)
{
    ref_laplace_node_geom.clear();
    ref_laplace_parameterization_built = false;
    active_end_s_norm = std::numeric_limits<double>::quiet_NaN();

    const unsigned int dim = mesh.mesh_dimension();
    if (dim != NDIM)
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): mesh dimension mismatch.\n");
    }

    const double Lx = std::max(ref_x_max - ref_x_min, 1.0e-12);
    const bool head_at_x_min = head_is_at_x_min();
    const double head_width = std::max(0.0, laplace_head_bc_width_over_L) * Lx;
    const double tail_width = std::max(0.0, laplace_tail_bc_width_over_L) * Lx;

    std::map<dof_id_type, const Node*> id_to_node;
    std::set<dof_id_type> boundary_node_ids;
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node* node = *node_it;
        id_to_node[node->id()] = node;
    }

    BoundaryInfo& boundary_info = mesh.get_boundary_info();
    std::set<dof_id_type> head_patch_node_ids;
    std::set<dof_id_type> tail_patch_node_ids;
    double local_head_side_count = 0.0;
    double local_tail_side_count = 0.0;

    for (auto elem_it = mesh.elements_begin(); elem_it != mesh.elements_end(); ++elem_it)
    {
        Elem* elem = *elem_it;
        if (!elem || !elem->active()) continue;

        for (unsigned int side = 0; side < elem->n_sides(); ++side)
        {
            if (elem->neighbor_ptr(side) != nullptr) continue;

            std::unique_ptr<const Elem> side_elem = elem->build_side_ptr(side);
            if (!side_elem || side_elem->n_nodes() == 0) continue;

            double x_centroid = 0.0;
            for (unsigned int k = 0; k < side_elem->n_nodes(); ++k)
            {
                const dof_id_type node_id = side_elem->node_id(k);
                const Node* node = id_to_node[node_id];
                boundary_node_ids.insert(node_id);
                x_centroid += (*node)(0);
            }
            x_centroid /= static_cast<double>(side_elem->n_nodes());

            const bool in_head_patch =
                head_at_x_min ? (x_centroid <= ref_x_min + head_width + 1.0e-12) :
                                (x_centroid >= ref_x_max - head_width - 1.0e-12);
            const bool in_tail_patch =
                head_at_x_min ? (x_centroid >= ref_x_max - tail_width - 1.0e-12) :
                                (x_centroid <= ref_x_min + tail_width + 1.0e-12);

            if (in_head_patch && in_tail_patch)
            {
                TBOX_ERROR("build_reference_laplace_parameterization(): head/tail Laplace boundary "
                           << "patches overlap; reduce LAPLACE_HEAD_BC_WIDTH_OVER_L or "
                           << "LAPLACE_TAIL_BC_WIDTH_OVER_L.\n");
            }
            if (in_head_patch)
            {
                boundary_info.add_side(elem, side, LAPLACE_HEAD_BID);
                local_head_side_count += 1.0;
                for (unsigned int k = 0; k < side_elem->n_nodes(); ++k)
                    head_patch_node_ids.insert(side_elem->node_id(k));
            }
            if (in_tail_patch)
            {
                boundary_info.add_side(elem, side, LAPLACE_TAIL_BID);
                local_tail_side_count += 1.0;
                for (unsigned int k = 0; k < side_elem->n_nodes(); ++k)
                    tail_patch_node_ids.insert(side_elem->node_id(k));
            }
        }
    }
    if (boundary_node_ids.empty())
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): no boundary sides found.\n");
    }

    double patch_counts[4] =
    {
        local_head_side_count,
        local_tail_side_count,
        static_cast<double>(head_patch_node_ids.size()),
        static_cast<double>(tail_patch_node_ids.size())
    };
    if (!mesh.is_replicated()) IBTK_MPI::sumReduction(patch_counts, 4);
    const double head_side_count = patch_counts[0];
    const double tail_side_count = patch_counts[1];
    const double head_node_count = patch_counts[2];
    const double tail_node_count = patch_counts[3];
    if (head_side_count < 0.5 || tail_side_count < 0.5 ||
        head_node_count < 0.5 || tail_node_count < 0.5)
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): failed to mark finite head/tail "
                   << "boundary side patches. head sides/nodes = "
                   << head_side_count << "/" << head_node_count
                   << ", tail sides/nodes = "
                   << tail_side_count << "/" << tail_node_count << "\n");
    }

    // ------------------------------------------------------------------
    // FE Galerkin harmonic coordinate solve:
    //   ∫ grad(phi)·grad(v) dX = 0, phi=0 on head, phi=1 on tail.
    // Head/tail are finite boundary side patches. Dirichlet constraints are
    // imposed through libMesh DofMap rather than element-local elimination.
    // ------------------------------------------------------------------
    EquationSystems laplace_es(mesh);
    LinearImplicitSystem& laplace_sys =
        laplace_es.add_system<LinearImplicitSystem>("laplace_phi");
    const unsigned int phi_var = laplace_sys.add_variable("phi", FIRST, LAGRANGE);

    std::vector<unsigned int> phi_vars(1, phi_var);
    {
        std::set<boundary_id_type> head_boundary_ids;
        head_boundary_ids.insert(LAPLACE_HEAD_BID);
        laplace_sys.get_dof_map().add_dirichlet_boundary(
            DirichletBoundary(head_boundary_ids, phi_vars, ConstFunction<Number>(0.0)));
    }
    {
        std::set<boundary_id_type> tail_boundary_ids;
        tail_boundary_ids.insert(LAPLACE_TAIL_BID);
        laplace_sys.get_dof_map().add_dirichlet_boundary(
            DirichletBoundary(tail_boundary_ids, phi_vars, ConstFunction<Number>(1.0)));
    }

    laplace_es.init();
    laplace_sys.assemble_before_solve = false;

    const DofMap& dof_map = laplace_sys.get_dof_map();
    pout << "  Laplace DofMap constraints = " << dof_map.n_constrained_dofs()
         << ", system dofs = " << laplace_sys.n_dofs() << "\n";
    std::map<dof_id_type, dof_id_type> node_to_dof;
    std::vector<dof_id_type> node_dofs;
    for (auto node_it = mesh.nodes_begin(); node_it != mesh.nodes_end(); ++node_it)
    {
        const Node* node = *node_it;
        dof_map.dof_indices(node, node_dofs, phi_var);
        if (!node_dofs.empty()) node_to_dof[node->id()] = node_dofs[0];
    }

    SparseMatrix<Number>& K = *laplace_sys.matrix;
    NumericVector<Number>& rhs = *laplace_sys.rhs;
    K.zero();
    rhs.zero();

    const FEType fe_type = dof_map.variable_type(phi_var);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule = fe_type.default_quadrature_rule(dim);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();

    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    std::vector<dof_id_type> dof_indices;

    for (auto el_it = mesh.active_elements_begin();
         el_it != mesh.active_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        dof_map.dof_indices(elem, dof_indices, phi_var);
        fe->reinit(elem);
        const unsigned int n = static_cast<unsigned int>(dof_indices.size());
        Ke.resize(n, n);
        Fe.resize(n);
        Ke.zero();
        Fe.zero();
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                    Ke(i, j) += JxW[qp] * (dphi[i][qp] * dphi[j][qp]);
        }

        dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
        K.add_matrix(Ke, dof_indices);
        rhs.add_vector(Fe, dof_indices);
    }
    K.close();
    rhs.close();

    laplace_sys.solve();
    NumericVector<Number>* phi_vec = laplace_sys.solution.get();
    dof_map.enforce_constraints_exactly(laplace_sys, phi_vec);
    phi_vec->close();
    laplace_sys.update();

    std::vector<Number> phi_dof_values;
    phi_vec->localize(phi_dof_values);

    std::map<dof_id_type, double> phi_node;
    double local_raw_phi_min = std::numeric_limits<double>::max();
    double local_raw_phi_max = -std::numeric_limits<double>::max();
    double local_phi_min = std::numeric_limits<double>::max();
    double local_phi_max = -std::numeric_limits<double>::max();
    double local_head_bc_error = 0.0;
    double local_tail_bc_error = 0.0;
    for (const auto& kv : node_to_dof)
    {
        const dof_id_type dof = kv.second;
        if (static_cast<std::size_t>(dof) >= phi_dof_values.size())
        {
            TBOX_ERROR("build_reference_laplace_parameterization(): missing localized "
                       "Laplace phi value for node " << kv.first << " dof " << dof
                       << ". Refusing x-based fallback.\n");
        }
        const double val =
            static_cast<double>(phi_dof_values[static_cast<std::size_t>(dof)]);
        local_raw_phi_min = std::min(local_raw_phi_min, val);
        local_raw_phi_max = std::max(local_raw_phi_max, val);
        phi_node[kv.first] = clamp01(val);
        local_phi_min = std::min(local_phi_min, phi_node[kv.first]);
        local_phi_max = std::max(local_phi_max, phi_node[kv.first]);
        if (head_patch_node_ids.find(kv.first) != head_patch_node_ids.end())
            local_head_bc_error = std::max(local_head_bc_error, std::abs(val));
        if (tail_patch_node_ids.find(kv.first) != tail_patch_node_ids.end())
            local_tail_bc_error = std::max(local_tail_bc_error, std::abs(val - 1.0));
    }

    double global_raw_phi_min = local_raw_phi_min;
    double global_raw_phi_max = local_raw_phi_max;
    double global_phi_min = local_phi_min;
    double global_phi_max = local_phi_max;
    double global_head_bc_error = local_head_bc_error;
    double global_tail_bc_error = local_tail_bc_error;
    IBTK_MPI::minReduction(&global_raw_phi_min, 1);
    IBTK_MPI::maxReduction(&global_raw_phi_max, 1);
    IBTK_MPI::minReduction(&global_phi_min, 1);
    IBTK_MPI::maxReduction(&global_phi_max, 1);
    IBTK_MPI::maxReduction(&global_head_bc_error, 1);
    IBTK_MPI::maxReduction(&global_tail_bc_error, 1);

    const double dirichlet_tol = 1.0e-10;
    if (global_head_bc_error > dirichlet_tol ||
        global_tail_bc_error > dirichlet_tol)
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): Laplace "
                   "Dirichlet constraints were not enforced exactly after solve. "
                   "max |phi(head)-0| = " << global_head_bc_error
                   << ", max |phi(tail)-1| = " << global_tail_bc_error
                   << ".\n");
    }

    const double phi_bound_tol = 1.0e-10;
    if (global_raw_phi_min < -phi_bound_tol ||
        global_raw_phi_max > 1.0 + phi_bound_tol)
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): constrained "
                   "Laplace coordinate left the admissible [0,1] range. "
                   "range = [" << global_raw_phi_min << ", "
                   << global_raw_phi_max << "].\n");
    }

    if (!(global_phi_min <= 0.05 && global_phi_max >= 0.95 &&
          global_phi_max - global_phi_min >= 0.50))
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): degenerate harmonic coordinate "
                   << "range [" << global_phi_min << ", " << global_phi_max
                   << "], constrained range [" << global_raw_phi_min << ", "
                   << global_raw_phi_max << "].\n");
    }

    // Average FE gradients back to nodes. For first-order triangles this is a
    // piecewise-constant Galerkin gradient; for higher-order elements this is a
    // quadrature-weighted nodal average.
    std::map<dof_id_type, VectorValue<double> > grad_sum;
    std::map<dof_id_type, double> grad_weight;
    for (auto el_it = mesh.active_elements_begin();
         el_it != mesh.active_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        dof_map.dof_indices(elem, dof_indices, phi_var);
        fe->reinit(elem);
        VectorValue<double> grad_elem(0.0, 0.0);
        double elem_w = 0.0;
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            VectorValue<double> g_qp(0.0, 0.0);
            for (unsigned int k = 0; k < dof_indices.size(); ++k)
            {
                const auto phi_it = phi_node.find(elem->node_id(k));
                if (phi_it == phi_node.end())
                {
                    TBOX_ERROR("build_reference_laplace_parameterization(): missing "
                               "Laplace phi node value for element node "
                               << elem->node_id(k) << ". Refusing x-based fallback.\n");
                }
                const double phi_k = phi_it->second;
                g_qp(0) += phi_k * dphi[k][qp](0);
                g_qp(1) += phi_k * dphi[k][qp](1);
            }
            grad_elem += JxW[qp] * g_qp;
            elem_w += JxW[qp];
        }
        if (elem_w > 0.0) grad_elem /= elem_w;
        for (unsigned int k = 0; k < elem->n_nodes(); ++k)
        {
            const dof_id_type node_id = elem->node_id(k);
            grad_sum[node_id] += elem_w * grad_elem;
            grad_weight[node_id] += elem_w;
        }
    }

    std::map<dof_id_type, VectorValue<double> > grad_phi_node;
    for (const auto& kv : id_to_node)
    {
        const dof_id_type node_id = kv.first;
        VectorValue<double> g(head_at_x_min ? 1.0 : -1.0, 0.0);
        const auto gs = grad_sum.find(node_id);
        const auto gw = grad_weight.find(node_id);
        if (gs != grad_sum.end() && gw != grad_weight.end() && gw->second > 0.0)
            g = gs->second / gw->second;
        const double g_norm = std::sqrt(std::max(g * g, 0.0));
        if (g_norm > 1.0e-14) g /= g_norm;
        grad_phi_node[node_id] = g;
    }

    // Convert the reference backbone end from Euclidean x to harmonic s/phi once.
    const double x_cut = ref_backbone_end_x;
    double local_min_dx = std::numeric_limits<double>::max();
    for (const dof_id_type node_id : boundary_node_ids)
    {
        const Node* node = id_to_node[node_id];
        local_min_dx = std::min(local_min_dx, std::abs((*node)(0) - x_cut));
    }
    double global_min_dx = local_min_dx;
    IBTK_MPI::minReduction(&global_min_dx, 1);
    double local_cut_sum = 0.0;
    double local_cut_count = 0.0;
    const double cut_tol = global_min_dx + 1.0e-8 * std::max(1.0, Lx);
    for (const dof_id_type node_id : boundary_node_ids)
    {
        const Node* node = id_to_node[node_id];
        if (std::abs((*node)(0) - x_cut) <= cut_tol)
        {
            local_cut_sum += phi_node[node_id];
            local_cut_count += 1.0;
        }
    }
    double cut_data[2] = { local_cut_sum, local_cut_count };
    IBTK_MPI::sumReduction(cut_data, 2);
    if (cut_data[1] > 0.5)
    {
        active_end_s_norm = clamp01(cut_data[0] / cut_data[1]);
    }
    else
    {
        TBOX_ERROR("build_reference_laplace_parameterization(): no Laplace phi nodes "
                   "found on REFERENCE_BACKBONE_END_X = " << x_cut << ".\n");
        active_end_s_norm = std::numeric_limits<double>::quiet_NaN();
    }

    // Strict literature-style geometry: eta=0 and h(s) are not obtained from
    // the old boundary-loop centerline.  Instead, each transverse section is
    // the pair of boundary intersections of the Laplace isocontour
    // {phi=s_norm}.  Its midpoint is the eta origin and half the distance
    // between the two boundary points is h(s).
    build_phi_isocontour_section_table(mesh, id_to_node, phi_node, grad_phi_node);

    double local_s_ref_min = std::numeric_limits<double>::max();
    double local_s_ref_max = -std::numeric_limits<double>::max();
    for (const auto& kv : phi_node)
    {
        const double s_ref = kv.second * std::max(ref_arc_length, 1.0e-12);
        local_s_ref_min = std::min(local_s_ref_min, s_ref);
        local_s_ref_max = std::max(local_s_ref_max, s_ref);
    }
    double global_s_ref_min = local_s_ref_min;
    double global_s_ref_max = local_s_ref_max;
    IBTK_MPI::minReduction(&global_s_ref_min, 1);
    IBTK_MPI::maxReduction(&global_s_ref_max, 1);

    for (const auto& kv : id_to_node)
    {
        const dof_id_type node_id = kv.first;
        const Node* node = kv.second;
        const double s_norm = phi_node[node_id];
        const PhiIsoSectionSample section = interpolate_phi_section_sample(s_norm);
        VectorValue<double> t = grad_phi_node[node_id];
        if ((t * section.t_hat) < 0.0) t *= -1.0;
        t = normalize_reference_vector(t, section.t_hat);
        VectorValue<double> n_hat(-t(1), t(0));
        if ((n_hat * section.n_hat) < 0.0) n_hat *= -1.0;

        ReferenceGeometrySample geom;
        geom.s = s_norm * std::max(ref_arc_length, 1.0e-12);
        geom.t_hat = t;
        geom.eta = ((*node)(0) - section.X_mid(0)) * n_hat(0) +
                   ((*node)(1) - section.X_mid(1)) * n_hat(1);
        ref_laplace_node_geom[node_id] = geom;
    }

    ref_laplace_parameterization_built = true;
    pout << "  FE-Galerkin Laplace reference parameterization built:"
         << " head sides/nodes = " << head_side_count << "/" << head_node_count
         << ", tail sides/nodes = " << tail_side_count << "/" << tail_node_count
         << "\n"
         << "    constrained phi range = [" << global_raw_phi_min << ", " << global_raw_phi_max
         << "]\n"
         << "    max |phi(head)-0| = " << global_head_bc_error
         << ", max |phi(tail)-1| = " << global_tail_bc_error
         << "\n"
         << "    active_end_s_norm = " << active_end_s_norm
         << ", s_ref range = [" << global_s_ref_min << ", " << global_s_ref_max
         << "]"
         << ", n_nodes = " << id_to_node.size()
         << ", eta/h mode = phi_isocontour_boundary_sections\n";
}

void build_reference_profile_from_mesh(MeshBase& mesh)
{
    build_reference_backbone_from_boundary(mesh);
    if (use_laplace_reference_parameterization)
    {
        extend_reference_centerline_to_tail_tip();
    }
    build_reference_centerline_segments();
    if (use_laplace_reference_parameterization)
    {
        build_reference_laplace_parameterization(mesh);
    }
    else
    {
        rebuild_reference_halfthickness_from_projection(mesh);
    }
}

void initialize_tail_tracking_points(MeshBase& mesh)
{
    const double invalid_id = static_cast<double>(std::numeric_limits<dof_id_type>::max());
    const double x_tol = 1.0e-10 * std::max(1.0, ref_body_length);
    const double y_tol = 1.0e-10 * std::max(1.0, ref_h_max);
    const bool tail_at_x_max = (wave_head_location == "x_min");

    double local_tail_x = tail_at_x_max ? -std::numeric_limits<double>::max() :
                                          std::numeric_limits<double>::max();
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        local_tail_x = tail_at_x_max ? std::max(local_tail_x, (*node)(0)) :
                                       std::min(local_tail_x, (*node)(0));
    }
    if (tail_at_x_max)
    {
        IBTK_MPI::maxReduction(&local_tail_x, 1);
    }
    else
    {
        IBTK_MPI::minReduction(&local_tail_x, 1);
    }

    double local_upper_y = -std::numeric_limits<double>::max();
    double local_lower_y =  std::numeric_limits<double>::max();
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        if (std::abs((*node)(0) - local_tail_x) > x_tol) continue;
        local_upper_y = std::max(local_upper_y, (*node)(1));
        local_lower_y = std::min(local_lower_y, (*node)(1));
    }
    IBTK_MPI::maxReduction(&local_upper_y, 1);
    IBTK_MPI::minReduction(&local_lower_y, 1);

    double local_upper_id = invalid_id;
    double local_lower_id = invalid_id;
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        if (std::abs((*node)(0) - local_tail_x) > x_tol) continue;
        if (std::abs((*node)(1) - local_upper_y) <= y_tol)
        {
            local_upper_id = std::min(local_upper_id, static_cast<double>(node->id()));
        }
        if (std::abs((*node)(1) - local_lower_y) <= y_tol)
        {
            local_lower_id = std::min(local_lower_id, static_cast<double>(node->id()));
        }
    }
    IBTK_MPI::minReduction(&local_upper_id, 1);
    IBTK_MPI::minReduction(&local_lower_id, 1);

    ref_tail_tip_center_point =
        libMesh::Point(local_tail_x, 0.5 * (local_upper_y + local_lower_y));

    if (local_upper_id >= invalid_id || local_lower_id >= invalid_id)
    {
        TBOX_ERROR("initialize_tail_tracking_points(): failed to locate tail tracking nodes.\n");
    }

    ref_tail_upper_node_id = static_cast<dof_id_type>(local_upper_id);
    ref_tail_lower_node_id = static_cast<dof_id_type>(local_lower_id);
}

void initialize_fin_root_tracking_points(MeshBase& mesh)
{
    const double invalid_id = static_cast<double>(std::numeric_limits<dof_id_type>::max());
    const double y_tol = 1.0e-10 * std::max(1.0, ref_h_max);
    const double fin_root_x = ref_backbone_end_x;

    // Step 1: find global minimum |x - REFERENCE_BACKBONE_END_X| among boundary nodes.
    if (!std::isfinite(fin_root_x))
    {
        pout << "WARNING: initialize_fin_root_tracking_points(): "
             << "ref_backbone_end_x is not finite; fin pitch diagnostic disabled.\n";
        s_fin_pitch_diag_enable = false;
        return;
    }
    double local_best_dist = std::numeric_limits<double>::max();
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        local_best_dist = std::min(local_best_dist,
                                   std::abs((*node)(0) - fin_root_x));
    }
    IBTK_MPI::minReduction(&local_best_dist, 1);
    const double snap_tol = 1.5 * local_best_dist +
                            1.0e-10 * std::max(1.0, ref_body_length);

    // Step 2: find upper/lower y among snapped boundary nodes.
    double local_upper_y = -std::numeric_limits<double>::max();
    double local_lower_y =  std::numeric_limits<double>::max();
    double local_snap_x_sum = 0.0;
    double local_snap_x_cnt = 0.0;
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        if (std::abs((*node)(0) - fin_root_x) > snap_tol) continue;
        local_upper_y = std::max(local_upper_y, (*node)(1));
        local_lower_y = std::min(local_lower_y, (*node)(1));
        local_snap_x_sum += (*node)(0);
        local_snap_x_cnt += 1.0;
    }
    IBTK_MPI::maxReduction(&local_upper_y, 1);
    IBTK_MPI::minReduction(&local_lower_y, 1);
    IBTK_MPI::sumReduction(&local_snap_x_sum, 1);
    IBTK_MPI::sumReduction(&local_snap_x_cnt, 1);
    const double snap_x = (local_snap_x_cnt > 0.5)
        ? local_snap_x_sum / local_snap_x_cnt : fin_root_x;

    // Step 3: find node IDs for the topmost and bottommost snapped nodes.
    double local_upper_id = invalid_id;
    double local_lower_id = invalid_id;
    for (auto it = mesh.bnd_nodes_begin(); it != mesh.bnd_nodes_end(); ++it)
    {
        const Node* node = *it;
        if (!node) continue;
        if (std::abs((*node)(0) - fin_root_x) > snap_tol) continue;
        if (std::abs((*node)(1) - local_upper_y) <= y_tol)
            local_upper_id = std::min(local_upper_id,
                                      static_cast<double>(node->id()));
        if (std::abs((*node)(1) - local_lower_y) <= y_tol)
            local_lower_id = std::min(local_lower_id,
                                      static_cast<double>(node->id()));
    }
    IBTK_MPI::minReduction(&local_upper_id, 1);
    IBTK_MPI::minReduction(&local_lower_id, 1);

    if (local_upper_id >= invalid_id || local_lower_id >= invalid_id)
    {
        pout << "WARNING: initialize_fin_root_tracking_points(): "
             << "failed to locate fin root nodes at x=" << fin_root_x
             << "; fin pitch diagnostic disabled.\n";
        s_fin_pitch_diag_enable = false;
        return;
    }

    ref_fin_root_center_ref = libMesh::Point(snap_x,
                                              0.5 * (local_upper_y + local_lower_y));
    ref_fin_root_upper_node_id = static_cast<dof_id_type>(local_upper_id);
    ref_fin_root_lower_node_id = static_cast<dof_id_type>(local_lower_id);

    pout << "  Fin root tracking: snap_x=" << snap_x
         << " (target=" << fin_root_x << ")"
         << ", upper_node=" << ref_fin_root_upper_node_id
         << ", lower_node=" << ref_fin_root_lower_node_id
         << ", y_upper=" << local_upper_y
         << ", y_lower=" << local_lower_y << "\n";
}

bool get_current_node_position(const DofMap& X_dof_map,
                               NumericVector<double>& X_ghost_vec,
                               MeshBase& mesh,
                               const dof_id_type node_id,
                               double& x_out,
                               double& y_out)
{
    double local_vals[3] = { 0.0, 0.0, 0.0 };
    std::vector<unsigned int> dof_idx_x, dof_idx_y;

    for (const Node* node : mesh.local_node_ptr_range())
    {
        if (!node || node->id() != node_id) continue;
        X_dof_map.dof_indices(node, dof_idx_x, 0);
        X_dof_map.dof_indices(node, dof_idx_y, 1);
        local_vals[0] = X_ghost_vec(dof_idx_x[0]);
        local_vals[1] = X_ghost_vec(dof_idx_y[0]);
        local_vals[2] = 1.0;
        break;
    }

    IBTK_MPI::sumReduction(local_vals, 3);
    if (local_vals[2] < 0.5) return false;

    x_out = local_vals[0];
    y_out = local_vals[1];
    return true;
}

double update_tail_cycle_amplitude(double loop_time, double tail_tip_y_rel)
{
    static int s_cycle_id = -1;
    static double s_cycle_min = std::numeric_limits<double>::quiet_NaN();
    static double s_cycle_max = std::numeric_limits<double>::quiet_NaN();
    static double s_last_cycle_A_norm = std::numeric_limits<double>::quiet_NaN();

    if (!(wave_frequency > 0.0)) return std::numeric_limits<double>::quiet_NaN();

    const double period = 1.0 / wave_frequency;
    const double phase_time = loop_time - wave_ramp_time;
    if (phase_time < 0.0) return std::numeric_limits<double>::quiet_NaN();

    const int cycle_id = static_cast<int>(std::floor(phase_time / period));
    if (s_cycle_id < 0)
    {
        s_cycle_id = cycle_id;
        s_cycle_min = tail_tip_y_rel;
        s_cycle_max = tail_tip_y_rel;
        return s_last_cycle_A_norm;
    }

    if (cycle_id != s_cycle_id)
    {
        if (std::isfinite(s_cycle_min) && std::isfinite(s_cycle_max))
        {
            s_last_cycle_A_norm =
                0.5 * (s_cycle_max - s_cycle_min) / std::max(fish_length, 1.0e-12);
        }
        s_cycle_id = cycle_id;
        s_cycle_min = tail_tip_y_rel;
        s_cycle_max = tail_tip_y_rel;
    }
    else
    {
        s_cycle_min = std::min(s_cycle_min, tail_tip_y_rel);
        s_cycle_max = std::max(s_cycle_max, tail_tip_y_rel);
    }

    return s_last_cycle_A_norm;
}

inline double body_halfthick_from_s(double s)
{
    if (ref_profile_s.size() < 2 || ref_halfthickness.size() != ref_profile_s.size()) return 0.0;
    if (s <= ref_profile_s.front()) return std::max(0.0, ref_halfthickness.front());
    if (s >= ref_profile_s.back()) return std::max(0.0, ref_halfthickness.back());
    return std::max(0.0, pchip_interpolate_halfthickness(s));
}

inline double reference_x_norm_from_s(double s)
{
    if (ref_profile_s.size() < 2 || ref_profile_x.size() != ref_profile_s.size())
    {
        return clamp01(s / std::max(ref_arc_length, 1.0e-12));
    }

    const double s_clamped = std::max(ref_profile_s.front(), std::min(s, ref_profile_s.back()));
    const double x = linear_interp_1d(ref_profile_s, ref_profile_x, s_clamped);
    const double Lx = std::max(fish_length, 1.0e-12);

    if (head_is_at_x_min())
    {
        return clamp01((x - x_leading) / Lx);
    }
    else
    {
        return clamp01((x_leading - x) / Lx);
    }
}

// Active wave uses the active-body coordinate as the phase length scale:
//   xi = (s_norm - active_s_start) / (active_s_end_effective - active_s_start),
//   phase = 2π xi / lambda_act.
// This keeps the muscle wave and bell envelope in the same active-body
// coordinate instead of phasing the active body by the passive caudal fin.
inline double active_phase_length_dimensional()
{
    return std::max(active_s_span_norm_effective() * ref_arc_length, 1.0e-12);
}

inline double active_phase_wavelength_dimensional()
{
    return active_wavelength_over_L * active_phase_length_dimensional();
}

inline double active_phase_slope_abs_over_s_norm()
{
    const double span = active_s_span_norm_effective();
    if (span <= 1.0e-12) return std::numeric_limits<double>::quiet_NaN();
    return 2.0 * M_PI /
           (std::max(active_wavelength_over_L, 1.0e-12) * span);
}

inline double active_phase_angle_from_s(const double s, const double time)
{
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double xi = active_xi_from_s_norm(s_norm);
    return 2.0 * M_PI * xi / std::max(active_wavelength_over_L, 1.0e-12) -
           wave_time_sign * wave_omega * time + active_phase0;
}

inline std::string normalize_mode_string(const std::string& mode_raw)
{
    std::string mode;
    mode.reserve(mode_raw.size());
    for (const char c : mode_raw)
    {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc)) continue;
        mode.push_back(c == '_' ? '-' : static_cast<char>(std::toupper(uc)));
    }
    return mode;
}

inline ActiveKShapeMode parse_active_k_shape_mode(const std::string& mode_raw)
{
    const std::string mode = normalize_mode_string(mode_raw);
    if (mode == "HALF-BELL") return ActiveKShapeMode::HALF_BELL;
    if (mode == "BELL") return ActiveKShapeMode::BELL;

    TBOX_ERROR("Unknown K_SHAPE_MODE = \"" << mode_raw
               << "\". Expected \"HALF-BELL\" or \"BELL\".\n");
    return ActiveKShapeMode::HALF_BELL;
}

inline ActiveMomentMode parse_active_moment_mode(const std::string& mode_raw)
{
    const std::string mode = normalize_mode_string(mode_raw);
    if (mode == "TRAVELING") return ActiveMomentMode::TRAVELING;
    if (mode == "STATIC") return ActiveMomentMode::STATIC;

    TBOX_ERROR("Unknown ACTIVE_MOMENT_MODE = \"" << mode_raw
               << "\". Expected \"TRAVELING\" or \"STATIC\".\n");
    return ActiveMomentMode::TRAVELING;
}

inline ActiveEnvelopeMode
parse_active_envelope_mode(const std::string& mode_raw)
{
    const std::string mode = normalize_mode_string(mode_raw);
    if (mode == "K-SHAPE" || mode == "KSHAPE" || mode == "LEGACY")
        return ActiveEnvelopeMode::K_SHAPE;
    if (mode == "POSTERIOR-RAMP" || mode == "POSTERIORRAMP" ||
        mode == "POSTERIOR")
        return ActiveEnvelopeMode::POSTERIOR_RAMP;
    if (mode == "POSTERIOR-B-COMPENSATED" ||
        mode == "POSTERIORBCOMPENSATED" ||
        mode == "B-COMPENSATED" ||
        mode == "BCOMPENSATED")
        return ActiveEnvelopeMode::POSTERIOR_B_COMPENSATED;

    TBOX_ERROR("Unknown ACTIVE_ENVELOPE_MODE = \"" << mode_raw
               << "\". Expected \"K_SHAPE\", \"POSTERIOR_RAMP\", or "
                  "\"POSTERIOR_B_COMPENSATED\".\n");
    return ActiveEnvelopeMode::K_SHAPE;
}

inline const char* active_moment_mode_name()
{
    switch (active_moment_mode)
    {
    case ActiveMomentMode::TRAVELING:
        return "TRAVELING";
    case ActiveMomentMode::STATIC:
        return "STATIC";
    }
    return "TRAVELING";
}

inline const char* active_envelope_mode_name()
{
    switch (active_envelope_mode)
    {
    case ActiveEnvelopeMode::K_SHAPE:
        return "K_SHAPE";
    case ActiveEnvelopeMode::POSTERIOR_RAMP:
        return "POSTERIOR_RAMP";
    case ActiveEnvelopeMode::POSTERIOR_B_COMPENSATED:
        return "POSTERIOR_B_COMPENSATED";
    }
    return "K_SHAPE";
}

inline const char* active_k_shape_mode_name()
{
    switch (active_k_shape_mode)
    {
    case ActiveKShapeMode::HALF_BELL:
        return "HALF-BELL";
    case ActiveKShapeMode::BELL:
        return "BELL";
    }
    return "HALF-BELL";
}

inline const char* active_k_shape_formula_string()
{
    switch (active_k_shape_mode)
    {
    case ActiveKShapeMode::HALF_BELL:
        return "0.5*(1-cos(pi*xi))";
    case ActiveKShapeMode::BELL:
        return "0.5*(1-cos(2*pi*xi))";
    }
    return "0.5*(1-cos(pi*xi))";
}

inline double muscle_moment_shape_from_xi(const double xi_in)
{
    const double xi = clamp01(xi_in);
    switch (active_k_shape_mode)
    {
    case ActiveKShapeMode::HALF_BELL:
        // Posterior-rising half-cosine: 0 at head, 1 at active-body end.
        return 0.5 * (1.0 - std::cos(M_PI * xi));
    case ActiveKShapeMode::BELL:
        // Normalized to peak=1: 0.5*[1 - cos(2*pi*xi)].
        return 0.5 * (1.0 - std::cos(2.0 * M_PI * xi));
    }
    return 0.5 * (1.0 - std::cos(M_PI * xi));
}

inline double longitudinal_active_envelope(double s_norm);
inline double active_mapping_q_abs_max_from_s(double s);

inline double posterior_ramp_shape_from_xi(const double xi_in)
{
    const double xi = clamp01(xi_in);
    const double g0 = clamp01(posterior_g0);
    const double p = std::max(posterior_power, 1.0e-12);
    return g0 + (1.0 - g0) * std::pow(xi, p);
}

inline double active_on_taper_from_s_norm(const double s_norm)
{
    const double s0 = active_s_start_norm_effective();
    const double s1 = active_s_end_norm_effective();
    if (s_norm <= s0 || s_norm >= s1) return 0.0;
    const double w = std::min(std::max(active_s_smooth, 0.0),
                              std::max(s1 - s0, 0.0));
    if (w <= 1.0e-12 || s_norm >= s0 + w) return 1.0;
    return smoothstep((s_norm - s0) / w);
}

inline double active_tail_taper_from_s_norm(const double s_norm)
{
    const double s0 = active_s_start_norm_effective();
    const double s1 = active_s_end_norm_effective();
    if (s_norm <= s0 || s_norm >= s1) return 0.0;
    const double w = std::min(std::max(tail_taper_width, 0.0),
                              std::max(s1 - s0, 0.0));
    if (w <= 1.0e-12 || s_norm <= s1 - w) return 1.0;
    return smoothstep((s1 - s_norm) / w);
}

inline double active_curvature_envelope_raw_from_s_norm(
    const double s_norm_in)
{
    const double s_norm = clamp01(s_norm_in);
    const double s0 = active_s_start_norm_effective();
    const double s1 = active_s_end_norm_effective();
    if (s1 <= s0 + 1.0e-12 || s_norm <= s0 || s_norm >= s1)
        return 0.0;

    const double xi = active_xi_from_s_norm(s_norm);
    if (active_envelope_mode == ActiveEnvelopeMode::POSTERIOR_RAMP ||
        active_envelope_mode ==
            ActiveEnvelopeMode::POSTERIOR_B_COMPENSATED)
    {
        const double G_posterior =
            active_on_taper_from_s_norm(s_norm) *
            posterior_ramp_shape_from_xi(xi) *
            active_tail_taper_from_s_norm(s_norm);
        if (active_envelope_mode ==
            ActiveEnvelopeMode::POSTERIOR_B_COMPENSATED)
        {
            const double B_local =
                std::max(get_target_bending_B_local(s_norm), 1.0e-16);
            const double B_ref =
                std::max(target_bending_B_caudal, 1.0e-16);
            return G_posterior *
                   std::pow(B_ref / B_local, b_compensation_alpha);
        }
        return G_posterior;
    }

    return longitudinal_active_envelope(s_norm) *
           muscle_moment_shape_from_xi(xi);
}

inline double active_curvature_envelope_cap_from_s_norm(
    const double s_norm)
{
    if (active_envelope_cap_safe_over_E <= 0.0 ||
        active_kappa_amp <= 0.0)
        return std::numeric_limits<double>::infinity();

    const double s = clamp01(s_norm) * std::max(ref_arc_length, 1.0e-12);
    const double h = body_halfthick_from_s(s);
    const double I2 = std::max((2.0 / 3.0) * h * h * h, 1.0e-18);
    const double chi = I2 * active_mapping_q_abs_max_from_s(s);
    if (chi <= 1.0e-30 || !std::isfinite(chi))
        return std::numeric_limits<double>::infinity();

    const double G_cap =
        active_envelope_cap_safe_over_E / (chi * active_kappa_amp);
    return std::isfinite(G_cap) ?
        std::max(0.0, G_cap) :
        std::numeric_limits<double>::infinity();
}

inline double active_moment_envelope_from_s_norm(const double s_norm)
{
    const double raw = active_curvature_envelope_raw_from_s_norm(s_norm);
    if (raw <= 0.0) return 0.0;
    const double cap = active_curvature_envelope_cap_from_s_norm(s_norm);
    return std::isfinite(cap) ? std::min(raw, cap) : raw;
}

inline double muscle_moment_drive_from_s(const double s, const double time)
{
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    return active_moment_envelope_from_s_norm(s_norm) *
           std::cos(active_phase_angle_from_s(s, time));
}

inline double muscle_moment_drive_amplitude_from_s(const double s)
{
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    return active_moment_envelope_from_s_norm(s_norm);
}

inline double active_moment_prefactor_from_sample(const double s_norm,
                                                  const double h,
                                                  const double time)
{
    (void)h;
    const double ramp = wave_ramp(time);
    if (ramp <= 0.0) return 0.0;
    return ramp * get_target_bending_B_local(s_norm) *
           active_kappa_amp;
}

inline double active_moment_prefactor_unit_ramp_from_sample(const double s_norm,
                                                            const double h)
{
    (void)h;
    return get_target_bending_B_local(s_norm) * active_kappa_amp;
}

inline double active_traveling_amplitude()
{
    return active_kappa_amp;
}

inline double active_moment_value_from_sample(const double s,
                                              const double s_norm,
                                              const double h,
                                              const double time)
{
    if (active_moment_mode == ActiveMomentMode::STATIC)
    {
        const double s0 = active_s_start_norm_effective();
        const double s1 = active_s_end_norm_effective();
        if (s_norm <= s0 || s_norm >= s1) return 0.0;
        return wave_ramp(time) * static_moment_m0;
    }

    const double prefactor = active_moment_prefactor_from_sample(s_norm, h, time);
    if (prefactor <= 0.0) return 0.0;

    const double drive = muscle_moment_drive_from_s(s, time);
    return prefactor * drive;
}

inline double active_curvature_moment_from_active_moment(const double Mm)
{
    // Small-strain bending power uses epsilon_ss = -eta*kappa, hence the
    // curvature-conjugate moment is -int(T_act*eta)dA.
    return -active_moment_to_stress_sign * Mm;
}

void
coordinate_mapping_function(libMesh::Point& X, const libMesh::Point& s, void*)
{
    X = s;
    if (std::abs(initial_bend_amplitude) > 0.0)
    {
        const double s_norm = reference_x_norm_from_point(s);
        X(1) += initial_bend_amplitude * std::sin(0.5 * M_PI * s_norm);
    }
}

inline const char* active_phase_time_sign_string()
{
    return wave_time_sign >= 0.0 ? "-" : "+";
}

inline const char* active_phase_propagation_s_string()
{
    if (wave_time_sign > 0.0) return "toward increasing phase coordinate (head-to-tail)";
    if (wave_time_sign < 0.0) return "toward decreasing phase coordinate (tail-to-head)";
    return "standing wave in phase coordinate";
}

inline double active_phase_propagation_x_sign()
{
    if (wave_time_sign == 0.0) return 0.0;
    const double ds_dt_sign = (wave_time_sign > 0.0) ? 1.0 : -1.0;
    const double dx_ds_sign = head_is_at_x_min() ? 1.0 : -1.0;
    return ds_dt_sign * dx_ds_sign;
}

inline double longitudinal_active_envelope(double s_norm)
{
    s_norm = clamp01(s_norm);
    const double s0 = active_s_start_norm_effective();
    const double s1 = active_s_end_norm_effective();
    if (s1 <= s0 + 1.0e-12) return 0.0;
    if (s_norm <= s0) return 0.0;
    if (s_norm >= s1) return 0.0;

    const double ds = std::max(std::min(active_s_smooth, 0.5 * (s1 - s0)), 1.0e-12);
    double w = 1.0;
    if (s_norm < s0 + ds)
    {
        w *= smoothstep((s_norm - s0) / ds);
    }
    if (s_norm > s1 - ds)
    {
        w *= smoothstep((s1 - s_norm) / ds);
    }
    return w;
}

// Lateral cosine band: weight is zero on the inner (1-f) fraction of half-thickness
// and rises as a cosine from the inner edge to the outer boundary (q=1).
inline double active_band_weight_unit(const double q_abs)
{
    const double q = clamp01(q_abs);
    const double f = clamp01(active_band_fraction);
    if (f <= 1.0e-12) return 0.0;
    const double inner = std::max(0.0, 1.0 - f);
    if (q <= inner) return 0.0;
    const double r = clamp01((q - inner) / f);
    return 0.5 * (1.0 - std::cos(M_PI * r));
}

inline double active_band_weight(const double eta, const double h)
{
    if (h <= 1.0e-12) return 0.0;
    const double q = std::abs(eta) / h;
    if (q > 1.0 + 1.0e-10) return 0.0;
    return active_band_weight_unit(q);
}

inline double active_band_second_moment_unit()
{
    static double cached_fraction = std::numeric_limits<double>::quiet_NaN();
    static double cached_value = std::numeric_limits<double>::quiet_NaN();

    if (std::abs(cached_fraction - active_band_fraction) <= 1.0e-14 &&
        std::isfinite(cached_value))
    {
        return cached_value;
    }

    const int n = 256;
    const double dq = 2.0 / static_cast<double>(n);
    double integral = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double q = -1.0 + (static_cast<double>(i) + 0.5) * dq;
        const double w = active_band_weight_unit(std::abs(q));
        integral += w * q * q * dq;
    }

    cached_fraction = active_band_fraction;
    cached_value = integral;
    return cached_value;
}

inline double active_band_second_moment(const double h)
{
    return active_band_second_moment_unit() *
           std::pow(std::max(h, 1.0e-12), active_i2_h_power);
}

inline double active_band_second_moment_geometric(const double h)
{
    return active_band_second_moment_unit() * h * h * h;
}

// Linear interpolation from the FE-precomputed eta_bar and I2_c tables.
// Returns 0 / fallback when the tables have not been built yet.
inline double fe_eta_bar_from_s_norm(const double s_norm)
{
    if (!s_fe_section_data_built || s_fe_eta_bar.empty()) return 0.0;
    const int nb = static_cast<int>(s_fe_eta_bar.size());
    const double t = clamp01(s_norm) * (nb - 1);
    const int i0 = std::min(static_cast<int>(t), nb - 2);
    const double alpha = t - static_cast<double>(i0);
    return (1.0 - alpha) * s_fe_eta_bar[i0] + alpha * s_fe_eta_bar[i0 + 1];
}

inline double fe_I2c_from_s_norm(const double s_norm)
{
    if (!s_fe_section_data_built || s_fe_I2_c.empty()) return 0.0;
    const int nb = static_cast<int>(s_fe_I2_c.size());
    const double t = clamp01(s_norm) * (nb - 1);
    const int i0 = std::min(static_cast<int>(t), nb - 2);
    const double alpha = t - static_cast<double>(i0);
    return (1.0 - alpha) * s_fe_I2_c[i0] + alpha * s_fe_I2_c[i0 + 1];
}

inline double fe_I2c_scaled_from_s_norm(const double s_norm,
                                        const double h,
                                        const double I2_eff)
{
    const double I2_c_raw = fe_I2c_from_s_norm(s_norm);
    if (I2_c_raw <= 1.0e-30) return 0.0;

    const double I2_geom = active_band_second_moment_geometric(h);
    if (I2_geom <= 1.0e-30 || I2_eff <= 1.0e-30) return I2_c_raw;
    return I2_c_raw * I2_eff / I2_geom;
}

inline bool active_section_correction_enabled()
{
    return use_fe_active_section_data && s_fe_section_data_built;
}

inline double active_I2_use(const double s_norm,
                            const double I2_eff,
                            const double I2_c)
{
    if (!active_section_correction_enabled() || I2_c <= 1.0e-30)
    {
        return I2_eff;
    }
    const double I2_floor =
        std::max(0.0, fe_section_i2_floor_ratio) * std::max(I2_eff, 0.0);
    return std::max(I2_c, I2_floor);
}

inline int fe_normalized_section_bin_from_s(const double s)
{
    const int n_bins = std::max(fe_normalized_stress_bins, 1);
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    return std::min(static_cast<int>(s_norm * n_bins), n_bins - 1);
}

inline double fe_normalized_section_shape(const double eta, const double h)
{
    const double magnitude = active_band_weight(eta, h);
    if (magnitude <= 0.0) return 0.0;
    return eta >= 0.0 ? magnitude : -magnitude;
}

inline bool interpolate_fe_normalized_section(
    const double s,
    FENormalizedSection& normalization)
{
    if (fe_normalized_sections.empty()) return false;
    const int n_bins = static_cast<int>(fe_normalized_sections.size());
    const double ds = std::max(ref_arc_length, 1.0e-12) / n_bins;
    const double bin_coordinate =
        std::max(0.0, std::min(s, ref_arc_length)) / ds - 0.5;
    int bin0 = static_cast<int>(std::floor(bin_coordinate));
    double alpha = bin_coordinate - bin0;
    if (bin0 < 0)
    {
        bin0 = 0;
        alpha = 0.0;
    }
    if (bin0 >= n_bins - 1)
    {
        bin0 = n_bins - 1;
        alpha = 0.0;
    }
    const int bin1 = std::min(bin0 + 1, n_bins - 1);
    const FENormalizedSection& n0 =
        fe_normalized_sections[static_cast<std::size_t>(bin0)];
    const FENormalizedSection& n1 =
        fe_normalized_sections[static_cast<std::size_t>(bin1)];
    if (!n0.valid && !n1.valid) return false;
    if (!n0.valid)
    {
        normalization = n1;
        return true;
    }
    if (!n1.valid)
    {
        normalization = n0;
        return true;
    }

    const auto lerp = [alpha](const double a, const double b)
    {
        return (1.0 - alpha) * a + alpha * b;
    };
    normalization = n0;
    normalization.s_mid = std::max(0.0, std::min(s, ref_arc_length));
    normalization.g_mean = lerp(n0.g_mean, n1.g_mean);
    normalization.q_scale = lerp(n0.q_scale, n1.q_scale);
    normalization.q_abs_max =
        std::abs(normalization.q_scale) *
        std::max(std::abs(-1.0 - normalization.g_mean),
                 std::abs(1.0 - normalization.g_mean));
    normalization.valid = true;
    return true;
}

inline double fe_normalized_section_q(
    const ReferenceGeometrySample& ref_geom)
{
    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double h = body_halfthick_from_s(s);

    FENormalizedSection normalization;
    if (interpolate_fe_normalized_section(s, normalization))
    {
        const double g = fe_normalized_section_shape(ref_geom.eta, h);
        return normalization.q_scale * (g - normalization.g_mean);
    }

    const double I2 = std::max(active_band_second_moment_geometric(h), 1.0e-18);
    return -active_band_weight(ref_geom.eta, h) * ref_geom.eta / I2;
}

inline double fe_normalized_section_q_abs_max(const double s_in)
{
    const double s = std::max(0.0, std::min(s_in, ref_arc_length));
    FENormalizedSection normalization;
    if (interpolate_fe_normalized_section(s, normalization))
        return normalization.q_abs_max;

    const double h = body_halfthick_from_s(s);
    const double I2 = std::max(active_band_second_moment_geometric(h), 1.0e-18);
    return h / I2;
}

inline double legacy_active_section_q(
    const ReferenceGeometrySample& ref_geom)
{
    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double h = body_halfthick_from_s(s);
    const double I2_eff = active_band_second_moment(h);
    const double I2_c =
        fe_I2c_scaled_from_s_norm(s_norm, h, I2_eff);
    const bool use_fe_section =
        active_section_correction_enabled() && I2_c > 1.0e-30;
    const double eta_bar =
        use_fe_section ? fe_eta_bar_from_s_norm(s_norm) : 0.0;
    const double I2_use = active_I2_use(s_norm, I2_eff, I2_c);
    if (I2_use <= 1.0e-30) return 0.0;
    return active_band_weight(ref_geom.eta, h) *
           (ref_geom.eta - eta_bar) / I2_use;
}

inline double legacy_active_section_q_abs_max(const double s_in)
{
    const double s = std::max(0.0, std::min(s_in, ref_arc_length));
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double h = body_halfthick_from_s(s);
    const double I2_eff = active_band_second_moment(h);
    const double I2_c =
        fe_I2c_scaled_from_s_norm(s_norm, h, I2_eff);
    const bool use_fe_section =
        active_section_correction_enabled() && I2_c > 1.0e-30;
    const double eta_bar =
        use_fe_section ? fe_eta_bar_from_s_norm(s_norm) : 0.0;
    const double I2_use = active_I2_use(s_norm, I2_eff, I2_c);
    if (I2_use <= 1.0e-30) return 0.0;
    return std::max(std::abs(h - eta_bar),
                    std::abs(-h - eta_bar)) / I2_use;
}

inline double active_mapping_q_abs_max_from_s(const double s)
{
    return fe_normalized_section_q_abs_max(s);
}

struct ActiveMomentCapData
{
    double q_abs_max = 0.0;
    double stress_cap = 0.0;
    double moment_cap = 0.0;
    double moment_applied = 0.0;
    double applied_over_command = 1.0;
};

inline ActiveMomentCapData active_moment_cap_data(
    const double moment,
    const double s,
    const double c1_local)
{
    ActiveMomentCapData data;
    data.q_abs_max = active_mapping_q_abs_max_from_s(s);
    data.stress_cap =
        active_t_act_max_over_c1 * std::max(c1_local, 0.0);
    if (data.q_abs_max <= 1.0e-30 || data.stress_cap <= 0.0)
    {
        data.moment_cap = 0.0;
        data.moment_applied = 0.0;
        data.applied_over_command = 0.0;
        return data;
    }

    data.moment_cap = data.stress_cap / data.q_abs_max;
    data.moment_applied =
        data.moment_cap *
        std::tanh(moment / std::max(data.moment_cap, 1.0e-30));
    data.applied_over_command =
        std::abs(moment) > 1.0e-30 ?
        data.moment_applied / moment : 1.0;
    return data;
}

inline double active_applied_moment_from_sample(const double moment,
                                                const double s)
{
    return active_moment_cap_data(
        moment, s,
        get_c1_s_passive_local(clamp01(s / std::max(ref_arc_length, 1.0e-12)))).moment_applied;
}

// =========================================================================
// Active section consistency diagnostic (Xu, Zhou & Yu 2024)
// =========================================================================
// Writes the full reference centerline to reference_centerline.csv.
//
// Columns: source, s, s_norm, x, y, h, t_hat_x, t_hat_y
//   source = "laplace"  → Laplace φ-isocontour sections (active body region)
//   source = "boundary" → backup boundary-chain midpoint nodes
//                         (head stub + caudal fin beyond REFERENCE_BACKBONE_END_X)
//
// Run this at startup to verify that:
//   1. The Laplace section midpoints form a smooth centerline.
//   2. The total arc length and the truncation point match expectations.
//   3. The boundary-chain extension (caudal fin) is continuous with the Laplace end.
// =========================================================================
static void
write_reference_centerline_csv()
{
    if (IBTK_MPI::getRank() != 0) return;

    std::ofstream out("reference_centerline.csv");
    if (!out.is_open())
    {
        TBOX_WARNING("write_reference_centerline_csv(): cannot open reference_centerline.csv\n");
        return;
    }

    out << "source,s,s_norm,x,y,h,t_hat_x,t_hat_y\n";
    out.setf(std::ios::scientific);
    out.precision(8);

    // ── Laplace φ-isocontour sections (active/body region) ─────────────────
    const double L_ref = std::max(ref_arc_length, 1.0e-12);
    for (const PhiIsoSectionSample& sec : ref_phi_sections)
    {
        if (!sec.valid) continue;
        out << "laplace"
            << "," << sec.s
            << "," << sec.s_norm
            << "," << sec.X_mid(0)
            << "," << sec.X_mid(1)
            << "," << sec.halfthickness
            << "," << sec.t_hat(0)
            << "," << sec.t_hat(1)
            << "\n";
    }

    // ── Boundary-chain backup nodes (full body including caudal fin) ────────
    for (std::size_t k = 0; k < ref_centerline_nodes.size(); ++k)
    {
        const double s_k = (ref_profile_s.size() > k) ? ref_profile_s[k] : 0.0;
        const double h_k = (ref_halfthickness.size() > k) ? ref_halfthickness[k] : 0.0;
        const double s_norm_k = s_k / L_ref;
        const libMesh::Point& Xk = ref_centerline_nodes[k];

        double tx = 1.0, ty = 0.0;
        if (k + 1 < ref_centerline_nodes.size())
        {
            const libMesh::Point& Xn = ref_centerline_nodes[k + 1];
            const double dx = Xn(0) - Xk(0), dy = Xn(1) - Xk(1);
            const double len = std::sqrt(dx * dx + dy * dy);
            if (len > 1.0e-14) { tx = dx / len; ty = dy / len; }
        }

        out << "boundary"
            << "," << s_k
            << "," << s_norm_k
            << "," << Xk(0)
            << "," << Xk(1)
            << "," << h_k
            << "," << tx
            << "," << ty
            << "\n";
    }

    pout << "  reference_centerline.csv written ("
         << ref_phi_sections.size() << " Laplace sections, "
         << ref_centerline_nodes.size() << " boundary-chain nodes, "
         << "arc_length=" << ref_arc_length << ")\n";
}

// Verifies: ∫ T_act dη ≈ 0 (zero net axial force) and the curvature-conjugate
// moment -∫T_act η dη reports the actual curvature-conjugate sign for unit Mm.
// =========================================================================
static void
write_active_section_consistency()
{
    if (IBTK_MPI::getRank() != 0) return;

    std::ofstream out("active_section_consistency.csv");
    if (!out.is_open())
    {
        TBOX_WARNING("write_active_section_consistency(): cannot open active_section_consistency.csv\n");
        return;
    }
    out << "s_norm,h,I2_eff,N_unit,stress_moment_unit,curvature_moment_unit"
        << ",expected_curvature_moment_unit,curvature_moment_error"
        << ",active_moment_to_stress_sign\n";
    out.setf(std::ios::scientific);
    out.precision(8);

    const int ns = 101;
    const int nq = 400;
    double max_abs_N = 0.0;
    double max_abs_M_error = 0.0;

    for (int i = 0; i < ns; ++i)
    {
        const double s_norm = static_cast<double>(i) / static_cast<double>(ns - 1);
        const double s = s_norm * std::max(ref_arc_length, 1.0e-12);
        const double h = body_halfthick_from_s(s);
        if (h < 1.0e-12) continue;
        const double I2 = active_band_second_moment(h);
        if (I2 <= 1.0e-30) continue;
        const double I2_geom = active_band_second_moment_geometric(h);

        double N = 0.0, M = 0.0;
        const double deta = 2.0 * h / static_cast<double>(nq);
        for (int q = 0; q < nq; ++q)
        {
            const double eta = -h + (static_cast<double>(q) + 0.5) * deta;
            const double w = active_band_weight(eta, h);
            // Unit raw Mm=1. Since epsilon_ss = -eta*kappa, the
            // curvature-conjugate moment is -int(T*eta)deta.
            const double T = active_moment_to_stress_sign * w * eta / I2;
            N += T * deta;
            M += T * eta * deta;
        }

        const double M_curv = -M;
        const double expected_M_curv =
            (I2 > 1.0e-30) ? -active_moment_to_stress_sign * I2_geom / I2 : 0.0;
        const double M_error = M_curv - expected_M_curv;
        max_abs_N = std::max(max_abs_N, std::abs(N));
        max_abs_M_error = std::max(max_abs_M_error, std::abs(M_error));
        out << s_norm << "," << h << "," << I2
            << "," << N << "," << M << "," << M_curv
            << "," << expected_M_curv << "," << M_error
            << "," << active_moment_to_stress_sign << "\n";
    }
    out.flush();
    pout << "  active_section_consistency.csv written (target: |N|<1e-6, "
         << "|M_curv-expected|<1e-3; expected M_curv includes "
         << "ACTIVE_I2_H_POWER scaling)\n";
    pout << "  active section max|N| = " << max_abs_N
         << ", max|M_curv-expected| = " << max_abs_M_error << "\n";
    if (max_abs_N > 1.0e-6 || max_abs_M_error > 1.0e-3)
    {
        TBOX_WARNING("Active section consistency check failed: max|N| = "
                     << max_abs_N << ", max|M_curv-expected| = " << max_abs_M_error
                     << ". Check ACTIVE_BAND_FRACTION / I2_eff.\n");
    }
}

static void
write_active_moment_profile_debug()
{
    if (IBTK_MPI::getRank() != 0 || !s_active_moment_profile_debug_enable) return;

    std::ofstream out(s_active_moment_profile_debug_filename.c_str());
    if (!out.is_open())
    {
        TBOX_WARNING("write_active_moment_profile_debug(): cannot open "
                     << s_active_moment_profile_debug_filename << "\n");
        return;
    }

    out << "s_norm,s,x_ref,x_ref_norm,xi,h,K_shape,active_envelope"
        << ",h2_active_envelope"
        << ",prefactor_unit_ramp"
        << ",Mm_full_phase0,Mm_full_phase_quarter,Mm_full_phase_half"
        << ",curvature_moment_full_phase0"
        << ",I2_eff,I2_c,I2_use,eta_bar,eta_bar_over_h"
        << ",selected_q_upper,selected_q_lower"
        << ",T_upper_phase0,T_lower_phase0,T_upper_plus_lower"
        << ",q_FE_upper,q_FE_lower,q_abs_max"
        << ",Mm_max,Mm_clamped,Mm_clamped_over_Mm"
        << ",M_command_amplitude,M_applied_amplitude,applied_amplitude_over_command"
        << ",curvature_moment_applied_phase0"
        << ",T_FE_upper,T_FE_lower"
        << ",fe_normalized_stress"
        << ",active_moment_to_stress_sign\n";
    out.setf(std::ios::scientific);
    out.precision(10);

    const int ns = std::max(101, reference_profile_bins);
    const double period = (wave_frequency > 0.0) ? 1.0 / wave_frequency : 1.0;
    for (int i = 0; i < ns; ++i)
    {
        const double s_norm = (ns > 1) ?
            static_cast<double>(i) / static_cast<double>(ns - 1) : 0.0;
        const double s = s_norm * std::max(ref_arc_length, 1.0e-12);
        libMesh::Point X_mid;
        VectorValue<double> t_mid;
        sample_reference_centerline_at_s(s, X_mid, t_mid);

        const double h = body_halfthick_from_s(s);
        const double xi = active_xi_from_s_norm(s_norm);
        const double K = muscle_moment_shape_from_xi(xi);
        const double env_s = active_moment_envelope_from_s_norm(s_norm);
        const double h2_active_envelope = h * h * env_s;
        const double prefactor = active_moment_prefactor_unit_ramp_from_sample(s_norm, h);
        const double drive0 = muscle_moment_drive_from_s(s, 0.0);
        const double drive_quarter = muscle_moment_drive_from_s(s, 0.25 * period);
        const double drive_half = muscle_moment_drive_from_s(s, 0.50 * period);
        const double Mm0 = prefactor * drive0;
        const double Mm_quarter = prefactor * drive_quarter;
        const double Mm_half = prefactor * drive_half;
        const double M_curv0 = active_curvature_moment_from_active_moment(Mm0);

        const double I2_eff = active_band_second_moment(h);
        const double I2_c =
            fe_I2c_scaled_from_s_norm(s_norm, h, I2_eff);
        const double I2_use = active_I2_use(s_norm, I2_eff, I2_c);
        const double eta_bar =
            (active_section_correction_enabled() && I2_c > 1.0e-30) ?
            fe_eta_bar_from_s_norm(s_norm) : 0.0;
        const double eta_bar_over_h = (h > 1.0e-12) ? eta_bar / h : 0.0;

        ReferenceGeometrySample upper;
        upper.s = s;
        upper.eta = h;
        ReferenceGeometrySample lower;
        lower.s = s;
        lower.eta = -h;
        const double q_FE_upper = fe_normalized_section_q(upper);
        const double q_FE_lower = fe_normalized_section_q(lower);
        const double selected_q_upper = q_FE_upper;
        const double selected_q_lower = q_FE_lower;
        const double c1_local_dbg = get_c1_s_passive_local(clamp01(s / std::max(ref_arc_length, 1.0e-12)));
        const ActiveMomentCapData cap = active_moment_cap_data(Mm0, s, c1_local_dbg);
        const double M_command_amp = std::abs(prefactor);
        const ActiveMomentCapData cap_amp =
            active_moment_cap_data(M_command_amp, s, c1_local_dbg);
        const double selected_sign = -active_moment_to_stress_sign;
        const double T_upper =
            selected_sign * cap.moment_applied * selected_q_upper;
        const double T_lower =
            selected_sign * cap.moment_applied * selected_q_lower;
        const double T_FE_upper = T_upper;
        const double T_FE_lower = T_lower;

        out << s_norm
            << "," << s
            << "," << X_mid(0)
            << "," << reference_x_norm_from_s(s)
            << "," << xi
            << "," << h
            << "," << K
            << "," << env_s
            << "," << h2_active_envelope
            << "," << prefactor
            << "," << Mm0
            << "," << Mm_quarter
            << "," << Mm_half
            << "," << M_curv0
            << "," << I2_eff
            << "," << I2_c
            << "," << I2_use
            << "," << eta_bar
            << "," << eta_bar_over_h
            << "," << selected_q_upper
            << "," << selected_q_lower
            << "," << T_upper
            << "," << T_lower
            << "," << (T_upper + T_lower)
            << "," << q_FE_upper
            << "," << q_FE_lower
            << "," << cap.q_abs_max
            << "," << cap.moment_cap
            << "," << cap.moment_applied
            << "," << cap.applied_over_command
            << "," << M_command_amp
            << "," << cap_amp.moment_applied
            << "," << cap_amp.applied_over_command
            << "," << active_curvature_moment_from_active_moment(
                            cap.moment_applied)
            << "," << T_FE_upper
            << "," << T_FE_lower
            << "," << 1
            << "," << active_moment_to_stress_sign
            << "\n";
    }
    out.flush();
    pout << "  " << s_active_moment_profile_debug_filename
         << " written (checks commanded/applied moment, cap ratio, q, and "
            "upper/lower active stress)\n";
}

// =========================================================================
// FE-consistent active section data: eta_bar(s) and I2_c(s)
//
// Computes from actual FE quadrature points (using ref_laplace_node_geom):
//   eta_bar(s) = Σ w(eta)*eta*JxW  /  Σ w(eta)*JxW
//   I2_c(s)    = [Σ_bin w(eta)*eta_c²*JxW] / Δs_bin   where  eta_c = eta - eta_bar
//
// For a perfectly symmetric mesh, eta_bar ≈ 0 and the raw bin-width-normalized
// I2_c ≈ active_band_second_moment_geometric(h).  Runtime rescales I2_c to
// ACTIVE_I2_H_POWER before using it in the stress denominator.
// For a mesh with small asymmetry, eta_c = eta - eta_bar re-centers the stress
// so that Σ T_act * JxW = 0 exactly at the FE quadrature level.
// =========================================================================
static void
write_fe_active_section_diagnostic()
{
    if (IBTK_MPI::getRank() != 0 || !s_fe_section_data_built) return;

    std::ofstream out(s_fe_section_diag_filename);
    if (!out.is_open())
    {
        TBOX_WARNING("write_fe_active_section_diagnostic(): cannot open "
                     << s_fe_section_diag_filename << "\n");
        return;
    }
    out << "s_norm,eta_bar,I2_ideal,I2_c_raw,I2_c_scaled"
        << ",I2_c_over_I2_ideal,eta_bar_over_h\n";
    out.setf(std::ios::scientific);
    out.precision(8);

    const int nb = s_fe_n_bins;
    double max_abs_eta_bar_over_h = 0.0;
    double min_I2_ratio = std::numeric_limits<double>::infinity();
    double max_I2_ratio = 0.0;
    int    n_I2_ratio   = 0;
    for (int i = 0; i < nb; ++i)
    {
        const double s_norm = (nb > 1) ?
            static_cast<double>(i) / static_cast<double>(nb - 1) : 0.0;
        const double s      = s_norm * std::max(ref_arc_length, 1.0e-12);
        const double h      = body_halfthick_from_s(s);
        const double I2_ideal = active_band_second_moment(h);
        const double I2_c_scaled = fe_I2c_scaled_from_s_norm(s_norm, h, I2_ideal);
        const double I2_ratio = (I2_ideal > 1.0e-30) ? I2_c_scaled / I2_ideal : 0.0;
        const double ebar_h = (h > 1.0e-12) ? s_fe_eta_bar[i] / h : 0.0;
        max_abs_eta_bar_over_h = std::max(max_abs_eta_bar_over_h, std::abs(ebar_h));
        if (I2_ideal > 1.0e-30 && I2_c_scaled > 1.0e-30)
        {
            min_I2_ratio = std::min(min_I2_ratio, I2_ratio);
            max_I2_ratio = std::max(max_I2_ratio, I2_ratio);
            ++n_I2_ratio;
        }
        out << s_norm << "," << s_fe_eta_bar[i] << "," << I2_ideal
            << "," << s_fe_I2_c[i] << "," << I2_c_scaled
            << "," << I2_ratio << "," << ebar_h << "\n";
    }
    out.flush();
    pout << "  fe_active_section.csv written\n";
    pout << "  max |eta_bar/h| = " << max_abs_eta_bar_over_h << "\n";
    if (n_I2_ratio > 0)
    {
        pout << "  FE I2_c/I2_ideal range = [" << min_I2_ratio
             << ", " << max_I2_ratio << "] over " << n_I2_ratio << " active bins\n";
        if (min_I2_ratio < 0.1 || max_I2_ratio > 10.0)
        {
            TBOX_WARNING("FE active section: I2_c/I2_ideal is far from O(1) "
                         << "(range [" << min_I2_ratio << ", " << max_I2_ratio
                         << "]). Check mesh symmetry, bin width normalization, and active band support.\n");
        }
    }
    if (max_abs_eta_bar_over_h > 1.0e-3)
    {
        TBOX_WARNING("FE active section: max |eta_bar/h| = " << max_abs_eta_bar_over_h
                     << " — mesh asymmetry is significant; eta_c correction is active.\n");
    }
    else
    {
        pout << "  mesh is nearly symmetric (|eta_bar/h| < 1e-3); "
                "eta_c correction is small.\n";
    }
}

static void
build_fe_section_data(MeshBase& mesh)
{
    if (!ref_laplace_parameterization_built)
    {
        pout << "  build_fe_section_data(): Laplace parameterization not built; skipping.\n";
        return;
    }

    s_fe_n_bins = std::max(8, reference_profile_bins);
    const int nb = s_fe_n_bins;
    s_fe_sum_w.assign(nb, 0.0);
    s_fe_sum_w_eta.assign(nb, 0.0);
    s_fe_sum_w_eta2.assign(nb, 0.0);
    s_fe_eta_bar.assign(nb, 0.0);
    s_fe_I2_c.assign(nb, 0.0);

    // FIFTH-order quadrature matches PK1_ACT_QUAD_ORDER.
    const libMesh::Order quad_order = libMesh::FIFTH;
    const unsigned int dim = mesh.mesh_dimension();
    // Determine FE type from the first local element.
    FEType fe_type(FIRST, LAGRANGE);
    {
        auto el0 = mesh.active_local_elements_begin();
        if (el0 != mesh.active_local_elements_end())
            fe_type = FEType((*el0)->default_order(), LAGRANGE);
    }
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase>  qrule(QBase::build(QGAUSS, dim, quad_order));
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<Real>&                    JxW = fe->get_JxW();
    const std::vector<std::vector<Real>>&       phi = fe->get_phi();

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        fe->reinit(elem);
        const unsigned int nq = static_cast<unsigned int>(JxW.size());
        const unsigned int nn = elem->n_nodes();

        for (unsigned int qp = 0; qp < nq; ++qp)
        {
            double s_qp = 0.0, eta_qp = 0.0;
            bool ok = true;
            for (unsigned int k = 0; k < nn; ++k)
            {
                auto it = ref_laplace_node_geom.find(elem->node_id(k));
                if (it == ref_laplace_node_geom.end()) { ok = false; break; }
                s_qp   += phi[k][qp] * it->second.s;
                eta_qp += phi[k][qp] * it->second.eta;
            }
            if (!ok) continue;

            const double h = body_halfthick_from_s(s_qp);
            if (h < 1.0e-10) continue;

            const double s_norm =
                clamp01(s_qp / std::max(ref_arc_length, 1.0e-12));
            // Round to nearest bin; clamp to valid range.
            const int bi = std::min(nb - 1,
                           std::max(0, static_cast<int>(s_norm * (nb - 1) + 0.5)));

            const double jw = JxW[qp];
            const double w = active_band_weight(eta_qp, h);
            if (w <= 0.0) continue;

            s_fe_sum_w[bi]     += w * jw;
            s_fe_sum_w_eta[bi] += w * eta_qp * jw;
            s_fe_sum_w_eta2[bi]+= w * eta_qp * eta_qp * jw;
        }
    }

    // Global reduction across MPI ranks.
    IBTK_MPI::sumReduction(s_fe_sum_w.data(),     nb);
    IBTK_MPI::sumReduction(s_fe_sum_w_eta.data(), nb);
    IBTK_MPI::sumReduction(s_fe_sum_w_eta2.data(),nb);

    for (int i = 0; i < nb; ++i)
    {
        const double s_norm_left = (i == 0 || nb <= 1) ?
            0.0 : (static_cast<double>(i) - 0.5) / static_cast<double>(nb - 1);
        const double s_norm_right = (i == nb - 1 || nb <= 1) ?
            1.0 : (static_cast<double>(i) + 0.5) / static_cast<double>(nb - 1);
        const double ds_bin = std::max((s_norm_right - s_norm_left) *
                                       std::max(ref_arc_length, 1.0e-12),
                                       1.0e-12);
        if (s_fe_sum_w[i] <= 1.0e-30)
        {
            s_fe_eta_bar[i] = 0.0;
            s_fe_I2_c[i]    = 0.0;
        }
        else
        {
            s_fe_eta_bar[i] = s_fe_sum_w_eta[i] / s_fe_sum_w[i];
            // The FE quadrature sums are area integrals over one s-bin:
            //   I2_bin ≈ Δs_bin * ∫_section w(eta)*(eta-eta_bar)^2 dη.
            // Divide by the physical bin width to recover the sectional second
            // moment required by T_act = sign*w*Mm*eta_c/I2_c.
            const double I2_bin = std::max(0.0,
                s_fe_sum_w_eta2[i] - s_fe_eta_bar[i] * s_fe_sum_w_eta[i]);
            s_fe_I2_c[i] = I2_bin / ds_bin;
        }
    }

    // Fill empty bins by nearest-neighbor continuation so interpolation of I2_c
    // does not blend valid section moments with zeros near sparse regions. The
    // actual active stress still falls back to the analytic I2 when all bins are empty.
    int first_valid = -1;
    for (int i = 0; i < nb; ++i)
    {
        if (s_fe_I2_c[i] > 1.0e-30) { first_valid = i; break; }
    }
    if (first_valid >= 0)
    {
        for (int i = 0; i < first_valid; ++i)
        {
            s_fe_eta_bar[i] = s_fe_eta_bar[first_valid];
            s_fe_I2_c[i]    = s_fe_I2_c[first_valid];
        }
        int last_valid = first_valid;
        for (int i = first_valid + 1; i < nb; ++i)
        {
            if (s_fe_I2_c[i] > 1.0e-30)
            {
                const int next_valid = i;
                for (int j = last_valid + 1; j < next_valid; ++j)
                {
                    const double a = static_cast<double>(j - last_valid) /
                                     static_cast<double>(next_valid - last_valid);
                    s_fe_eta_bar[j] = (1.0 - a) * s_fe_eta_bar[last_valid] +
                                      a * s_fe_eta_bar[next_valid];
                    s_fe_I2_c[j]    = (1.0 - a) * s_fe_I2_c[last_valid] +
                                      a * s_fe_I2_c[next_valid];
                }
                last_valid = next_valid;
            }
        }
        for (int i = last_valid + 1; i < nb; ++i)
        {
            s_fe_eta_bar[i] = s_fe_eta_bar[last_valid];
            s_fe_I2_c[i]    = s_fe_I2_c[last_valid];
        }
    }

    s_fe_section_data_built = true;
}

static void
initialize_fe_normalized_stress(MeshBase& mesh,
                                EquationSystems* equation_systems)
{
    fe_normalized_sections.clear();
    if (!equation_systems)
    {
        TBOX_ERROR("initialize_fe_normalized_stress(): null equation systems.\n");
    }

    const int n_bins = std::max(fe_normalized_stress_bins, 1);
    fe_normalized_stress_bins = n_bins;
    const double ds = std::max(ref_arc_length, 1.0e-12) / n_bins;
    const int n_fields = 4;
    std::vector<double> moments(
        static_cast<std::size_t>(n_bins * n_fields), 0.0);

    System& ref_geom_sys = equation_systems->get_system(REF_GEOM_SYSTEM_NAME);
    NumericVector<double>* ref_geom_vec = ref_geom_sys.solution.get();
    NumericVector<double>* ref_geom_ghost_vec =
        ref_geom_sys.current_local_solution.get();
    ref_geom_vec->close();
    copy_and_synch(*ref_geom_vec, *ref_geom_ghost_vec);

    const unsigned int dim = mesh.mesh_dimension();
    const DofMap& dof_map = ref_geom_sys.get_dof_map();
    const FEType fe_type = dof_map.variable_type(0);
    const libMesh::Order quad_order =
        Utility::string_to_enum<libMesh::Order>(
            fe_normalized_stress_quad_order);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule(new QGauss(dim, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<double> >& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();
    std::vector<std::vector<unsigned int> > dof_indices(REF_GEOM_N_VARS);

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;
        fe->reinit(elem);
        for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v)
            dof_map.dof_indices(elem, dof_indices[v], v);

        const unsigned int n_nodes = elem->n_nodes();
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            double s = 0.0;
            double eta = 0.0;
            RealGradient grad_s;
            grad_s.zero();
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                const double s_node =
                    (*ref_geom_ghost_vec)(dof_indices[REF_GEOM_S][k]);
                s += phi[k][qp] * s_node;
                eta += phi[k][qp] *
                       (*ref_geom_ghost_vec)(dof_indices[REF_GEOM_ETA][k]);
                grad_s.add_scaled(dphi[k][qp], s_node);
            }

            s = std::max(0.0, std::min(s, ref_arc_length));
            const double h = body_halfthick_from_s(s);
            const double g = fe_normalized_section_shape(eta, h);
            const double grad_s_norm =
                std::sqrt(std::max(grad_s * grad_s, 0.0));
            const double weight = grad_s_norm * JxW[qp];
            const std::size_t offset = static_cast<std::size_t>(
                fe_normalized_section_bin_from_s(s) * n_fields);
            moments[offset + 0] += weight;
            moments[offset + 1] += g * weight;
            moments[offset + 2] += eta * weight;
            moments[offset + 3] += g * eta * weight;
        }
    }

    IBTK_MPI::sumReduction(moments.data(), static_cast<int>(moments.size()));
    fe_normalized_sections.assign(
        static_cast<std::size_t>(n_bins), FENormalizedSection());

    int invalid_bins = 0;
    double max_unit_force = 0.0;
    double max_unit_moment_error = 0.0;
    for (int bin = 0; bin < n_bins; ++bin)
    {
        const std::size_t offset =
            static_cast<std::size_t>(bin * n_fields);
        const double area = moments[offset + 0];
        const double g_integral = moments[offset + 1];
        const double eta_integral = moments[offset + 2];
        const double g_eta_integral = moments[offset + 3];

        FENormalizedSection& normalization =
            fe_normalized_sections[static_cast<std::size_t>(bin)];
        normalization.s_mid = (bin + 0.5) * ds;
        if (area <= 1.0e-18)
        {
            ++invalid_bins;
            continue;
        }

        normalization.g_mean = g_integral / area;
        const double centered_g_eta =
            g_eta_integral - normalization.g_mean * eta_integral;
        if (std::abs(centered_g_eta) <= 1.0e-18)
        {
            ++invalid_bins;
            continue;
        }

        normalization.q_scale = -ds / centered_g_eta;
        normalization.q_abs_max =
            std::abs(normalization.q_scale) *
            std::max(std::abs(-1.0 - normalization.g_mean),
                     std::abs(1.0 - normalization.g_mean));
        normalization.unit_force =
            normalization.q_scale *
            (g_integral - normalization.g_mean * area) / ds;
        normalization.unit_moment =
            -normalization.q_scale * centered_g_eta / ds;
        normalization.valid =
            std::isfinite(normalization.q_scale) &&
            std::isfinite(normalization.q_abs_max);
        if (!normalization.valid)
        {
            ++invalid_bins;
            continue;
        }

        max_unit_force =
            std::max(max_unit_force, std::abs(normalization.unit_force));
        max_unit_moment_error =
            std::max(max_unit_moment_error,
                     std::abs(normalization.unit_moment - 1.0));
    }

    if (invalid_bins > 0)
    {
        TBOX_WARNING("FE_NORMALIZED_STRESS has " << invalid_bins
                     << " invalid/empty section bins; those bins use the "
                        "analytic zero-force/unit-moment fallback.\n");
    }
    pout << "  FE-normalized active stress: bins=" << n_bins
         << ", invalid=" << invalid_bins
         << ", max |unit force|=" << max_unit_force
         << ", max |unit moment-1|=" << max_unit_moment_error << "\n";
}

// =========================================================================
// Raw passive continuum PK1 stress.
static void
compute_raw_passive_dev_PK1_stress(TensorValue<double>& PP,
                                   const TensorValue<double>& FF,
                                   const libMesh::Point& X_ref)
{
    const double c1_local = get_c1_s_passive_local_from_reference_point(X_ref);
    PP = 2.0 * c1_local * (FF - tensor_inverse_transpose(FF, NDIM));
}

static void
compute_raw_passive_dil_PK1_stress(TensorValue<double>& PP,
                                   const TensorValue<double>& FF,
                                   const libMesh::Point& X_ref)
{
    const double J = FF.det();
    if (!(J > 1.0e-14) || !std::isfinite(J))
    {
        TBOX_ERROR("Passive dilational stress encountered non-positive "
                   "or non-finite det(F) = " << J
                   << " at X_ref=(" << X_ref(0) << ", " << X_ref(1)
                   << ").\n");
    }

    PP = kappa_vol * std::log(J) * tensor_inverse_transpose(FF, NDIM);
}

static void
compute_passive_dev_PK1_stress_impl(TensorValue<double>& PP,
                                    const TensorValue<double>& FF,
                                    const libMesh::Point& X_ref)
{
    const double J = FF.det();
    if (!(J > 1.0e-14) || !std::isfinite(J)) { PP = TensorValue<double>(0.0); return; }
    compute_raw_passive_dev_PK1_stress(PP, FF, X_ref);
}

static void
compute_passive_dil_PK1_stress_impl(TensorValue<double>& PP,
                                    const TensorValue<double>& FF,
                                    const libMesh::Point& X_ref)
{
    const double J = FF.det();
    if (!(J > 1.0e-14) || !std::isfinite(J)) { PP = TensorValue<double>(0.0); return; }
    compute_raw_passive_dil_PK1_stress(PP, FF, X_ref);
}

void PK1_dev_stress_function(TensorValue<double>& PP,
                             const TensorValue<double>& FF,
                             const libMesh::Point& /*X*/,
                             const libMesh::Point& X_ref,
                             Elem* const,
                             const std::vector<const std::vector<double>*>& system_var_data,
                             const std::vector<const std::vector<VectorValue<double> >*>&,
                             double /*time*/,
                             void*)
{
    compute_passive_dev_PK1_stress_impl(PP, FF, X_ref);
}

void PK1_dil_stress_function(TensorValue<double>& PP,
                             const TensorValue<double>& FF,
                             const libMesh::Point& /*X*/,
                             const libMesh::Point& X_ref,
                             Elem* const,
                             const std::vector<const std::vector<double>*>& system_var_data,
                             const std::vector<const std::vector<VectorValue<double> >*>&,
                             double /*time*/,
                             void*)
{
    compute_passive_dil_PK1_stress_impl(PP, FF, X_ref);
}

// =========================================================================
// Optional continuum structural damping.
//
// This is a continuum viscous stress, not the old EB/KV bending model. It uses
// the current fiber tangent and spatial rate-of-deformation tensor so rigid
// translation and rigid rotation do not generate damping stress:
//
//     sigma_visc = eta_s(s) * (a_hat . D . a_hat) * (a_hat tensor a_hat)
//     P_visc    = J * sigma_visc * F^{-T}
//
// with eta_s scaled from the local passive shear modulus by
// CONTINUUM_DAMPING_FACTOR / omega.
// =========================================================================
static void
compute_continuum_damping_PK1_stress_impl(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point& X_ref,
    const ReferenceGeometrySample& ref_geom,
    const TensorValue<double>& F_dot)
{
    PP = 0.0;
    if (!use_continuum_damping || continuum_damping_factor <= 0.0) return;

    const double J = FF.det();
    if (!(J > 1.0e-14) || !std::isfinite(J)) return;

    TensorValue<double> F_inv;
    tensor_inverse(F_inv, FF, NDIM);
    const TensorValue<double> F_inv_trans = tensor_inverse_transpose(FF, NDIM);
    const TensorValue<double> L = F_dot * F_inv;
    const TensorValue<double> D = 0.5 * (L + L.transpose());

    const VectorValue<double> a = FF * ref_geom.t_hat;
    const double a_norm = std::sqrt(std::max(a * a, 0.0));
    if (a_norm <= 1.0e-14) return;
    const VectorValue<double> a_hat = a / a_norm;

    const double axial_strain_rate = a_hat * (D * a_hat);
    const double c1_local = get_c1_s_passive_local_from_reference_point(X_ref);
    const double omega_ref = std::max(std::abs(wave_omega), 1.0e-12);
    const double eta_local =
        continuum_damping_factor * 2.0 * c1_local / omega_ref;
    const double T_raw = eta_local * axial_strain_rate;
    const double T_cap =
        continuum_damping_stress_cap_over_c1 * c1_local;
    const double T_visc = (T_cap > 0.0) ?
        std::max(-T_cap, std::min(T_cap, T_raw)) : T_raw;

    TensorValue<double> aa;
    outer_product(aa, a_hat, a_hat);
    const TensorValue<double> sigma_visc = T_visc * aa;
    PP = J * sigma_visc * F_inv_trans;
}

void PK1_continuum_damping_stress_function(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point&,
    const libMesh::Point& X_ref,
    Elem* const,
    const std::vector<const std::vector<double>*>& system_var_data,
    const std::vector<const std::vector<VectorValue<double> >*>& system_grad_var_data,
    double,
    void*)
{
    PP = 0.0;
    if (!use_continuum_damping || continuum_damping_factor <= 0.0) return;

    if (system_grad_var_data.size() < 2 || system_grad_var_data[1] == nullptr ||
        system_grad_var_data[1]->size() < NDIM)
    {
        TBOX_ERROR("PK1_continuum_damping_stress_function(): "
                   "velocity-gradient data are unavailable.\n");
    }

    TensorValue<double> F_dot;
    F_dot.zero();
    const std::vector<VectorValue<double> >& grad_U = *system_grad_var_data[1];
    for (unsigned int i = 0; i < NDIM; ++i)
    {
        for (unsigned int j = 0; j < NDIM; ++j)
        {
            F_dot(i, j) = grad_U[i](j);
        }
    }

    const ReferenceGeometrySample ref_geom =
        reference_geometry_from_system_data(system_var_data);
    compute_continuum_damping_PK1_stress_impl(PP, FF, X_ref, ref_geom, F_dot);
}

// =========================================================================
// Active internal bending moment as a zero-resultant axial fiber PK1 stress.
// =========================================================================
static void
compute_active_PK1_stress_impl(TensorValue<double>& PP,
                               const TensorValue<double>& FF,
                               const libMesh::Point& X_ref,
                               const ReferenceGeometrySample& ref_geom,
                               const double time)
{
    PP = 0.0;

    if (active_moment_mode == ActiveMomentMode::TRAVELING &&
        active_traveling_amplitude() <= 0.0) return;
    if (active_moment_mode == ActiveMomentMode::STATIC &&
        std::abs(static_moment_m0) <= 1.0e-30) return;

    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double h = body_halfthick_from_s(s);
    if (h < 1.0e-10) return;

    const double Mm = active_moment_value_from_sample(s, s_norm, h, time);
    if (std::abs(Mm) <= 1.0e-30) return;

    const ActiveMomentCapData cap = active_moment_cap_data(
        Mm, s, get_c1_s_passive_local_from_reference_point(X_ref));
    if (std::abs(cap.moment_applied) <= 1.0e-30) return;

    // FE q satisfies -int(q*eta)dA=1.
    const double q = fe_normalized_section_q(ref_geom);
    if (std::abs(q) <= 1.0e-30) return;
    const double T_act =
        -active_moment_to_stress_sign * cap.moment_applied * q;

    const VectorValue<double> f0 = ref_geom.t_hat;
    TensorValue<double> f_f;
    outer_product(f_f, f0, f0);
    PP = T_act * FF * f_f;
}

void PK1_active_stress_function(TensorValue<double>& PP,
                                const TensorValue<double>& FF,
                                const libMesh::Point&,
                                const libMesh::Point& X_ref,
                                Elem* const,
                                const std::vector<const std::vector<double>*>& system_var_data,
                                const std::vector<const std::vector<VectorValue<double> >*>&,
                                double time,
                                void*)
{
    const ReferenceGeometrySample ref_geom =
        reference_geometry_from_system_data(system_var_data);
    compute_active_PK1_stress_impl(PP, FF, X_ref, ref_geom, time);
}

// =========================================================================
// Reference and current COM integration over the Lagrangian FE mesh
//
// The reference COM is computed once from the reference structural mesh and
// is used as the origin of the body-fixed diagnostic transform. The current
// COM diagnostic is the current geometric centroid:
    //     x_cm = int chi(X,t) |J| dX / int |J| dX.
// =========================================================================
static void
initialize_reference_com_from_mesh(MeshBase& mesh)
{
    double x_sum = 0.0, y_sum = 0.0, area_sum = 0.0;

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem || elem->n_vertices() < 3) continue;

        const libMesh::Point& X0 = elem->point(0);
        for (unsigned int k = 1; k + 1 < elem->n_vertices(); ++k)
        {
            const libMesh::Point& X1 = elem->point(k);
            const libMesh::Point& X2 = elem->point(k + 1);

            const double ax = X1(0) - X0(0);
            const double ay = X1(1) - X0(1);
            const double bx = X2(0) - X0(0);
            const double by = X2(1) - X0(1);
            const double area = 0.5 * std::abs(ax * by - ay * bx);
            if (area <= 1.0e-30) continue;

            const double cx = (X0(0) + X1(0) + X2(0)) / 3.0;
            const double cy = (X0(1) + X1(1) + X2(1)) / 3.0;
            x_sum += cx * area;
            y_sum += cy * area;
            area_sum += area;
        }
    }

    double global[3] = { x_sum, y_sum, area_sum };
    IBTK_MPI::sumReduction(global, 3);
    if (global[2] <= 1.0e-30)
    {
        TBOX_ERROR("initialize_reference_com_from_mesh(): reference mesh area is zero.\n");
    }

    s_reference_xcom = global[0] / global[2];
    s_reference_ycom = global[1] / global[2];
    s_reference_area = global[2];
}

static void
compute_fish_com(Pointer<IBFEMethod>  ib_method_ops,
                 MeshBase&            mesh,
                 EquationSystems*     equation_systems,
                 double&              x_cm_out,
                 double&              y_cm_out,
                 double*              area_out = nullptr)
{
    if (!equation_systems) { x_cm_out = xcom_tracked; y_cm_out = ycom_tracked;
                             if (area_out) *area_out = 0.0; return; }

    // Get the current-coordinates system (χ at current time)
    System& X_sys       = equation_systems->get_system(ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec       = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);

    const DofMap& X_dof_map = X_sys.get_dof_map();
    const unsigned int dim = mesh.mesh_dimension();
    const FEType fe_type = X_dof_map.variable_type(0);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule = fe_type.default_quadrature_rule(dim);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<double>>& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();
    std::vector<std::vector<unsigned int>> dof_indices(NDIM);
    boost::multi_array<double, 2> X_node;

    double x_sum = 0.0, y_sum = 0.0, area_sum = 0.0;

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d)
            X_dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            TensorValue<double> FF;
            jacobian(FF, qp, X_node, dphi);
            const double area = std::abs(FF.det()) * JxW[qp];
            if (area <= 1.0e-30) continue;

            VectorValue<double> x_qp;
            interpolate(x_qp, qp, X_node, phi);
            x_sum += x_qp(0) * area;
            y_sum += x_qp(1) * area;
            area_sum += area;
        }
    }

    // Sum across MPI ranks
    double global[3] = { x_sum, y_sum, area_sum };
    IBTK_MPI::sumReduction(global, 3);

    x_cm_out = (global[2] > 0.0) ? global[0] / global[2] : xcom_tracked;
    y_cm_out = (global[2] > 0.0) ? global[1] / global[2] : ycom_tracked;
    if (area_out) *area_out = global[2];
}

struct GeometryConservationMetrics
{
    double area = 0.0;
    double area_abs = 0.0;
    double area_rel_error = std::numeric_limits<double>::quiet_NaN();
    double area_abs_rel_error = std::numeric_limits<double>::quiet_NaN();
    double J_min = std::numeric_limits<double>::quiet_NaN();
    double J_max = std::numeric_limits<double>::quiet_NaN();
    double J_mean = std::numeric_limits<double>::quiet_NaN();
    double J_rms_error = std::numeric_limits<double>::quiet_NaN();
    double J_max_abs_error = std::numeric_limits<double>::quiet_NaN();
};

static GeometryConservationMetrics
compute_geometry_conservation_metrics(Pointer<IBFEMethod>  ib_method_ops,
                                      MeshBase&            mesh,
                                      EquationSystems*     equation_systems)
{
    GeometryConservationMetrics metrics;
    if (!equation_systems) return metrics;

    System& X_sys = equation_systems->get_system(
        ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec       = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);

    const DofMap& X_dof_map = X_sys.get_dof_map();
    const unsigned int dim = mesh.mesh_dimension();
    const FEType fe_type = X_dof_map.variable_type(0);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule = fe_type.default_quadrature_rule(dim);
    fe->attach_quadrature_rule(qrule.get());
    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<RealGradient>>& dphi = fe->get_dphi();
    std::vector<std::vector<unsigned int>> dof_indices(NDIM);
    boost::multi_array<double, 2> X_node;

    double area = 0.0;
    double area_abs = 0.0;
    double ref_area_q = 0.0;
    double J_sum = 0.0;
    double J_err2_sum = 0.0;
    double J_min = std::numeric_limits<double>::infinity();
    double J_max = -std::numeric_limits<double>::infinity();
    double J_max_abs_error = 0.0;

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;
        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d)
            X_dof_map.dof_indices(elem, dof_indices[d], d);
        get_values_for_interpolation(X_node, *X_ghost_vec, dof_indices);

        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            TensorValue<double> FF;
            jacobian(FF, qp, X_node, dphi);
            const double J = FF.det();
            const double w = JxW[qp];
            area += J * w;
            area_abs += std::abs(J) * w;
            ref_area_q += w;
            J_sum += J * w;
            const double J_err = J - 1.0;
            J_err2_sum += J_err * J_err * w;
            J_min = std::min(J_min, J);
            J_max = std::max(J_max, J);
            J_max_abs_error = std::max(J_max_abs_error, std::abs(J_err));
        }
    }

    double global[8] = {
        area, area_abs, ref_area_q, J_sum, J_err2_sum,
        J_min, J_max, J_max_abs_error
    };
    IBTK_MPI::sumReduction(global, 5);
    IBTK_MPI::minReduction(&global[5], 1);
    IBTK_MPI::maxReduction(&global[6], 1);
    IBTK_MPI::maxReduction(&global[7], 1);

    metrics.area = global[0];
    metrics.area_abs = global[1];
    const double ref_area = s_reference_area > 1.0e-30 ? s_reference_area : global[2];
    if (ref_area > 1.0e-30)
    {
        metrics.area_rel_error = (metrics.area - ref_area) / ref_area;
        metrics.area_abs_rel_error = (metrics.area_abs - ref_area) / ref_area;
    }
    if (global[2] > 1.0e-30)
    {
        metrics.J_mean = global[3] / global[2];
        metrics.J_rms_error = std::sqrt(std::max(global[4] / global[2], 0.0));
    }
    metrics.J_min = global[5];
    metrics.J_max = global[6];
    metrics.J_max_abs_error = global[7];
    return metrics;
}

static void
write_geometry_conservation_diagnostics(const int            iteration_num,
                                        const double         loop_time,
                                        Pointer<IBFEMethod>  ib_method_ops,
                                        MeshBase&            mesh,
                                        EquationSystems*     equation_systems)
{
    if (!s_geometry_conservation_diag_enable) return;
    if (s_geometry_conservation_diag_interval > 1 &&
        (iteration_num % s_geometry_conservation_diag_interval != 0)) return;

    const GeometryConservationMetrics metrics =
        compute_geometry_conservation_metrics(ib_method_ops, mesh, equation_systems);

    if (IBTK_MPI::getRank() != 0) return;
    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_geometry_conservation_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_geometry_conservation_diagnostics(): cannot open "
                         << s_geometry_conservation_diag_filename << "\n");
            return;
        }
        out << "step,time,reference_area,current_area,current_area_abs"
            << ",area_rel_error,area_abs_rel_error"
            << ",J_min,J_max,J_mean,J_rms_error,J_max_abs_error\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    out << iteration_num
        << "," << loop_time
        << "," << s_reference_area
        << "," << metrics.area
        << "," << metrics.area_abs
        << "," << metrics.area_rel_error
        << "," << metrics.area_abs_rel_error
        << "," << metrics.J_min
        << "," << metrics.J_max
        << "," << metrics.J_mean
        << "," << metrics.J_rms_error
        << "," << metrics.J_max_abs_error
        << "\n";
    out.flush();
}

inline double forward_x_from_com(const double x_cm)
{
    const double forward_sign = head_is_at_x_min() ? -1.0 : 1.0;
    return forward_sign * (x_cm - s_reference_xcom);
}

static void
write_direction_debug_diagnostics(const int    iteration_num,
                                  const double loop_time,
                                  const double F_integral[NDIM],
                                  const double x_cm,
                                  const double y_cm,
                                  const double vcm_x,
                                  const double vcm_y,
                                  const double tail_A_norm,
                                  const double fish_area)
{
    if (!s_direction_debug_enable) return;
    if (s_direction_debug_interval > 1 &&
        (iteration_num % s_direction_debug_interval != 0)) return;
    if (IBTK_MPI::getRank() != 0) return;

    const double forward_sign = head_is_at_x_min() ? -1.0 : 1.0;
    const double x_forward = forward_x_from_com(x_cm);
    const double v_forward = forward_sign * vcm_x;
    const double v_lateral = forward_sign * vcm_y;

    // IB Lagrangian force integral (force fish structure exerts ON the fluid).
    // NOTE: in IBFE with neutral buoyancy this is NOT the net propulsive force
    // on the fish CM — it includes internal structural forces and does not satisfy
    // F = m*a for the CM. Kept only for reference.
    const double F_IB_on_fluid_fwd = forward_sign * F_integral[0];
    const double F_IB_on_fish_fwd  = -F_IB_on_fluid_fwd;
    const double F_IB_on_fish_lat  = -forward_sign * F_integral[1];

    double dt_eff    = std::numeric_limits<double>::quiet_NaN();
    double a_forward = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(s_direction_debug_prev_time))
    {
        dt_eff = loop_time - s_direction_debug_prev_time;
        if (dt_eff > 1.0e-12 && std::isfinite(s_direction_debug_prev_v_forward))
        {
            a_forward = (v_forward - s_direction_debug_prev_v_forward) / dt_eff;

            // Legacy IB-based impulse (retained but misleading — see F_IB note above)
            s_direction_debug_fish_impulse_forward += F_IB_on_fish_fwd * dt_eff;
            s_direction_debug_fish_work_forward    += F_IB_on_fish_fwd * v_forward * dt_eff;
        }
    }
    s_direction_debug_prev_time      = loop_time;
    s_direction_debug_prev_v_forward = v_forward;

    // Correct propulsive force from Newton's 2nd law: F_CM = m_eff * a_cm.
    // m_eff = fluid_density * fish_area (neutrally buoyant IBFE fish).
    // This IS the net force driving the fish CM; its sign is the thrust direction.
    const double F_CM_fwd = (std::isfinite(a_forward) && fish_area > 1.0e-30) ?
        fluid_density * fish_area * a_forward :
        std::numeric_limits<double>::quiet_NaN();

    // Correct CM impulse = ∫ F_CM dt  =  m_eff * Δv_fwd  (meaningful momentum integral)
    if (std::isfinite(F_CM_fwd) && std::isfinite(dt_eff))
        s_direction_debug_cm_impulse_fwd += F_CM_fwd * dt_eff;

    // Sliding-window cycle average of v_forward (~1 beat period)
    s_vfwd_win[s_vfwd_win_pos] = v_forward;
    s_vfwd_win_pos   = (s_vfwd_win_pos + 1) % VFWD_WIN;
    s_vfwd_win_count = std::min(s_vfwd_win_count + 1, VFWD_WIN);
    double v_fwd_cycle_avg = 0.0;
    for (int i = 0; i < s_vfwd_win_count; ++i) v_fwd_cycle_avg += s_vfwd_win[i];
    v_fwd_cycle_avg /= static_cast<double>(s_vfwd_win_count);

    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_direction_debug_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_direction_debug_diagnostics(): cannot open "
                         << s_direction_debug_filename << "\n");
            return;
        }
        // Column legend:
        //   F_IB_*  : Lagrangian IB force integral — NOT the true CM thrust (see note above)
        //   F_CM_fwd: Newton 2nd law force on fish CM = fluid_density*fish_area*a_fwd  [RELIABLE]
        //   cm_impulse_fwd: integral of F_CM_fwd — true forward momentum gained  [RELIABLE]
        //   v_fwd_cycle_avg: sliding-window mean of v_fwd over ~1 beat period    [RELIABLE]
        out << "step,time,dt_eff,forward_sign"
            << ",x_cm,y_cm,x_forward,vcm_x,vcm_y,v_forward,v_lateral,a_forward"
            << ",v_fwd_cycle_avg"
            << ",fish_area"
            << ",F_CM_fwd,cm_impulse_fwd"
            << ",F_IB_on_fluid_fwd,F_IB_on_fish_fwd,F_IB_on_fish_lat"
            << ",F_IB_impulse_fwd,F_IB_work_fwd"
            << ",wave_time_sign,active_moment_to_stress_sign,tail_A_norm\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    out << iteration_num
        << "," << loop_time
        << "," << dt_eff
        << "," << forward_sign
        << "," << x_cm
        << "," << y_cm
        << "," << x_forward
        << "," << vcm_x
        << "," << vcm_y
        << "," << v_forward
        << "," << v_lateral
        << "," << a_forward
        << "," << v_fwd_cycle_avg
        << "," << fish_area
        << "," << F_CM_fwd
        << "," << s_direction_debug_cm_impulse_fwd
        << "," << F_IB_on_fluid_fwd
        << "," << F_IB_on_fish_fwd
        << "," << F_IB_on_fish_lat
        << "," << s_direction_debug_fish_impulse_forward
        << "," << s_direction_debug_fish_work_forward
        << "," << wave_time_sign
        << "," << active_moment_to_stress_sign
        << "," << tail_A_norm
        << "\n";
    out.flush();
}

struct ReferenceFrame
{
    libMesh::Point X = libMesh::Point();
    VectorValue<double> t_hat = VectorValue<double>(1.0, 0.0);
    VectorValue<double> n_hat = VectorValue<double>(0.0, 1.0);
};

struct MidlineSample
{
    double s = 0.0;
    double s_norm = 0.0;
    ReferenceFrame ref;
    double x_lab = 0.0;
    double y_lab = 0.0;
    double x_body = 0.0;
    double y_body = 0.0;
    double h = 0.0;
};

double phase01(const double time)
{
    if (!(wave_frequency > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    double phase = std::fmod(time * wave_frequency, 1.0);
    if (phase < 0.0) phase += 1.0;
    return phase;
}

ReferenceFrame reference_frame_at_s(const double s_query)
{
    if (ref_centerline_segments.empty())
    {
        TBOX_ERROR("reference_frame_at_s(): reference centerline has not been built.\n");
    }

    const double s = std::max(0.0, std::min(s_query, ref_arc_length));
    const CenterlineSegment* seg = &ref_centerline_segments.front();
    if (s >= ref_arc_length)
    {
        seg = &ref_centerline_segments.back();
    }
    else
    {
        for (const CenterlineSegment& candidate : ref_centerline_segments)
        {
            if (s <= candidate.s0 + candidate.len + 1.0e-14)
            {
                seg = &candidate;
                break;
            }
        }
    }

    const double alpha = std::max(0.0, std::min((s - seg->s0) / std::max(seg->len, 1.0e-24), 1.0));
    ReferenceFrame frame;
    frame.X = libMesh::Point((1.0 - alpha) * seg->X0(0) + alpha * seg->X1(0),
                             (1.0 - alpha) * seg->X0(1) + alpha * seg->X1(1));
    frame.t_hat = seg->t_hat;
    frame.n_hat = seg->n_hat;
    return frame;
}

static void
compute_current_midline_samples_at_s(Pointer<IBFEMethod>        ib_method_ops,
                                     MeshBase&                  mesh,
                                     EquationSystems*           equation_systems,
                                     const std::vector<double>& target_s_values,
                                     const double               x_cm,
                                     const double               y_cm,
                                     std::vector<MidlineSample>& samples,
                                     double&                    theta_body,
                                     const double               boundary_x_end =
                                         std::numeric_limits<double>::quiet_NaN())
{
    if (!equation_systems) return;

    const double Lref = std::max(ref_arc_length, 1.0e-12);
    const int n = static_cast<int>(target_s_values.size());
    if (n <= 0)
    {
        samples.clear();
        theta_body = 0.0;
        return;
    }

    std::vector<double> target_s_values_clamped(static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k < n; ++k)
    {
        target_s_values_clamped[static_cast<std::size_t>(k)] =
            std::max(0.0, std::min(target_s_values[static_cast<std::size_t>(k)], Lref));
    }
    samples.assign(static_cast<std::size_t>(n), MidlineSample());

    System& X_sys = equation_systems->get_system(ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& X_dof_map = X_sys.get_dof_map();

    // Strict midline: intersect boundary edges with the requested phi/s
    // isocontour, then select the two outer intersections by projection along
    // the reference section normal. Boundary intersections are computed as
    // continuous points on FE boundary edges.

    const double x_scan_end  = std::isfinite(boundary_x_end) ?
        boundary_x_end : ref_backbone_end_x;
    const double x_tol       = 1.0e-10 * std::max(1.0, ref_body_length);
    const bool   head_at_xmin = head_is_at_x_min();

    // Phase 1: collect boundary-edge Laplace geometry on this MPI rank
    struct BEdge {
        double s_a, x_ref_a, y_ref_a, x_def_a, y_def_a;
        double s_b, x_ref_b, y_ref_b, x_def_b, y_def_b;
    };
    std::vector<BEdge> local_edges;
    local_edges.reserve(512);

    std::set<std::pair<dof_id_type,dof_id_type>> visited_edges;
    std::vector<unsigned int> dof_idx_x, dof_idx_y;

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;

        for (unsigned int side_id = 0; side_id < elem->n_sides(); ++side_id)
        {
            if (elem->neighbor_ptr(side_id) != nullptr) continue;

            std::unique_ptr<const Elem> side_elem = elem->build_side_ptr(side_id);
            if (!side_elem || side_elem->n_nodes() < 2) continue;

            // Iterate consecutive node pairs on the boundary side
            const unsigned int nside = side_elem->n_nodes();
            for (unsigned int ni = 0; ni + 1 < nside; ++ni)
            {
                const Node* nA = side_elem->node_ptr(ni);
                const Node* nB = side_elem->node_ptr(ni + 1);
                if (!nA || !nB) continue;

                // Canonical ordering to deduplicate shared edges
                const Node* nodeA = (nA->id() <= nB->id()) ? nA : nB;
                const Node* nodeB = (nA->id() <= nB->id()) ? nB : nA;
                if (!visited_edges.insert({nodeA->id(), nodeB->id()}).second) continue;

                // Body-region pre-filter: skip edges entirely beyond x_scan_end
                const double xA_ref = (*nodeA)(0);
                const double xB_ref = (*nodeB)(0);
                const bool aIn = head_at_xmin ?
                    (xA_ref <= x_scan_end + x_tol) : (xA_ref >= x_scan_end - x_tol);
                const bool bIn = head_at_xmin ?
                    (xB_ref <= x_scan_end + x_tol) : (xB_ref >= x_scan_end - x_tol);
                if (!aIn && !bIn) continue;

                // Laplace geometry for node A
                double sA = 0.0;
                bool haveA = false;
                if (use_laplace_reference_parameterization)
                {
                    const auto it = ref_laplace_node_geom.find(nodeA->id());
                    if (it != ref_laplace_node_geom.end())
                    { sA = it->second.s; haveA = true; }
                    else
                        TBOX_ERROR("compute_current_midline_samples_at_s(): missing Laplace "
                                   "reference geometry for boundary node "
                                   << nodeA->id() << ".\n");
                }
                if (!haveA && !use_laplace_reference_parameterization)
                {
                    const ProjectionResult proj = project_to_reference_centerline(*nodeA);
                    if (!proj.valid) continue;
                    sA = proj.s; haveA = true;
                }

                // Laplace geometry for node B
                double sB = 0.0;
                bool haveB = false;
                if (use_laplace_reference_parameterization)
                {
                    const auto it = ref_laplace_node_geom.find(nodeB->id());
                    if (it != ref_laplace_node_geom.end())
                    { sB = it->second.s; haveB = true; }
                    else
                        TBOX_ERROR("compute_current_midline_samples_at_s(): missing Laplace "
                                   "reference geometry for boundary node "
                                   << nodeB->id() << ".\n");
                }
                if (!haveB && !use_laplace_reference_parameterization)
                {
                    const ProjectionResult proj = project_to_reference_centerline(*nodeB);
                    if (!proj.valid) continue;
                    sB = proj.s; haveB = true;
                }
                if (!haveA || !haveB) continue;

                // Deformed positions
                X_dof_map.dof_indices(nodeA, dof_idx_x, 0);
                X_dof_map.dof_indices(nodeA, dof_idx_y, 1);
                if (dof_idx_x.empty() || dof_idx_y.empty()) continue;
                const double xdefA = (*X_ghost_vec)(dof_idx_x[0]);
                const double ydefA = (*X_ghost_vec)(dof_idx_y[0]);

                X_dof_map.dof_indices(nodeB, dof_idx_x, 0);
                X_dof_map.dof_indices(nodeB, dof_idx_y, 1);
                if (dof_idx_x.empty() || dof_idx_y.empty()) continue;
                const double xdefB = (*X_ghost_vec)(dof_idx_x[0]);
                const double ydefB = (*X_ghost_vec)(dof_idx_y[0]);

                local_edges.push_back({sA, (*nodeA)(0), (*nodeA)(1), xdefA, ydefA,
                                       sB, (*nodeB)(0), (*nodeB)(1), xdefB, ydefB});
            }
        }
    }

    double local_s_min = std::numeric_limits<double>::max();
    double local_s_max = -std::numeric_limits<double>::max();
    for (const BEdge& edge : local_edges)
    {
        local_s_min = std::min(local_s_min, std::min(edge.s_a, edge.s_b));
        local_s_max = std::max(local_s_max, std::max(edge.s_a, edge.s_b));
    }
    double global_s_min = local_s_min;
    double global_s_max = local_s_max;
    IBTK_MPI::minReduction(&global_s_min, 1);
    IBTK_MPI::maxReduction(&global_s_max, 1);
    if (!(global_s_min <= global_s_max))
    {
        TBOX_ERROR("compute_current_midline_samples_at_s(): no boundary edges available "
                   "for strict iso-s midline extraction.\n");
    }

    std::vector<double> target_s_for_intersection = target_s_values_clamped;
    for (double& s_val : target_s_for_intersection)
    {
        s_val = std::min(std::max(s_val, global_s_min), global_s_max);
    }

    const double inf = std::numeric_limits<double>::max();
    std::vector<double> min_p(static_cast<std::size_t>(n),  inf);
    std::vector<double> max_p(static_cast<std::size_t>(n), -inf);
    std::vector<double> min_x_cur(static_cast<std::size_t>(n), 0.0);
    std::vector<double> min_y_cur(static_cast<std::size_t>(n), 0.0);
    std::vector<double> max_x_cur(static_cast<std::size_t>(n), 0.0);
    std::vector<double> max_y_cur(static_cast<std::size_t>(n), 0.0);
    std::vector<double> min_found(static_cast<std::size_t>(n), 0.0);
    std::vector<double> max_found(static_cast<std::size_t>(n), 0.0);

    // Phase 2: iso-s intersection per edge per station
    for (const BEdge& edge : local_edges)
    {
        const double ds = edge.s_b - edge.s_a;
        if (std::abs(ds) < 1.0e-14 * Lref) continue; // degenerate: same s at both ends

        const double s_lo = std::min(edge.s_a, edge.s_b);
        const double s_hi = std::max(edge.s_a, edge.s_b);

        for (int k = 0; k < n; ++k)
        {
            const double ts = target_s_for_intersection[static_cast<std::size_t>(k)];
            if (ts < s_lo || ts > s_hi) continue;

            const double t     = std::max(0.0, std::min(1.0, (ts - edge.s_a) / ds));
            const double x_ref_P = (1.0 - t) * edge.x_ref_a + t * edge.x_ref_b;
            const double y_ref_P = (1.0 - t) * edge.y_ref_a + t * edge.y_ref_b;
            const double x_cur_P = (1.0 - t) * edge.x_def_a + t * edge.x_def_b;
            const double y_cur_P = (1.0 - t) * edge.y_def_a + t * edge.y_def_b;
            const PhiIsoSectionSample section =
                interpolate_phi_section_sample(ts / Lref);
            const double p =
                x_ref_P * section.n_hat(0) + y_ref_P * section.n_hat(1);
            const std::size_t j = static_cast<std::size_t>(k);
            if (p < min_p[j])
            {
                min_p[j] = p;
                min_x_cur[j] = x_cur_P;
                min_y_cur[j] = y_cur_P;
                min_found[j] = 1.0;
            }
            if (p > max_p[j])
            {
                max_p[j] = p;
                max_x_cur[j] = x_cur_P;
                max_y_cur[j] = y_cur_P;
                max_found[j] = 1.0;
            }
        }
    }

    std::vector<double> local_min_p = min_p;
    std::vector<double> local_max_p = max_p;
    IBTK_MPI::minReduction(min_p.data(), n);
    IBTK_MPI::maxReduction(max_p.data(), n);

    const double p_tol = 1.0e-12 * std::max(1.0, ref_h_max);
    for (int k = 0; k < n; ++k)
    {
        const std::size_t j = static_cast<std::size_t>(k);
        if (min_found[j] < 0.5 || std::abs(local_min_p[j] - min_p[j]) > p_tol)
        {
            min_x_cur[j] = 0.0;
            min_y_cur[j] = 0.0;
            min_found[j] = 0.0;
        }
        if (max_found[j] < 0.5 || std::abs(local_max_p[j] - max_p[j]) > p_tol)
        {
            max_x_cur[j] = 0.0;
            max_y_cur[j] = 0.0;
            max_found[j] = 0.0;
        }
    }

    IBTK_MPI::sumReduction(min_x_cur.data(), n);
    IBTK_MPI::sumReduction(min_y_cur.data(), n);
    IBTK_MPI::sumReduction(max_x_cur.data(), n);
    IBTK_MPI::sumReduction(max_y_cur.data(), n);
    IBTK_MPI::sumReduction(min_found.data(), n);
    IBTK_MPI::sumReduction(max_found.data(), n);

    for (int k = 0; k < n; ++k)
    {
        const std::size_t j = static_cast<std::size_t>(k);
        if (min_found[j] < 0.5 || max_found[j] < 0.5)
        {
            TBOX_ERROR("compute_current_midline_samples_at_s(): no boundary-edge iso-s "
                       "intersection found at s/L = "
                       << target_s_values_clamped[static_cast<std::size_t>(k)] / Lref
                       << ". Check the "
                          "boundary mesh near this station.\n");
        }
    }

    std::vector<double> x_cur(static_cast<std::size_t>(n), 0.0);
    std::vector<double> y_cur(static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k < n; ++k)
    {
        const std::size_t j = static_cast<std::size_t>(k);
        const double inv_min = 1.0 / std::max(min_found[j], 1.0);
        const double inv_max = 1.0 / std::max(max_found[j], 1.0);
        x_cur[j] = 0.5 * (min_x_cur[j] * inv_min + max_x_cur[j] * inv_max);
        y_cur[j] = 0.5 * (min_y_cur[j] * inv_min + max_y_cur[j] * inv_max);
    }

    double rot_num = 0.0;
    double rot_den = 0.0;
    for (int k = 0; k < n; ++k)
    {
        const double s = target_s_values_clamped[static_cast<std::size_t>(k)];
        samples[static_cast<std::size_t>(k)].s = s;
        samples[static_cast<std::size_t>(k)].s_norm = s / Lref;
        samples[static_cast<std::size_t>(k)].ref = reference_frame_at_s(s);
        samples[static_cast<std::size_t>(k)].x_lab = x_cur[static_cast<std::size_t>(k)];
        samples[static_cast<std::size_t>(k)].y_lab = y_cur[static_cast<std::size_t>(k)];

        const double rx = samples[static_cast<std::size_t>(k)].ref.X(0) - s_reference_xcom;
        const double ry = samples[static_cast<std::size_t>(k)].ref.X(1) - s_reference_ycom;
        const double cx = x_cur[static_cast<std::size_t>(k)] - x_cm;
        const double cy = y_cur[static_cast<std::size_t>(k)] - y_cm;
        rot_num += rx * cy - ry * cx;
        rot_den += rx * cx + ry * cy;
    }
    theta_body = (std::abs(rot_num) + std::abs(rot_den) > 1.0e-30) ?
        std::atan2(rot_num, rot_den) : 0.0;

    const double ct = std::cos(theta_body);
    const double st = std::sin(theta_body);
    for (int k = 0; k < n; ++k)
    {
        MidlineSample& sample = samples[static_cast<std::size_t>(k)];
        const double ref_rx = sample.ref.X(0) - s_reference_xcom;
        const double ref_ry = sample.ref.X(1) - s_reference_ycom;
        const double cur_rx = sample.x_lab - x_cm;
        const double cur_ry = sample.y_lab - y_cm;

        const double ref_lab_x = ct * ref_rx - st * ref_ry;
        const double ref_lab_y = st * ref_rx + ct * ref_ry;
        const double delta_x = cur_rx - ref_lab_x;
        const double delta_y = cur_ry - ref_lab_y;

        const double n_lab_x = ct * sample.ref.n_hat(0) - st * sample.ref.n_hat(1);
        const double n_lab_y = st * sample.ref.n_hat(0) + ct * sample.ref.n_hat(1);
        sample.h = delta_x * n_lab_x + delta_y * n_lab_y;

        sample.x_body = ct * cur_rx + st * cur_ry;
        sample.y_body = -st * cur_rx + ct * cur_ry;
    }
}

static void
compute_current_midline_samples(Pointer<IBFEMethod>       ib_method_ops,
                                MeshBase&                 mesh,
                                EquationSystems*          equation_systems,
                                const int                 n_stations,
                                const double              x_cm,
                                const double              y_cm,
                                std::vector<MidlineSample>& samples,
                                double&                   theta_body)
{
    const int n = std::max(n_stations, 2);
    const double Lref = std::max(ref_arc_length, 1.0e-12);
    std::vector<double> target_s_values(static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k < n; ++k)
    {
        target_s_values[static_cast<std::size_t>(k)] =
            (n > 1 ? Lref * static_cast<double>(k) / static_cast<double>(n - 1) : 0.0);
    }
    compute_current_midline_samples_at_s(ib_method_ops, mesh, equation_systems,
                                         target_s_values, x_cm, y_cm,
                                         samples, theta_body);
}

std::vector<double> compute_body_midline_curvature(const std::vector<MidlineSample>& samples)
{
    const int n = static_cast<int>(samples.size());
    std::vector<double> kappa(static_cast<std::size_t>(n),
                              std::numeric_limits<double>::quiet_NaN());
    if (n < 3) return kappa;

    for (int i = 0; i < n; ++i)
    {
        const int i0 = (i == 0) ? 0 : (i == n - 1 ? n - 3 : i - 1);
        const int i1 = i0 + 1;
        const int i2 = i0 + 2;

        const double x0 = samples[static_cast<std::size_t>(i0)].x_body;
        const double y0 = samples[static_cast<std::size_t>(i0)].y_body;
        const double x1 = samples[static_cast<std::size_t>(i1)].x_body;
        const double y1 = samples[static_cast<std::size_t>(i1)].y_body;
        const double x2 = samples[static_cast<std::size_t>(i2)].x_body;
        const double y2 = samples[static_cast<std::size_t>(i2)].y_body;

        const double dx0 = x1 - x0;
        const double dy0 = y1 - y0;
        const double dx1 = x2 - x1;
        const double dy1 = y2 - y1;
        const double ds0 = std::sqrt(dx0 * dx0 + dy0 * dy0);
        const double ds1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
        if (ds0 <= 1.0e-14 || ds1 <= 1.0e-14) continue;

        const double t0x = dx0 / ds0;
        const double t0y = dy0 / ds0;
        const double t1x = dx1 / ds1;
        const double t1y = dy1 / ds1;
        const double cross = t0x * t1y - t0y * t1x;
        const double dot = t0x * t1x + t0y * t1y;
        const double dtheta = std::atan2(cross, dot);
        kappa[static_cast<std::size_t>(i)] = 2.0 * dtheta / (ds0 + ds1);
    }

    return kappa;
}

double interpolate_station_scalar(const std::vector<double>& s_values,
                                  const std::vector<double>& values,
                                  const double s_query)
{
    if (s_values.empty() || values.empty() || s_values.size() != values.size())
        return 0.0;
    if (s_query <= s_values.front()) return values.front();
    if (s_query >= s_values.back()) return values.back();

    const auto it = std::lower_bound(s_values.begin(), s_values.end(), s_query);
    const std::size_t i1 =
        static_cast<std::size_t>(std::distance(s_values.begin(), it));
    const std::size_t i0 = (i1 > 0) ? i1 - 1 : 0;
    const double s0 = s_values[i0];
    const double s1 = s_values[i1];
    const double a = (std::abs(s1 - s0) > 1.0e-14) ?
        (s_query - s0) / (s1 - s0) : 0.0;
    return (1.0 - a) * values[i0] + a * values[i1];
}

static void
smooth_station_scalar_field(std::vector<double>& values, const int passes)
{
    const int n = static_cast<int>(values.size());
    if (n < 3 || passes <= 0) return;

    std::vector<double> tmp(values.size(), 0.0);
    for (int pass = 0; pass < passes; ++pass)
    {
        tmp = values;
        values.front() = 0.75 * tmp.front() + 0.25 * tmp[1];
        for (int i = 1; i + 1 < n; ++i)
        {
            values[static_cast<std::size_t>(i)] =
                0.25 * tmp[static_cast<std::size_t>(i - 1)] +
                0.50 * tmp[static_cast<std::size_t>(i)] +
                0.25 * tmp[static_cast<std::size_t>(i + 1)];
        }
        values.back() = 0.25 * tmp[static_cast<std::size_t>(n - 2)] +
                        0.75 * tmp.back();
    }
}

static void
cap_station_scalar_field(std::vector<double>& values, const double cap_abs)
{
    if (!(cap_abs > 0.0) || !std::isfinite(cap_abs)) return;
    for (double& value : values)
    {
        if (!std::isfinite(value))
        {
            value = 0.0;
            continue;
        }
        value = std::max(-cap_abs, std::min(cap_abs, value));
    }
}

static void
write_geometry_sign_diagnostics()
{
    if (IBTK_MPI::getRank() != 0) return;

    std::ofstream out("geometry_sign_diag.csv");
    if (!out.is_open())
    {
        TBOX_WARNING("write_geometry_sign_diagnostics(): cannot open geometry_sign_diag.csv\n");
        return;
    }

    const int n = 41;
    const double Lref = std::max(ref_arc_length, 1.0e-12);
    const double eps = 1.0e-4 * Lref;
    std::vector<MidlineSample> samples(static_cast<std::size_t>(n));

    for (int k = 0; k < n; ++k)
    {
        const double s_norm = (n > 1) ?
            static_cast<double>(k) / static_cast<double>(n - 1) : 0.0;
        const double s = s_norm * Lref;
        const ReferenceFrame frame = reference_frame_at_s(s);
        const double eta_test = eps * std::sin(2.0 * M_PI * s_norm);

        MidlineSample& sample = samples[static_cast<std::size_t>(k)];
        sample.s = s;
        sample.s_norm = s_norm;
        sample.ref = frame;
        sample.x_body = frame.X(0) - s_reference_xcom + eta_test * frame.n_hat(0);
        sample.y_body = frame.X(1) - s_reference_ycom + eta_test * frame.n_hat(1);
        sample.h = eta_test;
    }

    const std::vector<double> kappa = compute_body_midline_curvature(samples);

    out << "s_norm,x_ref,y_ref,t_x,t_y,n_x,n_y,t_cross_n"
        << ",eta_test,eta_second_derivative,kappa_synthetic"
        << ",kappa_over_eta_second\n";
    out.setf(std::ios::scientific);
    out.precision(10);
    for (int k = 0; k < n; ++k)
    {
        const MidlineSample& sample = samples[static_cast<std::size_t>(k)];
        const double eta_test = sample.h;
        const double eta_second =
            -std::pow(2.0 * M_PI, 2.0) * eta_test / (Lref * Lref);
        const double ratio =
            (std::abs(eta_second) > 1.0e-30 && std::isfinite(kappa[static_cast<std::size_t>(k)])) ?
            kappa[static_cast<std::size_t>(k)] / eta_second :
            std::numeric_limits<double>::quiet_NaN();
        const double t_cross_n =
            sample.ref.t_hat(0) * sample.ref.n_hat(1) -
            sample.ref.t_hat(1) * sample.ref.n_hat(0);
        out << sample.s_norm
            << "," << sample.ref.X(0)
            << "," << sample.ref.X(1)
            << "," << sample.ref.t_hat(0)
            << "," << sample.ref.t_hat(1)
            << "," << sample.ref.n_hat(0)
            << "," << sample.ref.n_hat(1)
            << "," << t_cross_n
            << "," << eta_test
            << "," << eta_second
            << "," << kappa[static_cast<std::size_t>(k)]
            << "," << ratio
            << "\n";
    }
    out.flush();
    pout << "  geometry_sign_diag.csv written (expect t_cross_n ~= +1 and "
         << "kappa/eta_second ~= +1 away from nodes where eta_second=0)\n";
}

inline double phase_from_fourier_coeffs(const double c, const double s)
{
    if (c * c + s * s <= 1.0e-30)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::atan2(s, c);
}

inline double wrapped_phase_diff(const double a, const double b)
{
    if (!std::isfinite(a) || !std::isfinite(b))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

std::vector<double> unwrap_phase_series(const std::vector<double>& phase)
{
    std::vector<double> unwrapped = phase;
    for (std::size_t k = 1; k < unwrapped.size(); ++k)
    {
        if (!std::isfinite(unwrapped[k]) || !std::isfinite(unwrapped[k - 1])) continue;
        double delta = unwrapped[k] - unwrapped[k - 1];
        while (delta > M_PI)
        {
            unwrapped[k] -= 2.0 * M_PI;
            delta = unwrapped[k] - unwrapped[k - 1];
        }
        while (delta < -M_PI)
        {
            unwrapped[k] += 2.0 * M_PI;
            delta = unwrapped[k] - unwrapped[k - 1];
        }
    }
    return unwrapped;
}

double phase_slope_over_s_norm(const std::vector<MidlineSample>& samples,
                               const std::vector<double>& phase_unwrapped)
{
    double w_sum = 0.0;
    double x_sum = 0.0;
    double y_sum = 0.0;
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        if (!std::isfinite(phase_unwrapped[k])) continue;
        w_sum += 1.0;
        x_sum += samples[k].s_norm;
        y_sum += phase_unwrapped[k];
    }
    if (w_sum < 2.0) return std::numeric_limits<double>::quiet_NaN();

    const double x_bar = x_sum / w_sum;
    const double y_bar = y_sum / w_sum;
    double num = 0.0;
    double den = 0.0;
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        if (!std::isfinite(phase_unwrapped[k])) continue;
        const double dx = samples[k].s_norm - x_bar;
        const double dy = phase_unwrapped[k] - y_bar;
        num += dx * dy;
        den += dx * dx;
    }
    return den > 1.0e-30 ? num / den : std::numeric_limits<double>::quiet_NaN();
}

static void
write_curvature_phase_diagnostics(const int            iteration_num,
                                  const double         loop_time,
                                  Pointer<IBFEMethod>  ib_method_ops,
                                  MeshBase&            mesh,
                                  EquationSystems*     equation_systems,
                                  const double         x_cm,
                                  const double         y_cm)
{
    if (!s_curvature_phase_diag_enable) return;
    if (s_curvature_phase_diag_interval > 1 &&
        (iteration_num % s_curvature_phase_diag_interval != 0)) return;

    const int n = std::max(s_curvature_phase_diag_stations, 3);
    std::vector<MidlineSample> samples;
    double theta_body = 0.0;
    compute_current_midline_samples(ib_method_ops, mesh, equation_systems,
                                    n, x_cm, y_cm, samples, theta_body);
    std::vector<double> kappa_body = compute_body_midline_curvature(samples);

    // ── Body displacement amplitude envelope (per-station, per beat cycle) ──
    // Resize / initialise on first call or station-count change.
    if (s_y_body_cyc_max.size() != samples.size())
    {
        s_y_body_cyc_max.assign(samples.size(), -1.0e30);
        s_y_body_cyc_min.assign(samples.size(),  1.0e30);
        s_y_body_amp.assign(samples.size(), std::numeric_limits<double>::quiet_NaN());
        s_y_body_cyc_idx = -1;
    }
    // Detect start of a new integer beat cycle; save amplitude from the completed cycle.
    const int cyc_idx = (wave_frequency > 0.0) ?
        static_cast<int>(loop_time * wave_frequency) : -1;
    if (cyc_idx > s_y_body_cyc_idx && s_y_body_cyc_idx >= 0)
    {
        for (std::size_t k = 0; k < samples.size(); ++k)
        {
            const double span = s_y_body_cyc_max[k] - s_y_body_cyc_min[k];
            s_y_body_amp[k] = (span > 0.0) ? 0.5 * span :
                               std::numeric_limits<double>::quiet_NaN();
        }
        std::fill(s_y_body_cyc_max.begin(), s_y_body_cyc_max.end(), -1.0e30);
        std::fill(s_y_body_cyc_min.begin(), s_y_body_cyc_min.end(),  1.0e30);
    }
    s_y_body_cyc_idx = cyc_idx;
    // Update running min/max with current snapshot.
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        if (std::isfinite(samples[k].y_body))
        {
            s_y_body_cyc_max[k] = std::max(s_y_body_cyc_max[k], samples[k].y_body);
            s_y_body_cyc_min[k] = std::min(s_y_body_cyc_min[k], samples[k].y_body);
        }
    }

    const double start_time = std::isfinite(s_curvature_phase_diag_start_time) ?
        s_curvature_phase_diag_start_time : wave_ramp_time;
    const bool update_accum = loop_time + 1.0e-12 >= start_time;

    if (s_curvature_cos_accum.size() != samples.size())
    {
        s_curvature_cos_accum.assign(samples.size(), 0.0);
        s_curvature_sin_accum.assign(samples.size(), 0.0);
        s_activation_cos_accum.assign(samples.size(), 0.0);
        s_activation_sin_accum.assign(samples.size(), 0.0);
        s_prev_curvature_body.assign(samples.size(),
                                     std::numeric_limits<double>::quiet_NaN());
        s_curvature_phase_accum_time = 0.0;
        s_curvature_phase_last_time = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_positive_work_accum = 0.0;
        s_curvature_phase_signed_work_accum = 0.0;
        s_curvature_phase_effective_positive_work_accum = 0.0;
        s_curvature_phase_effective_signed_work_accum = 0.0;
        s_curvature_phase_last_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_last_positive_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_last_effective_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_last_effective_positive_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_samples = 0;
    }

    double signed_power = std::numeric_limits<double>::quiet_NaN();
    double positive_power = std::numeric_limits<double>::quiet_NaN();
    double effective_signed_power = std::numeric_limits<double>::quiet_NaN();
    double effective_positive_power = std::numeric_limits<double>::quiet_NaN();
    if (update_accum)
    {
        const double dt_eff = std::isfinite(s_curvature_phase_last_time) ?
            loop_time - s_curvature_phase_last_time : 0.0;
        const double c_t = std::cos(wave_omega * loop_time);
        const double s_t = std::sin(wave_omega * loop_time);

        if (dt_eff > 1.0e-12)
        {
            signed_power = 0.0;
            positive_power = 0.0;
            effective_signed_power = 0.0;
            effective_positive_power = 0.0;
            const double ds_station =
                samples.size() > 1 ? std::max(ref_arc_length, 1.0e-12) /
                                      static_cast<double>(samples.size() - 1) :
                                      std::max(ref_arc_length, 1.0e-12);

            for (std::size_t k = 0; k < samples.size(); ++k)
            {
                const double h_geo = body_halfthick_from_s(samples[k].s);
                const double Mm =
                    active_moment_value_from_sample(samples[k].s,
                                                    samples[k].s_norm,
                                                    h_geo,
                                                    loop_time);
                const double Mm_applied =
                    active_applied_moment_from_sample(Mm, samples[k].s);
                const double M_kappa =
                    active_curvature_moment_from_active_moment(Mm_applied);
                if (std::isfinite(kappa_body[k]))
                {
                    s_curvature_cos_accum[k] += kappa_body[k] * c_t * dt_eff;
                    s_curvature_sin_accum[k] += kappa_body[k] * s_t * dt_eff;
                }
                if (std::abs(M_kappa) > 1.0e-30)
                {
                    s_activation_cos_accum[k] += M_kappa * c_t * dt_eff;
                    s_activation_sin_accum[k] += M_kappa * s_t * dt_eff;
                }

                if (std::isfinite(kappa_body[k]) &&
                    std::isfinite(s_prev_curvature_body[k]))
                {
                    const double kappa_dot =
                        (kappa_body[k] - s_prev_curvature_body[k]) / dt_eff;
                    const double p_density = Mm * kappa_dot;
                    const double p_density_effective = M_kappa * kappa_dot;
                    signed_power += p_density * ds_station;
                    positive_power += std::max(p_density, 0.0) * ds_station;
                    effective_signed_power += p_density_effective * ds_station;
                    effective_positive_power +=
                        std::max(p_density_effective, 0.0) * ds_station;
                }
            }

            s_curvature_phase_accum_time += dt_eff;
            s_curvature_phase_signed_work_accum += signed_power * dt_eff;
            s_curvature_phase_positive_work_accum += positive_power * dt_eff;
            s_curvature_phase_effective_signed_work_accum +=
                effective_signed_power * dt_eff;
            s_curvature_phase_effective_positive_work_accum +=
                effective_positive_power * dt_eff;
            s_curvature_phase_last_power = signed_power;
            s_curvature_phase_last_positive_power = positive_power;
            s_curvature_phase_last_effective_power = effective_signed_power;
            s_curvature_phase_last_effective_positive_power =
                effective_positive_power;
            ++s_curvature_phase_samples;
        }

        s_curvature_phase_last_time = loop_time;
        s_prev_curvature_body = kappa_body;
    }

    if (IBTK_MPI::getRank() != 0) return;

    std::vector<double> curvature_phase(samples.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> activation_phase(samples.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> phase_lag(samples.size(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        curvature_phase[k] = phase_from_fourier_coeffs(s_curvature_cos_accum[k],
                                                       s_curvature_sin_accum[k]);
        activation_phase[k] = phase_from_fourier_coeffs(s_activation_cos_accum[k],
                                                        s_activation_sin_accum[k]);
        phase_lag[k] = wrapped_phase_diff(curvature_phase[k], activation_phase[k]);
    }
    const std::vector<double> curvature_phase_unwrapped =
        unwrap_phase_series(curvature_phase);
    const double curvature_phase_slope =
        phase_slope_over_s_norm(samples, curvature_phase_unwrapped);
    const double active_phase_slope_abs = active_phase_slope_abs_over_s_norm();
    const double active_phase_slope_expected = wave_time_sign * active_phase_slope_abs;
    const double signed_traveling_wave_index =
        (std::isfinite(curvature_phase_slope) &&
         std::isfinite(active_phase_slope_abs) &&
         active_phase_slope_abs > 1.0e-30) ?
        curvature_phase_slope / active_phase_slope_abs :
        std::numeric_limits<double>::quiet_NaN();
    const double traveling_wave_index =
        std::isfinite(signed_traveling_wave_index) ?
        std::min(std::abs(signed_traveling_wave_index), 1.0) :
        std::numeric_limits<double>::quiet_NaN();
    const double drive_following_index =
        (std::isfinite(curvature_phase_slope) &&
         std::isfinite(active_phase_slope_expected) &&
         std::abs(active_phase_slope_expected) > 1.0e-30) ?
        curvature_phase_slope / active_phase_slope_expected :
        std::numeric_limits<double>::quiet_NaN();

    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_curvature_phase_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_ERROR("write_curvature_phase_diagnostics(): cannot open "
                       << s_curvature_phase_diag_filename << "\n");
        }
        out << "step,time,cycle,phase,phase_samples,phase_start_time,phase_accum_time"
            << ",s_norm,x_ref_norm,x_lab_norm,y_lab_norm,y_prop_norm"
            << ",x_body_norm,y_body_norm,h_norm"
            << ",A_body_norm"
            << ",kappa_body,kappa_body_L,activation_drive,activation_drive_amp"
            << ",active_envelope,active_moment_prefactor"
            << ",active_moment,active_stress_section_moment,curvature_conjugate_moment"
            << ",active_moment_to_stress_sign"
            << ",curvature_phase,activation_phase,phase_lag"
            << ",curvature_phase_unwrapped,curvature_phase_slope"
            << ",active_phase_slope_abs,active_phase_slope_expected"
            << ",traveling_wave_index,signed_traveling_wave_index,drive_following_index"
            << ",theta_body,x_cm,y_cm,y_cm_relative"
            << ",raw_active_power,raw_active_positive_power"
            << ",raw_active_signed_work,raw_active_positive_work"
            << ",active_power,active_positive_power,active_signed_work,active_positive_work"
            << ",effective_active_power,effective_active_positive_power"
            << ",effective_active_signed_work,effective_active_positive_work"
            << "\n";
        out.flush();
    }

    const double Lref = std::max(ref_arc_length, 1.0e-12);
    out.setf(std::ios::scientific);
    out.precision(10);
    for (std::size_t k = 0; k < samples.size(); ++k)
    {
        const double h_geo = body_halfthick_from_s(samples[k].s);
        const double active_envelope =
            active_moment_envelope_from_s_norm(samples[k].s_norm);
        const double active_moment_prefactor =
            active_moment_prefactor_from_sample(samples[k].s_norm,
                                                h_geo,
                                                loop_time);
        const double active_moment =
            active_moment_value_from_sample(samples[k].s,
                                            samples[k].s_norm,
                                            h_geo,
                                            loop_time);
        const double active_moment_applied =
            active_applied_moment_from_sample(active_moment, samples[k].s);
        const double active_stress_section_moment =
            active_moment_to_stress_sign * active_moment_applied;
        const double curvature_conjugate_moment =
            active_curvature_moment_from_active_moment(active_moment_applied);
        out << iteration_num
            << "," << loop_time
            << "," << (wave_frequency > 0.0 ? loop_time * wave_frequency :
                        std::numeric_limits<double>::quiet_NaN())
            << "," << phase01(loop_time)
            << "," << s_curvature_phase_samples
            << "," << start_time
            << "," << s_curvature_phase_accum_time
            << "," << samples[k].s_norm
            << "," << reference_x_norm_from_s(samples[k].s)
            << "," << samples[k].x_lab / Lref
            << "," << samples[k].y_lab / Lref
            << "," << (samples[k].y_lab - s_reference_ycom) / Lref
            << "," << samples[k].x_body / Lref
            << "," << samples[k].y_body / Lref
            << "," << samples[k].h / Lref
            << "," << (s_y_body_amp.size() > k && std::isfinite(s_y_body_amp[k]) ?
                        s_y_body_amp[k] / Lref :
                        std::numeric_limits<double>::quiet_NaN())
            << "," << kappa_body[k]
            << "," << kappa_body[k] * Lref
            << "," << muscle_moment_drive_from_s(samples[k].s, loop_time)
            << "," << muscle_moment_drive_amplitude_from_s(samples[k].s)
            << "," << active_envelope
            << "," << active_moment_prefactor
            << "," << active_moment
            << "," << active_stress_section_moment
            << "," << curvature_conjugate_moment
            << "," << active_moment_to_stress_sign
            << "," << curvature_phase[k]
            << "," << activation_phase[k]
            << "," << phase_lag[k]
            << "," << curvature_phase_unwrapped[k]
            << "," << curvature_phase_slope
            << "," << active_phase_slope_abs
            << "," << active_phase_slope_expected
            << "," << traveling_wave_index
            << "," << signed_traveling_wave_index
            << "," << drive_following_index
            << "," << theta_body
            << "," << x_cm
            << "," << y_cm
            << "," << y_cm - s_reference_ycom
            << "," << s_curvature_phase_last_power
            << "," << s_curvature_phase_last_positive_power
            << "," << s_curvature_phase_signed_work_accum
            << "," << s_curvature_phase_positive_work_accum
            << "," << s_curvature_phase_last_effective_power
            << "," << s_curvature_phase_last_effective_positive_power
            << "," << s_curvature_phase_effective_signed_work_accum
            << "," << s_curvature_phase_effective_positive_work_accum
            << "," << s_curvature_phase_last_effective_power
            << "," << s_curvature_phase_last_effective_positive_power
            << "," << s_curvature_phase_effective_signed_work_accum
            << "," << s_curvature_phase_effective_positive_work_accum
            << "\n";
    }
    out.flush();
}

// =========================================================================
// Force decomposition diagnostic
//
// The net weak load from an internal stress field can be small even when that
// stress is dynamically important. Therefore this diagnostic reports both:
//   1) weak-form resultant loads for sign checks, and
//   2) local weak-load L1 measures and stress power PP:dF/dt for dominance.
// =========================================================================
struct ForceDecompAccumulator
{
    double Fx = 0.0;
    double Fy = 0.0;
    double L1 = 0.0;
    double forward_abs = 0.0;
    double lateral_abs = 0.0;
    double P = 0.0;
    double P_pos = 0.0;
    double P_neg = 0.0;
    double P_abs = 0.0;
    double P_weak = 0.0;
    double P_weak_abs = 0.0;
};

enum ForceDecompComponent
{
    FORCE_DEV = 0,
    FORCE_DIL = 1,
    FORCE_DAMPING = 2,
    FORCE_ACTIVE = 3,
    FORCE_SUM = 4,
    FORCE_N_COMPONENTS = 5
};

inline const char*
force_decomp_component_name(const int c)
{
    switch (c)
    {
    case FORCE_DEV:        return "dev";
    case FORCE_DIL:        return "dil";
    case FORCE_DAMPING:    return "damping";
    case FORCE_ACTIVE:     return "active";
    case FORCE_SUM:        return "sum";
    default:               return "unknown";
    }
}

inline double
section_moment_bin_width(const int i, const int nb)
{
    if (nb <= 1) return std::max(ref_arc_length, 1.0e-12);
    const double left = (i == 0) ?
        0.0 : (static_cast<double>(i) - 0.5) / static_cast<double>(nb - 1);
    const double right = (i == nb - 1) ?
        1.0 : (static_cast<double>(i) + 0.5) / static_cast<double>(nb - 1);
    return std::max((right - left) * std::max(ref_arc_length, 1.0e-12),
                    1.0e-12);
}

inline double
axial_fiber_stress_from_PK1(const TensorValue<double>& PP,
                            const TensorValue<double>& FF,
                            const VectorValue<double>& f0)
{
    const VectorValue<double> a = FF * f0;
    const double a2 = std::max(a * a, 0.0);
    if (a2 <= 1.0e-24) return 0.0;
    return (a * (PP * f0)) / a2;
}

struct SectionMomentBin
{
    double area = 0.0;
    double s_sum = 0.0;
    double x_sum = 0.0;
    double h_sum = 0.0;
    double c1_sum = 0.0;
    double N[FORCE_N_COMPONENTS] = {};
    double M[FORCE_N_COMPONENTS] = {};
};

static void
write_section_moment_decomposition_diagnostics(
    const int            iteration_num,
    const double         loop_time,
    Pointer<IBFEMethod>  ib_method_ops,
    MeshBase&            mesh,
    EquationSystems*     equation_systems)
{
    if (!s_section_moment_diag_enable) return;
    if (s_section_moment_diag_interval > 1 &&
        (iteration_num % s_section_moment_diag_interval != 0)) return;
    if (!equation_systems) return;

    const unsigned int dim = mesh.mesh_dimension();
    const int nb = std::max(3, (s_section_moment_diag_bins > 0) ?
                            s_section_moment_diag_bins : reference_profile_bins);
    std::vector<SectionMomentBin> bins(static_cast<std::size_t>(nb));

    System& X_sys = equation_systems->get_system(
        ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec       = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);

    System& U_sys = equation_systems->get_system(
        ib_method_ops->getVelocitySystemName());
    NumericVector<double>* U_vec       = U_sys.solution.get();
    NumericVector<double>* U_ghost_vec = U_sys.current_local_solution.get();
    U_vec->close();
    copy_and_synch(*U_vec, *U_ghost_vec);

    System& ref_geom_sys = equation_systems->get_system(REF_GEOM_SYSTEM_NAME);
    NumericVector<double>* ref_geom_vec = ref_geom_sys.solution.get();
    NumericVector<double>* ref_geom_ghost_vec =
        ref_geom_sys.current_local_solution.get();
    ref_geom_vec->close();
    copy_and_synch(*ref_geom_vec, *ref_geom_ghost_vec);

    const DofMap& X_dof_map = X_sys.get_dof_map();
    const DofMap& U_dof_map = U_sys.get_dof_map();
    const DofMap& ref_geom_dof_map = ref_geom_sys.get_dof_map();
    const FEType fe_type = X_dof_map.variable_type(0);
    const libMesh::Order quad_order =
        Utility::string_to_enum<libMesh::Order>(s_section_moment_quad_order);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule(new QGauss(dim, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<double> >& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();
    std::vector<std::vector<unsigned int> > X_dof_indices(NDIM);
    std::vector<std::vector<unsigned int> > U_dof_indices(NDIM);
    std::vector<std::vector<unsigned int> > ref_geom_dof_indices(REF_GEOM_N_VARS);
    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;

        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            X_dof_map.dof_indices(elem, X_dof_indices[d], d);
            U_dof_map.dof_indices(elem, U_dof_indices[d], d);
        }
        for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v)
            ref_geom_dof_map.dof_indices(elem, ref_geom_dof_indices[v], v);
        get_values_for_interpolation(X_node, *X_ghost_vec, X_dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, U_dof_indices);

        const unsigned int n_nodes = elem->n_nodes();
        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            TensorValue<double> FF;
            jacobian(FF, qp, X_node, dphi);

            TensorValue<double> F_dot;
            F_dot.zero();
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                for (unsigned int i = 0; i < NDIM; ++i)
                {
                    const double u_ki = U_node[k][i];
                    for (unsigned int j = 0; j < NDIM; ++j)
                    {
                        F_dot(i, j) += u_ki * dphi[k][qp](j);
                    }
                }
            }

            libMesh::Point X_ref_qp(0.0, 0.0, 0.0);
            double ref_geom_values[REF_GEOM_N_VARS] = { 0.0, 0.0, 0.0, 0.0 };
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                    X_ref_qp(d) += phi[k][qp] * elem->point(k)(d);
                for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v)
                    ref_geom_values[v] +=
                        phi[k][qp] * (*ref_geom_ghost_vec)(ref_geom_dof_indices[v][k]);
            }

            ReferenceGeometrySample ref_geom;
            ref_geom.s = std::max(0.0, std::min(ref_geom_values[REF_GEOM_S],
                                                ref_arc_length));
            ref_geom.eta = ref_geom_values[REF_GEOM_ETA];
            const VectorValue<double> t_raw(ref_geom_values[REF_GEOM_T_X],
                                            ref_geom_values[REF_GEOM_T_Y]);
            const double t_norm = std::sqrt(std::max(t_raw * t_raw, 0.0));
            if (t_norm <= 1.0e-14) continue;
            ref_geom.t_hat = t_raw / t_norm;

            TensorValue<double> PP[FORCE_N_COMPONENTS];
            for (int c = 0; c < FORCE_N_COMPONENTS; ++c) PP[c] = 0.0;
            const double c1_local = get_c1_s_passive_local_from_reference_point(X_ref_qp);
            compute_passive_dev_PK1_stress_impl(PP[FORCE_DEV], FF, X_ref_qp);
            compute_passive_dil_PK1_stress_impl(PP[FORCE_DIL], FF, X_ref_qp);
            compute_continuum_damping_PK1_stress_impl(PP[FORCE_DAMPING],
                                                      FF, X_ref_qp, ref_geom,
                                                      F_dot);
            compute_active_PK1_stress_impl(PP[FORCE_ACTIVE], FF, X_ref_qp,
                                           ref_geom, loop_time);
            PP[FORCE_SUM] = PP[FORCE_DEV] + PP[FORCE_DIL] +
                            PP[FORCE_DAMPING] + PP[FORCE_ACTIVE];

            const double s_norm = clamp01(ref_geom.s /
                                          std::max(ref_arc_length, 1.0e-12));
            const int bi = std::min(nb - 1, std::max(0,
                static_cast<int>(s_norm * static_cast<double>(nb - 1) + 0.5)));
            SectionMomentBin& bin = bins[static_cast<std::size_t>(bi)];
            const double jw = JxW[qp];
            const double h = body_halfthick_from_s(ref_geom.s);
            bin.area += jw;
            bin.s_sum += ref_geom.s * jw;
            bin.x_sum += X_ref_qp(0) * jw;
            bin.h_sum += h * jw;
            bin.c1_sum += c1_local * jw;

            for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
            {
                const double T_tt =
                    axial_fiber_stress_from_PK1(PP[c], FF, ref_geom.t_hat);
                bin.N[c] += T_tt * jw;
                bin.M[c] += -T_tt * ref_geom.eta * jw;
            }
        }
    }

    const int n_values_per_bin = 5 + 2 * FORCE_N_COMPONENTS;
    std::vector<double> reduced(static_cast<std::size_t>(nb * n_values_per_bin), 0.0);
    std::size_t r = 0;
    for (int i = 0; i < nb; ++i)
    {
        const SectionMomentBin& bin = bins[static_cast<std::size_t>(i)];
        reduced[r++] = bin.area;
        reduced[r++] = bin.s_sum;
        reduced[r++] = bin.x_sum;
        reduced[r++] = bin.h_sum;
        reduced[r++] = bin.c1_sum;
        for (int c = 0; c < FORCE_N_COMPONENTS; ++c) reduced[r++] = bin.N[c];
        for (int c = 0; c < FORCE_N_COMPONENTS; ++c) reduced[r++] = bin.M[c];
    }
    IBTK_MPI::sumReduction(reduced.data(), static_cast<int>(reduced.size()));

    r = 0;
    for (int i = 0; i < nb; ++i)
    {
        SectionMomentBin& bin = bins[static_cast<std::size_t>(i)];
        bin.area = reduced[r++];
        bin.s_sum = reduced[r++];
        bin.x_sum = reduced[r++];
        bin.h_sum = reduced[r++];
        bin.c1_sum = reduced[r++];
        for (int c = 0; c < FORCE_N_COMPONENTS; ++c) bin.N[c] = reduced[r++];
        for (int c = 0; c < FORCE_N_COMPONENTS; ++c) bin.M[c] = reduced[r++];
    }

    if (IBTK_MPI::getRank() != 0) return;
    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_section_moment_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_section_moment_decomposition_diagnostics(): cannot open "
                         << s_section_moment_diag_filename << "\n");
            return;
        }
        out << "step,time,bin,s_norm,s_mean,x_ref_mean,h_mean,c1_mean,area,ds_bin";
        for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
            out << ",N_" << force_decomp_component_name(c);
        for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
            out << ",M_" << force_decomp_component_name(c);
        out << ",M_passive,M_resist,M_active_model,M_active_error,R_active_resist,R_mag,C_oppose,R_internal_moment_residual\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    for (int i = 0; i < nb; ++i)
    {
        const SectionMomentBin& bin = bins[static_cast<std::size_t>(i)];
        const double ds_bin = section_moment_bin_width(i, nb);
        const double s_norm_center = (nb > 1) ?
            static_cast<double>(i) / static_cast<double>(nb - 1) : 0.0;
        const double inv_area = (bin.area > 1.0e-30) ? 1.0 / bin.area : 0.0;
        const double s_mean = (bin.area > 1.0e-30) ?
            bin.s_sum * inv_area : s_norm_center * std::max(ref_arc_length, 1.0e-12);
        const double x_ref_mean = (bin.area > 1.0e-30) ?
            bin.x_sum * inv_area : std::numeric_limits<double>::quiet_NaN();
        const double h_mean = (bin.area > 1.0e-30) ?
            bin.h_sum * inv_area : body_halfthick_from_s(s_mean);
        const double c1_mean = (bin.area > 1.0e-30) ?
            bin.c1_sum * inv_area : std::numeric_limits<double>::quiet_NaN();

        double N_sec[FORCE_N_COMPONENTS];
        double M_sec[FORCE_N_COMPONENTS];
        for (int c = 0; c < FORCE_N_COMPONENTS; ++c)
        {
            N_sec[c] = bin.N[c] / ds_bin;
            M_sec[c] = bin.M[c] / ds_bin;
        }
        const double s_model = std::max(0.0, std::min(s_mean, ref_arc_length));
        const double s_model_norm = clamp01(s_model / std::max(ref_arc_length, 1.0e-12));
        const double h_model = body_halfthick_from_s(s_model);
        const double Mm_model =
            active_moment_value_from_sample(s_model, s_model_norm, h_model, loop_time);
        const double Mm_model_applied =
            active_applied_moment_from_sample(Mm_model, s_model);
        const double M_active_model =
            active_curvature_moment_from_active_moment(Mm_model_applied);
        const double M_passive = M_sec[FORCE_DEV] + M_sec[FORCE_DIL];
        const double M_resist = M_passive + M_sec[FORCE_DAMPING];
        const double R_active_resist =
            M_sec[FORCE_ACTIVE] / (std::abs(M_resist) + 1.0e-30);
        const double Ma_abs = std::abs(M_sec[FORCE_ACTIVE]);
        const double Mr_abs = std::abs(M_resist);
        const double R_mag     = Mr_abs / (Ma_abs + 1.0e-30);
        const double C_oppose  = -(M_sec[FORCE_ACTIVE] * M_resist) /
                                  (Ma_abs * Mr_abs + 1.0e-30);
        const double R_internal_moment_residual =
            std::abs(M_sec[FORCE_ACTIVE] + M_resist) /
            (Ma_abs + Mr_abs + 1.0e-30);

        out << iteration_num
            << "," << loop_time
            << "," << i
            << "," << s_norm_center
            << "," << s_mean
            << "," << x_ref_mean
            << "," << h_mean
            << "," << c1_mean
            << "," << bin.area
            << "," << ds_bin;
        for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
            out << "," << N_sec[c];
        for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
            out << "," << M_sec[c];
        out << "," << M_passive
            << "," << M_resist
            << "," << M_active_model
            << "," << (M_sec[FORCE_ACTIVE] - M_active_model)
            << "," << R_active_resist
            << "," << R_mag
            << "," << C_oppose
            << "," << R_internal_moment_residual
            << "\n";
    }
    out.flush();
}

static void
write_force_decomposition_diagnostics(const int            iteration_num,
                                      const double         loop_time,
                                      Pointer<IBFEMethod>  ib_method_ops,
                                      MeshBase&            mesh,
                                      EquationSystems*     equation_systems,
                                      const double         F_integral[NDIM],
                                      const double         F_power_on_fluid,
                                      const double         x_cm,
                                      const double         y_cm,
                                      const double         vcm_x,
                                      const double         vcm_y)
{
    if (!s_force_decomp_diag_enable) return;
    if (s_force_decomp_diag_interval > 1 &&
        (iteration_num % s_force_decomp_diag_interval != 0)) return;
    if (!equation_systems) return;

    const double forward_sign = head_is_at_x_min() ? -1.0 : 1.0;
    const unsigned int dim = mesh.mesh_dimension();

    System& X_sys = equation_systems->get_system(
        ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec       = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);

    System& U_sys = equation_systems->get_system(
        ib_method_ops->getVelocitySystemName());
    NumericVector<double>* U_vec       = U_sys.solution.get();
    NumericVector<double>* U_ghost_vec = U_sys.current_local_solution.get();
    U_vec->close();
    copy_and_synch(*U_vec, *U_ghost_vec);

    System& ref_geom_sys = equation_systems->get_system(REF_GEOM_SYSTEM_NAME);
    NumericVector<double>* ref_geom_vec = ref_geom_sys.solution.get();
    NumericVector<double>* ref_geom_ghost_vec =
        ref_geom_sys.current_local_solution.get();
    ref_geom_vec->close();
    copy_and_synch(*ref_geom_vec, *ref_geom_ghost_vec);

    const DofMap& X_dof_map = X_sys.get_dof_map();
    const DofMap& U_dof_map = U_sys.get_dof_map();
    const DofMap& ref_geom_dof_map = ref_geom_sys.get_dof_map();
    const FEType fe_type = X_dof_map.variable_type(0);
    const libMesh::Order quad_order =
        Utility::string_to_enum<libMesh::Order>(s_force_decomp_quad_order);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    std::unique_ptr<QBase> qrule(new QGauss(dim, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<double>& JxW = fe->get_JxW();
    const std::vector<std::vector<double> >& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();
    std::vector<std::vector<unsigned int> > X_dof_indices(NDIM);
    std::vector<std::vector<unsigned int> > U_dof_indices(NDIM);
    std::vector<std::vector<unsigned int> > ref_geom_dof_indices(REF_GEOM_N_VARS);
    boost::multi_array<double, 2> X_node;
    boost::multi_array<double, 2> U_node;
    std::vector<ForceDecompAccumulator> acc(FORCE_N_COMPONENTS);

    auto accumulate_qp =
        [&](const int c,
            const TensorValue<double>& PP,
            const TensorValue<double>& F_dot,
            const double JxW_qp,
            const std::vector<RealGradient>& dphi_qp,
            std::vector<std::vector<VectorValue<double> > >& elem_loads)
        {
            double p = 0.0;
            for (unsigned int i = 0; i < NDIM; ++i)
                for (unsigned int j = 0; j < NDIM; ++j)
                    p += PP(i, j) * F_dot(i, j);
            p *= JxW_qp;

            acc[c].P += p;
            acc[c].P_pos += std::max(p, 0.0);
            acc[c].P_neg += std::min(p, 0.0);
            acc[c].P_abs += std::abs(p);
            acc[FORCE_SUM].P += p;
            acc[FORCE_SUM].P_pos += std::max(p, 0.0);
            acc[FORCE_SUM].P_neg += std::min(p, 0.0);
            acc[FORCE_SUM].P_abs += std::abs(p);

            for (std::size_t k = 0; k < dphi_qp.size(); ++k)
            {
                VectorValue<double> f_node(0.0, 0.0);
                for (unsigned int i = 0; i < NDIM; ++i)
                {
                    double f_i = 0.0;
                    for (unsigned int j = 0; j < NDIM; ++j)
                        f_i -= PP(i, j) * dphi_qp[k](j) * JxW_qp;
                    f_node(i) = f_i;
                }
                elem_loads[c][k] += f_node;
                elem_loads[FORCE_SUM][k] += f_node;
            }
        };

    for (auto el_it = mesh.active_local_elements_begin();
         el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        if (!elem) continue;

        fe->reinit(elem);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            X_dof_map.dof_indices(elem, X_dof_indices[d], d);
            U_dof_map.dof_indices(elem, U_dof_indices[d], d);
        }
        for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v)
            ref_geom_dof_map.dof_indices(elem, ref_geom_dof_indices[v], v);
        get_values_for_interpolation(X_node, *X_ghost_vec, X_dof_indices);
        get_values_for_interpolation(U_node, *U_ghost_vec, U_dof_indices);

        const unsigned int n_nodes = elem->n_nodes();
        std::vector<std::vector<VectorValue<double> > > elem_loads(FORCE_N_COMPONENTS);
        for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
            elem_loads[c].assign(n_nodes, VectorValue<double>(0.0, 0.0));

        for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
        {
            TensorValue<double> FF;
            jacobian(FF, qp, X_node, dphi);

            TensorValue<double> F_dot;
            F_dot.zero();
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                for (unsigned int i = 0; i < NDIM; ++i)
                {
                    const double u_ki = U_node[k][i];
                    for (unsigned int j = 0; j < NDIM; ++j)
                    {
                        F_dot(i, j) += u_ki * dphi[k][qp](j);
                    }
                }
            }

            libMesh::Point X_ref_qp(0.0, 0.0, 0.0);
            double ref_geom_values[REF_GEOM_N_VARS] = { 0.0, 0.0, 0.0, 0.0 };
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                    X_ref_qp(d) += phi[k][qp] * elem->point(k)(d);
                for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v)
                    ref_geom_values[v] +=
                        phi[k][qp] * (*ref_geom_ghost_vec)(ref_geom_dof_indices[v][k]);
            }

            ReferenceGeometrySample ref_geom;
            ref_geom.s = std::max(0.0, std::min(ref_geom_values[REF_GEOM_S],
                                                ref_arc_length));
            ref_geom.eta = ref_geom_values[REF_GEOM_ETA];
            const VectorValue<double> t_raw(ref_geom_values[REF_GEOM_T_X],
                                            ref_geom_values[REF_GEOM_T_Y]);
            const double t_norm = std::sqrt(std::max(t_raw * t_raw, 0.0));
            if (t_norm <= 1.0e-14) continue;
            ref_geom.t_hat = t_raw / t_norm;

            TensorValue<double> PP_dev(0.0), PP_dil(0.0);
            TensorValue<double> PP_damping(0.0), PP_active(0.0);
            compute_passive_dev_PK1_stress_impl(PP_dev, FF, X_ref_qp);
            compute_passive_dil_PK1_stress_impl(PP_dil, FF, X_ref_qp);
            compute_continuum_damping_PK1_stress_impl(PP_damping, FF,
                                                      X_ref_qp, ref_geom,
                                                      F_dot);
            compute_active_PK1_stress_impl(PP_active, FF, X_ref_qp, ref_geom,
                                           loop_time);

            std::vector<RealGradient> dphi_qp(n_nodes);
            for (unsigned int k = 0; k < n_nodes; ++k) dphi_qp[k] = dphi[k][qp];
            accumulate_qp(FORCE_DEV, PP_dev, F_dot, JxW[qp], dphi_qp, elem_loads);
            accumulate_qp(FORCE_DIL, PP_dil, F_dot, JxW[qp], dphi_qp, elem_loads);
            accumulate_qp(FORCE_DAMPING, PP_damping, F_dot, JxW[qp], dphi_qp, elem_loads);
            accumulate_qp(FORCE_ACTIVE, PP_active, F_dot, JxW[qp], dphi_qp, elem_loads);
        }

        for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
        {
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                const VectorValue<double>& f = elem_loads[c][k];
                const VectorValue<double> u(U_node[k][0], U_node[k][1]);
                const double p_weak = f * u;
                acc[c].Fx += f(0);
                acc[c].Fy += f(1);
                acc[c].L1 += std::sqrt(std::max(f * f, 0.0));
                acc[c].forward_abs += std::abs(forward_sign * f(0));
                acc[c].lateral_abs += std::abs(forward_sign * f(1));
                acc[c].P_weak += p_weak;
                acc[c].P_weak_abs += std::abs(p_weak);
            }
        }
    }

    std::vector<double> reduced;
    reduced.reserve(FORCE_N_COMPONENTS * 11);
    for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
    {
        reduced.push_back(acc[c].Fx);
        reduced.push_back(acc[c].Fy);
        reduced.push_back(acc[c].L1);
        reduced.push_back(acc[c].forward_abs);
        reduced.push_back(acc[c].lateral_abs);
        reduced.push_back(acc[c].P);
        reduced.push_back(acc[c].P_pos);
        reduced.push_back(acc[c].P_neg);
        reduced.push_back(acc[c].P_abs);
        reduced.push_back(acc[c].P_weak);
        reduced.push_back(acc[c].P_weak_abs);
    }
    IBTK_MPI::sumReduction(reduced.data(), static_cast<int>(reduced.size()));

    std::size_t r = 0;
    for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
    {
        acc[c].Fx          = reduced[r++];
        acc[c].Fy          = reduced[r++];
        acc[c].L1          = reduced[r++];
        acc[c].forward_abs = reduced[r++];
        acc[c].lateral_abs = reduced[r++];
        acc[c].P           = reduced[r++];
        acc[c].P_pos       = reduced[r++];
        acc[c].P_neg       = reduced[r++];
        acc[c].P_abs       = reduced[r++];
        acc[c].P_weak      = reduced[r++];
        acc[c].P_weak_abs  = reduced[r++];
    }

    const double dt_eff = std::isfinite(s_force_decomp_prev_time) ?
        loop_time - s_force_decomp_prev_time : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(dt_eff) && dt_eff > 1.0e-12)
    {
        s_force_decomp_work_dev     += acc[FORCE_DEV].P * dt_eff;
        s_force_decomp_work_dil     += acc[FORCE_DIL].P * dt_eff;
        s_force_decomp_work_damping += acc[FORCE_DAMPING].P * dt_eff;
        s_force_decomp_work_active  += acc[FORCE_ACTIVE].P * dt_eff;
        s_force_decomp_work_sum     += acc[FORCE_SUM].P * dt_eff;
        s_force_decomp_work_weak_dev     += acc[FORCE_DEV].P_weak * dt_eff;
        s_force_decomp_work_weak_dil     += acc[FORCE_DIL].P_weak * dt_eff;
        s_force_decomp_work_weak_damping += acc[FORCE_DAMPING].P_weak * dt_eff;
        s_force_decomp_work_weak_active  += acc[FORCE_ACTIVE].P_weak * dt_eff;
        s_force_decomp_work_weak_sum     += acc[FORCE_SUM].P_weak * dt_eff;
        if (std::isfinite(F_power_on_fluid))
        {
            s_force_decomp_work_ib_on_fluid += F_power_on_fluid * dt_eff;
        }
    }
    s_force_decomp_prev_time = loop_time;

    int dominant_L1 = FORCE_DEV;
    int dominant_Pabs = FORCE_DEV;
    for (int c = FORCE_DIL; c <= FORCE_ACTIVE; ++c)
    {
        if (acc[c].L1 > acc[dominant_L1].L1) dominant_L1 = c;
        if (acc[c].P_abs > acc[dominant_Pabs].P_abs) dominant_Pabs = c;
    }

    if (IBTK_MPI::getRank() != 0) return;
    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_force_decomp_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_force_decomposition_diagnostics(): cannot open "
                         << s_force_decomp_diag_filename << "\n");
            return;
        }
        out << "step,time,dt_eff,forward_sign,x_cm,y_cm,vcm_x,vcm_y,v_forward"
            << ",F_total_x,F_total_y,F_total_forward_on_fluid,F_total_forward_on_fish"
            << ",P_IB_on_fluid,P_IB_on_fish";
        for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
        {
            const char* name = force_decomp_component_name(c);
            out << ",F_weak_" << name << "_x"
                << ",F_weak_" << name << "_y"
                << ",F_weak_" << name << "_forward_on_fluid"
                << ",F_weak_" << name << "_forward_on_fish"
                << ",F_weak_L1_" << name
                << ",F_weak_forward_abs_" << name
                << ",F_weak_lateral_abs_" << name
                << ",P_" << name
                << ",P_pos_" << name
                << ",P_neg_" << name
                << ",P_abs_" << name
                << ",P_weak_on_fluid_" << name
                << ",P_weak_on_fish_" << name
                << ",P_weak_abs_" << name;
        }
        out << ",F_weak_sum_error_x,F_weak_sum_error_y"
            << ",W_dev,W_dil,W_damping,W_active,W_sum"
            << ",W_weak_on_fluid_dev,W_weak_on_fluid_dil"
            << ",W_weak_on_fluid_damping,W_weak_on_fluid_active"
            << ",W_weak_on_fluid_sum,W_IB_on_fluid,W_IB_on_fish"
            << ",dominant_L1_component,dominant_Pabs_component\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    out << iteration_num
        << "," << loop_time
        << "," << dt_eff
        << "," << forward_sign
        << "," << x_cm
        << "," << y_cm
        << "," << vcm_x
        << "," << vcm_y
        << "," << forward_sign * vcm_x
        << "," << F_integral[0]
        << "," << F_integral[1]
        << "," << forward_sign * F_integral[0]
        << "," << -forward_sign * F_integral[0]
        << "," << F_power_on_fluid
        << "," << -F_power_on_fluid;
    for (int c = FORCE_DEV; c <= FORCE_SUM; ++c)
    {
        out << "," << acc[c].Fx
            << "," << acc[c].Fy
            << "," << forward_sign * acc[c].Fx
            << "," << -forward_sign * acc[c].Fx
            << "," << acc[c].L1
            << "," << acc[c].forward_abs
            << "," << acc[c].lateral_abs
            << "," << acc[c].P
            << "," << acc[c].P_pos
            << "," << acc[c].P_neg
            << "," << acc[c].P_abs
            << "," << acc[c].P_weak
            << "," << -acc[c].P_weak
            << "," << acc[c].P_weak_abs;
    }
    out << "," << (acc[FORCE_SUM].Fx - F_integral[0])
        << "," << (acc[FORCE_SUM].Fy - F_integral[1])
        << "," << s_force_decomp_work_dev
        << "," << s_force_decomp_work_dil
        << "," << s_force_decomp_work_damping
        << "," << s_force_decomp_work_active
        << "," << s_force_decomp_work_sum
        << "," << s_force_decomp_work_weak_dev
        << "," << s_force_decomp_work_weak_dil
        << "," << s_force_decomp_work_weak_damping
        << "," << s_force_decomp_work_weak_active
        << "," << s_force_decomp_work_weak_sum
        << "," << s_force_decomp_work_ib_on_fluid
        << "," << -s_force_decomp_work_ib_on_fluid
        << "," << force_decomp_component_name(dominant_L1)
        << "," << force_decomp_component_name(dominant_Pabs)
        << "\n";
    out.flush();
}

// =========================================================================
// Fin pitch diagnostic
//
// Tracks:
//   tail tip center (x_tip, y_tip) from ref_tail_upper/lower_node_id
//   fin root center (x_root, y_root) from ref_fin_root_upper/lower_node_id
//   theta_fin_lab = atan2(y_tip - y_root, x_tip - x_root)
//   v_y_tip, v_y_root = finite-difference lateral velocities
//   act_phase_root = active muscle phase at fin root location
// =========================================================================
static void
write_fin_pitch_diagnostics(const int            iteration_num,
                             const double         loop_time,
                             Pointer<IBFEMethod>  ib_method_ops,
                             MeshBase&            mesh,
                             EquationSystems*     equation_systems,
                             const double         x_cm,
                             const double         y_cm)
{
    if (!s_fin_pitch_diag_enable) return;
    if (s_fin_pitch_diag_interval > 1 &&
        (iteration_num % s_fin_pitch_diag_interval != 0)) return;
    if (!equation_systems) return;

    // Load the coordinates system.
    System& X_sys = equation_systems->get_system(
        ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);
    const DofMap& X_dof_map = X_sys.get_dof_map();

    // Get current tip center (average of upper and lower tail nodes).
    double x_tip_u = 0.0, y_tip_u = 0.0, x_tip_l = 0.0, y_tip_l = 0.0;
    const bool have_tip_u = get_current_node_position(
        X_dof_map, *X_ghost_vec, mesh, ref_tail_upper_node_id,  x_tip_u, y_tip_u);
    const bool have_tip_l = get_current_node_position(
        X_dof_map, *X_ghost_vec, mesh, ref_tail_lower_node_id,  x_tip_l, y_tip_l);

    // Get current fin root center (average of upper and lower fin root nodes).
    double x_root_u = 0.0, y_root_u = 0.0, x_root_l = 0.0, y_root_l = 0.0;
    const bool have_root_u = get_current_node_position(
        X_dof_map, *X_ghost_vec, mesh, ref_fin_root_upper_node_id, x_root_u, y_root_u);
    const bool have_root_l = get_current_node_position(
        X_dof_map, *X_ghost_vec, mesh, ref_fin_root_lower_node_id, x_root_l, y_root_l);

    if (!have_tip_u || !have_tip_l || !have_root_u || !have_root_l) return;

    const double x_tip  = 0.5 * (x_tip_u  + x_tip_l);
    const double y_tip  = 0.5 * (y_tip_u  + y_tip_l);
    const double x_root = 0.5 * (x_root_u + x_root_l);
    const double y_root = 0.5 * (y_root_u + y_root_l);
    const double dx_root = x_root_u - x_root_l;
    const double dy_root = y_root_u - y_root_l;
    const double root_gap = std::sqrt(dx_root * dx_root + dy_root * dy_root);
    const int root_is_degenerate =
        root_gap <= 1.0e-8 * std::max(1.0, ref_body_length) ? 1 : 0;

    // Fin chord vector and orientation in the lab frame.
    const double dx_fin = x_tip  - x_root;
    const double dy_fin = y_tip  - y_root;
    const double theta_fin_lab = std::atan2(dy_fin, dx_fin);
    const double fin_length = std::sqrt(dx_fin * dx_fin + dy_fin * dy_fin);

    // Position relative to COM.
    const double x_tip_rel  = x_tip  - x_cm;
    const double y_tip_rel  = y_tip  - y_cm;
    const double x_root_rel = x_root - x_cm;
    const double y_root_rel = y_root - y_cm;

    // Finite-difference lateral (y) velocity at tip and root.
    double v_y_tip  = std::numeric_limits<double>::quiet_NaN();
    double v_y_root = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(s_fin_pitch_prev_time))
    {
        const double dt = loop_time - s_fin_pitch_prev_time;
        if (dt > 1.0e-12)
        {
            v_y_tip  = (y_tip  - s_fin_pitch_prev_y_tip)  / dt;
            v_y_root = (y_root - s_fin_pitch_prev_y_root) / dt;
        }
    }
    s_fin_pitch_prev_y_tip  = y_tip;
    s_fin_pitch_prev_y_root = y_root;
    s_fin_pitch_prev_time   = loop_time;

    // Active muscle phase at the fin root in the same active-body coordinate
    // used by the PK1 stress, not the full fish x/L coordinate.
    const double s_root_norm = active_end_s_norm_effective();
    const double xi_root = active_xi_from_s_norm(s_root_norm);
    const double act_phase_root = 2.0 * M_PI * xi_root /
        std::max(active_wavelength_over_L, 1.0e-12) -
        wave_time_sign * wave_omega * loop_time + active_phase0;

    if (IBTK_MPI::getRank() != 0) return;

    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_fin_pitch_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_fin_pitch_diagnostics(): cannot open "
                         << s_fin_pitch_diag_filename << "\n");
            return;
        }
        out << "step,time"
            << ",x_tip,y_tip,x_root,y_root"
            << ",x_tip_rel,y_tip_rel,x_root_rel,y_root_rel"
            << ",theta_fin_lab,fin_length"
            << ",root_gap,root_is_degenerate"
            << ",s_root_norm,active_xi_root"
            << ",v_y_tip,v_y_root,act_phase_root\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    out << iteration_num
        << "," << loop_time
        << "," << x_tip  << "," << y_tip
        << "," << x_root << "," << y_root
        << "," << x_tip_rel  << "," << y_tip_rel
        << "," << x_root_rel << "," << y_root_rel
        << "," << theta_fin_lab
        << "," << fin_length
        << "," << root_gap
        << "," << root_is_degenerate
        << "," << s_root_norm
        << "," << xi_root
        << "," << v_y_tip
        << "," << v_y_root
        << "," << act_phase_root
        << "\n";
    out.flush();
}

// =========================================================================
// Midline history CSV
//
// Writes a snapshot of the full midline at regular step intervals.
// Each row corresponds to one station at one time. The file contains both
// the body-frame lateral displacement y_body (= deformation pattern) and
// the propulsion-pattern lateral displacement y_prop = y_lab.
// These are the two Y matrices needed for COD and 2-D Fourier analysis.
// =========================================================================
static void
write_midline_history(const int            iteration_num,
                      const double         loop_time,
                      Pointer<IBFEMethod>  ib_method_ops,
                      MeshBase&            mesh,
                      EquationSystems*     equation_systems,
                      const double         x_cm,
                      const double         y_cm)
{
    if (!s_midline_hist_enable) return;
    if (s_midline_hist_interval > 1 &&
        (iteration_num % s_midline_hist_interval != 0)) return;

    const int n = std::max(s_midline_hist_stations, 3);
    const double Lref = std::max(ref_arc_length, 1.0e-12);

    std::vector<MidlineSample> samples;
    double theta_body = 0.0;
    compute_current_midline_samples(ib_method_ops, mesh, equation_systems,
                                    n, x_cm, y_cm, samples, theta_body);
    const std::vector<double> kappa_body = compute_body_midline_curvature(samples);

    // Midline extraction uses MPI reductions, so every rank must participate.
    // Only rank 0 should append rows to the shared CSV.
    if (IBTK_MPI::getRank() != 0) return;

    const double ph = phase01(loop_time);

    std::ofstream f(s_midline_hist_filename,
                    s_midline_hist_header_done ? std::ios::app : std::ios::out);
    if (!f.is_open()) return;

    if (!s_midline_hist_header_done)
    {
        f << "time,cycle_phase,station,s_norm"
          << ",x_lab,y_lab,x_cm,y_cm,theta_body"
          << ",x_body,y_body"   // deformation frame (remove translation + rotation)
          << ",y_prop"          // propulsion pattern: lab-frame lateral motion
          << ",curvature\n";
        s_midline_hist_header_done = true;
    }

    f.setf(std::ios::scientific);
    f.precision(8);
    for (int k = 0; k < static_cast<int>(samples.size()); ++k)
    {
        const double kv = (k < static_cast<int>(kappa_body.size()))
            ? kappa_body[static_cast<std::size_t>(k)]
            : std::numeric_limits<double>::quiet_NaN();
        // Keep lateral recoil/heave in the propulsion pattern; only forward
        // translation is irrelevant for the Y(s,t) matrix used downstream.
        const double y_prop = samples[k].y_lab;
        f << loop_time
          << "," << ph
          << "," << k
          << "," << (samples[k].s / Lref)
          << "," << samples[k].x_lab
          << "," << samples[k].y_lab
          << "," << x_cm
          << "," << y_cm
          << "," << theta_body
          << "," << samples[k].x_body
          << "," << samples[k].y_body
          << "," << y_prop
          << "," << kv
          << "\n";
    }
}

// =========================================================================
// Retained diagnostics for bending, J positivity, and active-work dominance.
// =========================================================================
static void
write_test_diagnostics(const int            iteration_num,
                       const double         loop_time,
                       Pointer<IBFEMethod>  ib_method_ops,
                       MeshBase&            mesh,
                       EquationSystems*     equation_systems)
{
    const bool curvature_phase_due =
        s_curvature_phase_diag_enable &&
        (s_curvature_phase_diag_interval <= 1 ||
         (iteration_num % s_curvature_phase_diag_interval == 0));
    const bool force_decomp_due =
        s_force_decomp_diag_enable &&
        (s_force_decomp_diag_interval <= 1 ||
         (iteration_num % s_force_decomp_diag_interval == 0));
    const bool section_moment_due =
        s_section_moment_diag_enable &&
        (s_section_moment_diag_interval <= 1 ||
         (iteration_num % s_section_moment_diag_interval == 0));
    const bool geometry_conservation_due =
        s_geometry_conservation_diag_enable &&
        (s_geometry_conservation_diag_interval <= 1 ||
         (iteration_num % s_geometry_conservation_diag_interval == 0));
    const bool midline_hist_due =
        s_midline_hist_enable &&
        (s_midline_hist_interval <= 1 ||
         (iteration_num % s_midline_hist_interval == 0));
    if (!curvature_phase_due && !force_decomp_due && !section_moment_due &&
        !geometry_conservation_due && !midline_hist_due) return;
    if (!equation_systems) return;

    const unsigned int dim = mesh.mesh_dimension();
    double F_integral[NDIM] = { 0.0, 0.0 };
    double F_power_on_fluid = std::numeric_limits<double>::quiet_NaN();
    if (force_decomp_due)
    {
        System& F_sys = equation_systems->get_system(ib_method_ops->getForceSystemName());
        NumericVector<double>* F_vec = F_sys.solution.get();
        NumericVector<double>* F_ghost_vec = F_sys.current_local_solution.get();
        F_vec->close();
        copy_and_synch(*F_vec, *F_ghost_vec);

        const DofMap& F_dof_map = F_sys.get_dof_map();
        const FEType fe_type = F_dof_map.variable_type(0);
        std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
        std::unique_ptr<QBase> qrule = fe_type.default_quadrature_rule(dim);
        fe->attach_quadrature_rule(qrule.get());
        const std::vector<double>& JxW = fe->get_JxW();
        const std::vector<std::vector<double>>& phi = fe->get_phi();
        std::vector<std::vector<unsigned int>> dof_indices(NDIM);
        boost::multi_array<double, 2> F_node;
        boost::multi_array<double, 2> U_node;
        VectorValue<double> F_qp;
        VectorValue<double> U_qp;

        const DofMap* U_dof_map_ptr = nullptr;
        NumericVector<double>* U_ghost_vec = nullptr;
        std::vector<std::vector<unsigned int>> U_dof_indices(NDIM);
        {
            System& U_sys = equation_systems->get_system(
                ib_method_ops->getVelocitySystemName());
            NumericVector<double>* U_vec = U_sys.solution.get();
            U_ghost_vec = U_sys.current_local_solution.get();
            U_vec->close();
            copy_and_synch(*U_vec, *U_ghost_vec);
            U_dof_map_ptr = &U_sys.get_dof_map();
            F_power_on_fluid = 0.0;
        }

        for (auto el_it = mesh.active_local_elements_begin();
             el_it != mesh.active_local_elements_end(); ++el_it)
        {
            const Elem* elem = *el_it;
            fe->reinit(elem);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                F_dof_map.dof_indices(elem, dof_indices[d], d);
                U_dof_map_ptr->dof_indices(elem, U_dof_indices[d], d);
            }
            get_values_for_interpolation(F_node, *F_ghost_vec, dof_indices);
            get_values_for_interpolation(U_node, *U_ghost_vec, U_dof_indices);
            for (unsigned int qp = 0; qp < qrule->n_points(); ++qp)
            {
                interpolate(F_qp, qp, F_node, phi);
                for (unsigned int d = 0; d < NDIM; ++d)
                    F_integral[d] += F_qp(d) * JxW[qp];
                interpolate(U_qp, qp, U_node, phi);
                F_power_on_fluid += (F_qp * U_qp) * JxW[qp];
            }
        }
        IBTK_MPI::sumReduction(F_integral, NDIM);
        IBTK_MPI::sumReduction(&F_power_on_fluid, 1);
    }

    double x_cm_new = xcom_tracked;
    double y_cm_new = ycom_tracked;
    compute_fish_com(ib_method_ops, mesh, equation_systems, x_cm_new, y_cm_new, nullptr);

    double vcm_x = xcom_vel;
    double vcm_y = ycom_vel;
    if (std::isfinite(xcom_tracked_time))
    {
        const double dt_eff = loop_time - xcom_tracked_time;
        if (dt_eff > 1.0e-12)
        {
            vcm_x = (x_cm_new - xcom_tracked) / dt_eff;
            vcm_y = (y_cm_new - ycom_tracked) / dt_eff;
        }
    }
    xcom_tracked = x_cm_new;
    ycom_tracked = y_cm_new;
    xcom_tracked_time = loop_time;
    xcom_vel     = vcm_x;
    ycom_vel     = vcm_y;

    if (force_decomp_due)
    {
        write_force_decomposition_diagnostics(iteration_num, loop_time,
                                              ib_method_ops, mesh, equation_systems,
                                              F_integral, F_power_on_fluid,
                                              x_cm_new, y_cm_new,
                                              vcm_x, vcm_y);
    }

    if (geometry_conservation_due)
    {
        write_geometry_conservation_diagnostics(iteration_num, loop_time,
                                                ib_method_ops, mesh,
                                                equation_systems);
    }

    if (section_moment_due)
    {
        write_section_moment_decomposition_diagnostics(iteration_num, loop_time,
                                                       ib_method_ops, mesh,
                                                       equation_systems);
    }

    if (curvature_phase_due)
        write_curvature_phase_diagnostics(iteration_num, loop_time,
                                          ib_method_ops, mesh, equation_systems,
                                          x_cm_new, y_cm_new);

    if (midline_hist_due)
        write_midline_history(iteration_num, loop_time,
                              ib_method_ops, mesh, equation_systems,
                              x_cm_new, y_cm_new);
}

} // namespace ModelData
using namespace ModelData;

// =========================================================================
// main()
// =========================================================================
int main(int argc, char* argv[])
{
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    const LibMeshInit& init = ibtk_init.getLibMeshInit();

    {
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // ── Parse geometry ─────────────────────────────────────────────────
        Mesh mesh(init.comm(), NDIM);
        mesh.read("fish2d.msh");
        mesh.prepare_for_use();

        // ── Read fluid parameters used by diagnostics ─────────────────────
        fluid_viscosity = input_db->getDouble("MU");
        fluid_density =
            input_db->getDoubleWithDefault("RHO", fluid_density);
        fluid_viscosity = std::max(fluid_viscosity, 1.0e-30);
        fluid_density = std::max(fluid_density, 1.0e-30);

        c1_s_passive =
            input_db->getDoubleWithDefault("C1_S_PASSIVE", c1_s_passive);
        c1_s_passive_anterior =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_ANTERIOR", c1_s_passive);
        c1_s_passive_peduncle =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_PEDUNCLE", c1_s_passive);
        c1_s_passive_caudal =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_CAUDAL", c1_s_passive);
        c1_s_body_transition_s =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_BODY_TRANSITION_S",
                                           c1_s_body_transition_s);
        c1_s_body_transition_w =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_BODY_TRANSITION_W",
                                           c1_s_body_transition_w);
        c1_s_caudal_transition_s =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_CAUDAL_TRANSITION_S",
                                           c1_s_caudal_transition_s);
        c1_s_caudal_transition_w =
            input_db->getDoubleWithDefault("C1_S_PASSIVE_CAUDAL_TRANSITION_W",
                                           c1_s_caudal_transition_w);
        kappa_vol = input_db->getDoubleWithDefault("KAPPA_VOL_PASSIVE",
                                                   kappa_vol);
        use_continuum_damping =
            input_db->getBoolWithDefault("USE_CONTINUUM_DAMPING",
                                         use_continuum_damping);
        continuum_damping_factor =
            input_db->getDoubleWithDefault("CONTINUUM_DAMPING_FACTOR",
                                           continuum_damping_factor);
        continuum_damping_stress_cap_over_c1 =
            input_db->getDoubleWithDefault(
                "CONTINUUM_DAMPING_STRESS_CAP_OVER_C1",
                continuum_damping_stress_cap_over_c1);
        target_bending_B_body =
            input_db->getDoubleWithDefault("B_BODY", target_bending_B_body);
        target_bending_B_peduncle =
            input_db->getDoubleWithDefault("B_PEDUNCLE",
                                           target_bending_B_peduncle);
        target_bending_B_caudal =
            input_db->getDoubleWithDefault("B_CAUDAL",
                                           target_bending_B_caudal);
        target_bending_B_body_transition_s =
            input_db->getDoubleWithDefault(
                "B_TARGET_BODY_TRANSITION_S",
                target_bending_B_body_transition_s);
        target_bending_B_body_transition_w =
            input_db->getDoubleWithDefault(
                "B_TARGET_BODY_TRANSITION_W",
                target_bending_B_body_transition_w);
        target_bending_B_caudal_transition_s =
            input_db->getDoubleWithDefault(
                "B_TARGET_CAUDAL_TRANSITION_S",
                target_bending_B_caudal_transition_s);
        target_bending_B_caudal_transition_w =
            input_db->getDoubleWithDefault(
                "B_TARGET_CAUDAL_TRANSITION_W",
                target_bending_B_caudal_transition_w);
        c1_s_passive = std::max(c1_s_passive, 1.0e-12);
        c1_s_passive_anterior = std::max(c1_s_passive_anterior, 1.0e-12);
        c1_s_passive_peduncle = std::max(c1_s_passive_peduncle, 1.0e-12);
        c1_s_passive_caudal = std::max(c1_s_passive_caudal, 1.0e-12);
        kappa_vol = std::max(0.0, kappa_vol);
        continuum_damping_factor = std::max(0.0, continuum_damping_factor);
        continuum_damping_stress_cap_over_c1 =
            std::max(0.0, continuum_damping_stress_cap_over_c1);
        c1_s_body_transition_s = clamp01(c1_s_body_transition_s);
        c1_s_body_transition_w = std::max(c1_s_body_transition_w, 1.0e-12);
        c1_s_caudal_transition_s = clamp01(c1_s_caudal_transition_s);
        c1_s_caudal_transition_w = std::max(c1_s_caudal_transition_w, 1.0e-12);
        target_bending_B_body = std::max(0.0, target_bending_B_body);
        target_bending_B_peduncle =
            std::max(0.0, target_bending_B_peduncle);
        target_bending_B_caudal = std::max(0.0, target_bending_B_caudal);
        target_bending_B_body_transition_s =
            clamp01(target_bending_B_body_transition_s);
        target_bending_B_body_transition_w =
            std::max(target_bending_B_body_transition_w, 1.0e-12);
        target_bending_B_caudal_transition_s =
            clamp01(target_bending_B_caudal_transition_s);
        target_bending_B_caudal_transition_w =
            std::max(target_bending_B_caudal_transition_w, 1.0e-12);

        // ── Read geometry / actuation parameters ──────────────────────────
        fish_length  = input_db->getDoubleWithDefault("FISH_LENGTH", fish_length);
        x_leading    = input_db->getDoubleWithDefault("X_LEADING",   x_leading);
        wave_frequency = input_db->getDoubleWithDefault("WAVE_FREQUENCY", wave_frequency);
        wave_ramp_time = input_db->getDoubleWithDefault("WAVE_RAMP_TIME", wave_ramp_time);
        wave_time_sign = input_db->getDoubleWithDefault("WAVE_TIME_SIGN", wave_time_sign);
        wave_omega     = 2.0 * M_PI * wave_frequency;

        if (input_db->keyExists("PRESCRIBED_TUNA_KAPPA") ||
            input_db->keyExists("PRESCRIBED_TUNA_ETA"))
        {
            TBOX_ERROR("This executable is an active-bending self-propelled case, "
                       "not Case P prescribed-target validation. Remove "
                       "PRESCRIBED_TUNA_KAPPA/PRESCRIBED_TUNA_ETA or use the "
                       "prescribed-tether executable.\n");
        }

        const bool requested_b_kappa =
            input_db->getBoolWithDefault(
                "B_KAPPA_TARGET_ACTIVE_MOMENT", true);
        if (!requested_b_kappa)
        {
            TBOX_ERROR("fish4-5_2 is the dedicated B-kappa target branch; "
                       "B_KAPPA_TARGET_ACTIVE_MOMENT must be TRUE.\n");
        }
        active_kappa_amp =
            input_db->getDoubleWithDefault("ACTIVE_KAPPA_AMP",
                                           active_kappa_amp);
        active_moment_mode =
            parse_active_moment_mode(input_db->getStringWithDefault(
                "ACTIVE_MOMENT_MODE", active_moment_mode_name()));
        static_moment_m0 =
            input_db->getDoubleWithDefault("STATIC_MOMENT_M0", static_moment_m0);
        initial_bend_amplitude =
            input_db->getDoubleWithDefault("INITIAL_BEND_AMPLITUDE",
                                           initial_bend_amplitude);
        active_wavelength_over_L =
            input_db->getDoubleWithDefault("LAMBDA_ACT_OVER_LACT",
                                           active_wavelength_over_L);
        active_phase0 =
            input_db->getDoubleWithDefault("ACTIVE_PHASE0", active_phase0);
        active_moment_to_stress_sign =
            input_db->getDoubleWithDefault("ACTIVE_MOMENT_TO_STRESS_SIGN",
                                           active_moment_to_stress_sign);
        active_k_shape_mode =
            parse_active_k_shape_mode(input_db->getStringWithDefault(
                "K_SHAPE_MODE", active_k_shape_mode_name()));
        active_envelope_mode =
            parse_active_envelope_mode(input_db->getStringWithDefault(
                "ACTIVE_ENVELOPE_MODE", active_envelope_mode_name()));
        posterior_g0 =
            input_db->getDoubleWithDefault("POSTERIOR_G0", posterior_g0);
        posterior_power =
            input_db->getDoubleWithDefault("POSTERIOR_POWER", posterior_power);
        b_compensation_alpha =
            input_db->getDoubleWithDefault("B_COMPENSATION_ALPHA",
                                           b_compensation_alpha);
        tail_taper_width =
            input_db->getDoubleWithDefault("TAIL_TAPER_WIDTH",
                                           tail_taper_width);
        active_envelope_cap_safe_over_E =
            input_db->getDoubleWithDefault(
                "ACTIVE_ENVELOPE_CAP_SAFE_OVER_E",
                active_envelope_cap_safe_over_E);
        active_s_start  = input_db->getDoubleWithDefault("ACTIVE_S_START",  active_s_start);
        active_s_end    = input_db->getDoubleWithDefault("ACTIVE_S_END",    active_s_end);
        active_s_smooth = input_db->getDoubleWithDefault("ACTIVE_S_SMOOTH", active_s_smooth);
        active_band_fraction =
            input_db->getDoubleWithDefault("ACTIVE_BAND_FRACTION", active_band_fraction);
        active_i2_h_power =
            input_db->getDoubleWithDefault("ACTIVE_I2_H_POWER", active_i2_h_power);
        active_t_act_max_over_c1 =
            input_db->getDoubleWithDefault("ACTIVE_T_ACT_MAX_OVER_C1",
                                           active_t_act_max_over_c1);
        reference_profile_bins =
            input_db->getIntegerWithDefault("REFERENCE_PROFILE_BINS", reference_profile_bins);
        reference_backbone_end_x =
            input_db->getDoubleWithDefault("REFERENCE_BACKBONE_END_X", reference_backbone_end_x);
        use_laplace_reference_parameterization =
            input_db->getBoolWithDefault("USE_LAPLACE_REFERENCE_PARAMETERIZATION",
                                         use_laplace_reference_parameterization);
        use_fe_active_section_data =
            input_db->getBoolWithDefault("USE_FE_ACTIVE_SECTION_DATA",
                                         use_fe_active_section_data);
        const bool requested_fe_normalized =
            input_db->getBoolWithDefault("FE_NORMALIZED_STRESS", true);
        if (!requested_fe_normalized)
        {
            TBOX_ERROR("fish4-5_2 does not provide the old active-stress "
                       "mapping; FE_NORMALIZED_STRESS must be TRUE.\n");
        }
        fe_normalized_stress_bins =
            input_db->getIntegerWithDefault(
                "FE_NORMALIZED_STRESS_BINS",
                fe_normalized_stress_bins);
        fe_normalized_stress_quad_order =
            input_db->getStringWithDefault(
                "FE_NORMALIZED_STRESS_QUAD_ORDER",
                fe_normalized_stress_quad_order);
        fe_section_i2_floor_ratio =
            input_db->getDoubleWithDefault("FE_SECTION_I2_FLOOR_RATIO",
                                           fe_section_i2_floor_ratio);
        laplace_head_bc_width_over_L =
            input_db->getDoubleWithDefault("LAPLACE_HEAD_BC_WIDTH_OVER_L",
                                           laplace_head_bc_width_over_L);
        laplace_tail_bc_width_over_L =
            input_db->getDoubleWithDefault("LAPLACE_TAIL_BC_WIDTH_OVER_L",
                                           laplace_tail_bc_width_over_L);
        active_s_start = clamp01(active_s_start);
        if (active_s_end >= 0.0) active_s_end = clamp01(active_s_end);
        active_s_smooth = std::max(0.0, active_s_smooth);
        active_band_fraction = std::max(0.0, active_band_fraction);
        active_i2_h_power = std::max(0.0, active_i2_h_power);
        fe_section_i2_floor_ratio = std::max(0.0, fe_section_i2_floor_ratio);
        active_wavelength_over_L = std::max(active_wavelength_over_L, 1.0e-12);
        active_moment_to_stress_sign =
            (active_moment_to_stress_sign >= 0.0) ? 1.0 : -1.0;
        active_t_act_max_over_c1 = std::max(active_t_act_max_over_c1, 1.0e-12);
        fe_normalized_stress_bins = std::max(1, fe_normalized_stress_bins);
        reference_profile_bins = std::max(8, reference_profile_bins);
        laplace_head_bc_width_over_L = std::max(0.0, laplace_head_bc_width_over_L);
        laplace_tail_bc_width_over_L = std::max(0.0, laplace_tail_bc_width_over_L);
        if (active_kappa_amp < 0.0)
        {
            TBOX_ERROR("ACTIVE_KAPPA_AMP must be non-negative.\n");
        }
        posterior_g0 = clamp01(posterior_g0);
        posterior_power = std::max(posterior_power, 1.0e-12);
        if (b_compensation_alpha < 0.0)
        {
            TBOX_ERROR("B_COMPENSATION_ALPHA must be non-negative.\n");
        }
        if (active_envelope_mode ==
            ActiveEnvelopeMode::POSTERIOR_B_COMPENSATED)
        {
            if (target_bending_B_caudal <= 0.0)
            {
                TBOX_ERROR("POSTERIOR_B_COMPENSATED requires B_CAUDAL > 0.\n");
            }
        }
        tail_taper_width = std::max(0.0, tail_taper_width);
        active_envelope_cap_safe_over_E =
            std::max(0.0, active_envelope_cap_safe_over_E);

        if (input_db->getBoolWithDefault("BETA_ACT_CALIBRATION", false))
        {
            TBOX_ERROR("fish4-5_2 is calibrated with ACTIVE_KAPPA_AMP; "
                       "BETA_ACT_CALIBRATION is not supported.\n");
        }

        // ── Midline history CSV ────────────────────────────────────────────
        s_midline_hist_enable =
            input_db->getBoolWithDefault("MIDLINE_HIST_ENABLE", s_midline_hist_enable);
        s_midline_hist_interval =
            input_db->getIntegerWithDefault("MIDLINE_HIST_INTERVAL",
                                            s_midline_hist_interval);
        s_midline_hist_stations =
            input_db->getIntegerWithDefault("MIDLINE_HIST_STATIONS",
                                            s_midline_hist_stations);
        s_midline_hist_filename =
            input_db->getStringWithDefault("MIDLINE_HIST_FILENAME",
                                           s_midline_hist_filename);
        s_midline_hist_interval = std::max(1, s_midline_hist_interval);
        s_midline_hist_stations = std::max(3, s_midline_hist_stations);

        if (active_s_end >= 0.0 && active_s_end <= active_s_start)
        {
            TBOX_ERROR("ACTIVE_S_END must be greater than ACTIVE_S_START.\n");
        }

        build_reference_profile_from_mesh(mesh);
        initialize_tail_tracking_points(mesh);
        if (use_laplace_reference_parameterization && use_fe_active_section_data)
        {
            pout << "  Building FE-consistent active section data...\n";
            build_fe_section_data(mesh);
        }
        if (active_s_span_norm_effective() <= 1.0e-12)
        {
            TBOX_ERROR("Effective active-body span is zero: check ACTIVE_S_START "
                       "and ACTIVE_S_END.\n");
        }
        initialize_reference_com_from_mesh(mesh);

        // ── Read only the retained test diagnostics ─────────────────────────
        s_curvature_phase_diag_enable = input_db->getBoolWithDefault(
                                             "CURVATURE_PHASE_DIAG_ENABLE",
                                             s_curvature_phase_diag_enable);
        s_curvature_phase_diag_interval = std::max(1, input_db->getIntegerWithDefault(
                                                        "CURVATURE_PHASE_DIAG_INTERVAL",
                                                        s_curvature_phase_diag_interval));
        s_curvature_phase_diag_filename = input_db->getStringWithDefault(
                                               "CURVATURE_PHASE_DIAG_FILENAME",
                                               s_curvature_phase_diag_filename);
        s_curvature_phase_diag_stations = std::max(3, input_db->getIntegerWithDefault(
                                                        "CURVATURE_PHASE_DIAG_STATIONS",
                                                        s_curvature_phase_diag_stations));
        s_curvature_phase_diag_start_time = input_db->getDoubleWithDefault(
                                                "CURVATURE_PHASE_DIAG_START_TIME",
                                                wave_ramp_time);
        s_progress_print_interval = std::max(1, input_db->getIntegerWithDefault(
                                                 "PROGRESS_PRINT_INTERVAL",
                                                 s_progress_print_interval));

        s_force_decomp_diag_enable = input_db->getBoolWithDefault(
            "FORCE_DECOMP_DIAG_ENABLE", s_force_decomp_diag_enable);
        s_force_decomp_diag_interval = std::max(1, input_db->getIntegerWithDefault(
            "FORCE_DECOMP_DIAG_INTERVAL", s_force_decomp_diag_interval));
        s_force_decomp_diag_filename = input_db->getStringWithDefault(
            "FORCE_DECOMP_DIAG_FILENAME", s_force_decomp_diag_filename);
        s_force_decomp_quad_order = input_db->getStringWithDefault(
            "FORCE_DECOMP_QUAD_ORDER", s_force_decomp_quad_order);

        s_section_moment_diag_enable = input_db->getBoolWithDefault(
            "SECTION_MOMENT_DIAG_ENABLE", s_section_moment_diag_enable);
        s_section_moment_diag_interval = std::max(1, input_db->getIntegerWithDefault(
            "SECTION_MOMENT_DIAG_INTERVAL", s_section_moment_diag_interval));
        s_section_moment_diag_filename = input_db->getStringWithDefault(
            "SECTION_MOMENT_DIAG_FILENAME", s_section_moment_diag_filename);
        s_section_moment_quad_order = input_db->getStringWithDefault(
            "SECTION_MOMENT_QUAD_ORDER", s_section_moment_quad_order);
        s_section_moment_diag_bins = input_db->getIntegerWithDefault(
            "SECTION_MOMENT_DIAG_BINS", s_section_moment_diag_bins);

        s_active_moment_profile_debug_enable =
            input_db->getBoolWithDefault(
                "ACTIVE_MOMENT_PROFILE_DEBUG_ENABLE",
                s_active_moment_profile_debug_enable);
        s_active_moment_profile_debug_filename =
            input_db->getStringWithDefault(
                "ACTIVE_MOMENT_PROFILE_DEBUG_FILENAME",
                s_active_moment_profile_debug_filename);

        s_geometry_conservation_diag_enable = input_db->getBoolWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_ENABLE", s_geometry_conservation_diag_enable);
        s_geometry_conservation_diag_interval = std::max(1, input_db->getIntegerWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_INTERVAL", s_geometry_conservation_diag_interval));
        s_geometry_conservation_diag_filename = input_db->getStringWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_FILENAME", s_geometry_conservation_diag_filename);

        // ── Print startup summary ─────────────────────────────────────────
        const double L_act = active_phase_length_dimensional();
        const double L_ref = std::max(ref_arc_length, 1.0e-12);
        const double L_fish = std::max(fish_length, 1.0e-12);
        const double lambda_phase = active_phase_wavelength_dimensional();
        const double active_s0 = active_s_start_norm_effective();
        const double active_s1 = active_s_end_norm_effective();
        const double active_s_span = active_s_span_norm_effective();
        const double U_act   = wave_frequency * lambda_phase;
        const double Re_act  = fluid_density * U_act * L_fish /
            std::max(fluid_viscosity, 1.0e-30);
        pout << "\n=== IBFE continuum fish: B-kappa + FE-normalized stress ===\n";
        pout << "  fish length = " << fish_length
             << ", reference arc length = " << ref_arc_length << "\n";
        pout << "  active s_norm range = [" << active_s0 << ", " << active_s1
             << "], span = " << active_s_span << "\n";
        pout << "  lambda_act = " << lambda_phase
             << ", U_act = " << U_act
             << ", Re_act = " << Re_act << "\n";
        pout << "  passive C1 profile = "
             << c1_s_passive_anterior << " / "
             << c1_s_passive_peduncle << " / "
             << c1_s_passive_caudal << "\n";
        pout << "  passive C1 samples s_norm = 0, 0.45, 0.65, 0.82, 0.86, 0.90, 1: "
             << get_c1_s_passive_local(0.00) << ", "
             << get_c1_s_passive_local(0.45) << ", "
             << get_c1_s_passive_local(0.65) << ", "
             << get_c1_s_passive_local(0.82) << ", "
             << get_c1_s_passive_local(0.86) << ", "
             << get_c1_s_passive_local(0.90) << ", "
             << get_c1_s_passive_local(1.00) << "\n";
        pout << "  passive KAPPA_VOL = " << kappa_vol << "\n";
        pout << "  continuum structural damping = "
             << (use_continuum_damping ? "ON" : "OFF")
             << ", factor = " << continuum_damping_factor
             << ", stress cap = " << continuum_damping_stress_cap_over_c1
             << " * C1_S_PASSIVE(x)\n";
        pout << "  active moment mode = " << active_moment_mode_name()
             << ", ACTIVE_KAPPA_AMP = " << active_kappa_amp
             << ", static M0 = " << static_moment_m0
             << ", initial bend amplitude = " << initial_bend_amplitude << "\n";
        pout << "  B-kappa target active moment = ON"
             << ", K_shape = " << active_k_shape_formula_string()
             << ", active stress cap = "
             << active_t_act_max_over_c1 << " * C1_S_PASSIVE(x)\n";
        pout << "  active B reference body/peduncle/caudal = "
             << target_bending_B_body << " / "
             << target_bending_B_peduncle << " / "
             << target_bending_B_caudal
             << " (active scaling only; not passive stiffness)\n";
        pout << "  active target curvature = ACTIVE_KAPPA_AMP"
                "*G_active(s)*cos(active phase)\n";
        pout << "  active moment = B_target(s)*active_target_curvature\n";
        pout << "  active envelope mode = " << active_envelope_mode_name()
             << ", POSTERIOR_G0 = " << posterior_g0
             << ", POSTERIOR_POWER = " << posterior_power
             << ", B_COMPENSATION_ALPHA = " << b_compensation_alpha
             << ", TAIL_TAPER_WIDTH = " << tail_taper_width
             << ", envelope cap safe/E = "
             << (active_envelope_cap_safe_over_E > 0.0 ?
                 std::to_string(active_envelope_cap_safe_over_E) :
                 "disabled")
             << "\n";
        pout << "  active zone request s/L = [" << active_s_start << ", ";
        if (active_s_end < 0.0)
            pout << "AUTO_REFERENCE_BACKBONE_END";
        else
            pout << active_s_end;
        pout << "], effective end = " << active_s1
             << ", taper = " << active_s_smooth << "\n";
        pout << "  active band fraction = " << active_band_fraction
             << ", I2_eff_unit = " << active_band_second_moment_unit()
             << ", ACTIVE_I2_H_POWER = " << active_i2_h_power << "\n";
        pout << "  FE-normalized stress = ON"
             << ", bins = " << fe_normalized_stress_bins
             << ", quad_order = " << fe_normalized_stress_quad_order
             << "\n";
        pout << "  active moment-to-stress mapping = FE q(eta), "
                "zero resultant force and unit curvature moment\n";
        pout << "  FE active section correction = "
             << (active_section_correction_enabled() ? "ON" : "OFF")
             << " (bypassed by FE_NORMALIZED_STRESS)"
             << "\n";
        pout << "  active stress = "
             << -active_moment_to_stress_sign
             << " * Mm * q_FE(eta); curvature-conjugate moment = "
             << -active_moment_to_stress_sign << " * Mm\n";
        pout << "  reference profile bins = " << reference_profile_bins << "\n";
        pout << "  reference parameterization = "
             << (use_laplace_reference_parameterization ?
                 "LAPLACE_GRAPH_HARMONIC" : "CENTERLINE_PROJECTION")
             << "\n";
        if (use_laplace_reference_parameterization)
        {
            pout << "  Laplace BC widths head/tail over L = "
                 << laplace_head_bc_width_over_L << " / "
                 << laplace_tail_bc_width_over_L << "\n";
            pout << "  Laplace Dirichlet constraints = enforced_exactly\n";
        }
        pout << "  midline extraction = STRICT_BOUNDARY_EDGE_INTERSECTIONS\n";
        pout << "    body sections: phi/s isocontour boundary-edge intersections\n";
        pout << "  reference x-range = [" << ref_x_min << ", " << ref_x_max << "]\n";
        pout << "  X_LEADING interpreted as " << wave_head_location << "\n";
        pout << "  reference s direction = head-to-tail\n";
        pout << "  wave_time_sign = " << wave_time_sign << "\n";
        pout << "  active phase coordinate = ACTIVE_BODY_XI\n";
        pout << "  active wavelength over active length = " << active_wavelength_over_L << "\n";
        pout << "  active phase0 = " << active_phase0 << "\n";
        pout << "  active phase = 2*pi*xi/LAMBDA_ACT_OVER_LACT "
             << active_phase_time_sign_string()
             << " omega*t + ACTIVE_PHASE0\n";
        pout << "  K_shape mode = " << active_k_shape_mode_name()
             << ", K_shape(xi) = " << active_k_shape_formula_string() << "\n";
        pout << "  active phase-speed direction in s = "
             << active_phase_propagation_s_string() << "\n";
        pout << "  active phase-speed x sign = "
             << active_phase_propagation_x_sign() << "\n";
        pout << "  reference backbone end x = " << ref_backbone_end_x << "\n";
        pout << "  reference arc length = " << ref_arc_length << "\n";
        pout << "  reference mesh COM = ("
             << s_reference_xcom << ", " << s_reference_ycom << ")\n";
        pout << "  reference mesh area = " << s_reference_area << "\n";
        pout << "  mesh h_max/L = " << ref_h_max / std::max(fish_length, 1.0e-12) << "\n";
        pout << "  progress print interval = " << s_progress_print_interval << "\n";
        pout << "  retained CSV diagnostics = "
             << s_midline_hist_filename << " (whole-body bending), "
             << s_curvature_phase_diag_filename << " (curvature phase), "
             << s_geometry_conservation_diag_filename << " (J_min), "
             << s_force_decomp_diag_filename << " (active work), "
             << s_section_moment_diag_filename << " (section moments)\n";
        pout << "  midline history diagnostics = "
             << (s_midline_hist_enable ? "on" : "off")
             << ", interval = " << s_midline_hist_interval
             << ", stations = " << s_midline_hist_stations
             << ", file = " << s_midline_hist_filename << "\n";
        pout << "  curvature/phase diagnostics = "
             << (s_curvature_phase_diag_enable ? "on" : "off")
             << ", interval = " << s_curvature_phase_diag_interval
             << ", stations = " << s_curvature_phase_diag_stations
             << ", start_time = " << s_curvature_phase_diag_start_time
             << ", file = " << s_curvature_phase_diag_filename << "\n";
        pout << "  force decomposition diagnostics = "
             << (s_force_decomp_diag_enable ? "on" : "off")
             << ", interval = " << s_force_decomp_diag_interval
             << ", quad_order = " << s_force_decomp_quad_order
             << ", file = " << s_force_decomp_diag_filename << "\n";
        pout << "  section moment diagnostics = "
             << (s_section_moment_diag_enable ? "on" : "off")
             << ", interval = " << s_section_moment_diag_interval
             << ", bins = " << (s_section_moment_diag_bins > 0 ?
                                s_section_moment_diag_bins : reference_profile_bins)
             << ", quad_order = " << s_section_moment_quad_order
             << ", file = " << s_section_moment_diag_filename << "\n";
        pout << "  geometry conservation diagnostics = "
             << (s_geometry_conservation_diag_enable ? "on" : "off")
             << ", interval = " << s_geometry_conservation_diag_interval
             << ", file = " << s_geometry_conservation_diag_filename << "\n";
        pout << "  active moment profile debug = "
             << (s_active_moment_profile_debug_enable ? "on" : "off")
             << ", file = " << s_active_moment_profile_debug_filename << "\n";

        pout << "  effective active-zone length = "
             << active_s_span * ref_arc_length;
        if (active_s_end < 0.0)
        {
            pout << " (auto endpoint from REFERENCE_BACKBONE_END_X)\n";
        }
        else
        {
            pout << " (requested unclipped length = "
                 << (active_s_end - active_s_start) * ref_arc_length
                 << ")\n";
        }

        if (std::abs(ref_body_length - fish_length) > 1.0e-6)
        {
            pout << "  WARNING: mesh body length (" << ref_body_length
                 << ") differs from input FISH_LENGTH (" << fish_length << ")\n";
        }
        const double dist_x_leading_to_mesh_edge =
            std::min(std::abs(ref_x_min - x_leading), std::abs(ref_x_max - x_leading));
        if (dist_x_leading_to_mesh_edge > 1.0e-6)
        {
            pout << "  WARNING: input X_LEADING (" << x_leading
                 << ") differs from both mesh x edges [" << ref_x_min
                 << ", " << ref_x_max << "]\n";
        }
        pout << "=======================================================\n\n";

        // ── Build IBAMR solver objects ─────────────────────────────────────
        Pointer<INSHierarchyIntegrator> navier_stokes_integrator;
        const string solver_type =
            app_initializer->getComponentDatabase("Main")->getString("solver_type");
        if (solver_type == "STAGGERED")
        {
            navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
                "INSStaggeredHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
        }
        else
        {
            navier_stokes_integrator = new INSCollocatedHierarchyIntegrator(
                "INSCollocatedHierarchyIntegrator",
                app_initializer->getComponentDatabase("INSCollocatedHierarchyIntegrator"));
        }

        Pointer<IBFEMethod> ib_method_ops = new IBFEMethod(
            "IBFEMethod",
            app_initializer->getComponentDatabase("IBFEMethod"),
            &mesh,
            app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"),
            true,
            app_initializer->getRestartReadDirectory(),
            app_initializer->getRestartRestoreNumber());

        Pointer<IBHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator(
                "IBHierarchyIntegrator",
                app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                ib_method_ops,
                navier_stokes_integrator);

        Pointer<CartesianGridGeometry<NDIM> > grid_geometry =
            new CartesianGridGeometry<NDIM>(
                "CartesianGeometry",
                app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy =
            new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>(
                "StandardTagAndInitialize", time_integrator,
                app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM> > load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer",
                                   app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm =
            new GriddingAlgorithm<NDIM>(
                "GriddingAlgorithm",
                app_initializer->getComponentDatabase("GriddingAlgorithm"),
                error_detector, box_generator, load_balancer);

        std::vector<int> ref_geom_vars(REF_GEOM_N_VARS);
        for (unsigned int v = 0; v < REF_GEOM_N_VARS; ++v) ref_geom_vars[v] = v;
        const std::vector<SystemData> ref_geom_sys_data(
            1, SystemData(REF_GEOM_SYSTEM_NAME, ref_geom_vars));
        IBFEMethod::PK1StressFcnData PK1_dev_data(
            PK1_dev_stress_function, ref_geom_sys_data);
        IBFEMethod::PK1StressFcnData PK1_dil_data(
            PK1_dil_stress_function, ref_geom_sys_data);
        IBFEMethod::PK1StressFcnData PK1_act_data(
            PK1_active_stress_function, ref_geom_sys_data);
        std::vector<int> velocity_vars(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) velocity_vars[d] = d;
        std::vector<SystemData> ref_geom_velocity_grad_sys_data = ref_geom_sys_data;
        ref_geom_velocity_grad_sys_data.push_back(
            SystemData(ib_method_ops->getVelocitySystemName(),
                       std::vector<int>(), velocity_vars));
        IBFEMethod::PK1StressFcnData PK1_damp_data(
            PK1_continuum_damping_stress_function,
            ref_geom_velocity_grad_sys_data);

        PK1_dev_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_DEV_QUAD_ORDER", "THIRD"));
        PK1_dil_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_DIL_QUAD_ORDER", "THIRD"));
        PK1_act_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_ACT_QUAD_ORDER", "FIFTH"));
        PK1_damp_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_DAMP_QUAD_ORDER", "FIFTH"));

        ib_method_ops->registerPK1StressFunction(PK1_dev_data);
        ib_method_ops->registerPK1StressFunction(PK1_dil_data);
        if (use_continuum_damping)
        {
            ib_method_ops->registerPK1StressFunction(PK1_damp_data);
        }
        ib_method_ops->registerPK1StressFunction(PK1_act_data);
        ib_method_ops->registerInitialCoordinateMappingFunction(
            coordinate_mapping_function);

        ib_method_ops->initializeFEEquationSystems();
        EquationSystems* equation_systems =
            ib_method_ops->getFEDataManager()->getEquationSystems();
        add_reference_geometry_system(equation_systems);

        // ── Post-processor ─────────────────────────────────────────────────
        Pointer<IBFEPostProcessor> ib_post_processor =
            new IBFECentroidPostProcessor("IBFEPostProcessor",
                                          ib_method_ops->getFEDataManager());
        ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT,
                                                   IBFEPostProcessor::FF_fcn);

        IBFEMethod::PK1StressFcnData pk1_act_post_data(
            PK1_active_stress_function, ref_geom_sys_data);
        ib_post_processor->registerTensorVariable(
            "sigma_active", MONOMIAL, CONSTANT,
            IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
            ref_geom_sys_data, &pk1_act_post_data);
        IBFEMethod::PK1StressFcnData pk1_damp_post_data(
            PK1_continuum_damping_stress_function,
            ref_geom_velocity_grad_sys_data);
        if (use_continuum_damping)
        {
            ib_post_processor->registerTensorVariable(
                "sigma_continuum_structural_damping", MONOMIAL, CONSTANT,
                IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                ref_geom_velocity_grad_sys_data, &pk1_damp_post_data);
        }

        // ── Initial conditions / BCs ────────────────────────────────────────
        if (input_db->keyExists("VelocityInitialConditions"))
        {
            Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
                "u_init",
                app_initializer->getComponentDatabase("VelocityInitialConditions"),
                grid_geometry);
            navier_stokes_integrator->registerVelocityInitialConditions(u_init);
        }

        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            u_bc_coefs[d] = new muParserRobinBcCoefs(
                "u_bc_coefs_" + std::to_string(d),
                app_initializer->getComponentDatabase("VelocityBcCoefs_" + std::to_string(d)),
                grid_geometry);
        }
        navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);

        const bool dump_viz_data = app_initializer->dumpVizData();
        const int  viz_dump_interval = app_initializer->getVizDumpInterval();
        // Save exodus filename now — app_initializer will be nulled after initializePatchHierarchy.
#ifdef LIBMESH_HAVE_EXODUS_API
        const string exodus_filename = app_initializer->getExodusIIFilename();
#else
        const string exodus_filename;
#endif
        const bool uses_visit = dump_viz_data && app_initializer->getVisItDataWriter();
        const bool uses_exodus = dump_viz_data && !exodus_filename.empty();
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit) time_integrator->registerVisItDataWriter(visit_data_writer);
        std::unique_ptr<ExodusII_IO> exodus_io =
            uses_exodus ? std::make_unique<ExodusII_IO>(mesh) : nullptr;
        if (uses_exodus)
        {
            exodus_io->append(RestartManager::getManager()->isFromRestart());
        }

        // ── Initialize hierarchy ────────────────────────────────────────────
        ib_method_ops->initializeFEData();
        fill_reference_geometry_system(mesh, equation_systems);
        initialize_fe_normalized_stress(mesh, equation_systems);
        write_active_moment_profile_debug();

        if (ib_post_processor) ib_post_processor->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);
        app_initializer.setNull();

        // ── t=0 diagnostics ─────────────────────────────────────────────────
        int    iteration_num = time_integrator->getIntegratorStep();
        double loop_time     = time_integrator->getIntegratorTime();

        // Initialise COM at t=0
        compute_fish_com(ib_method_ops, mesh, equation_systems,
                         xcom_tracked, ycom_tracked);
        xcom_tracked_time = loop_time;
        xcom_vel = 0.0;
        ycom_vel = 0.0;

        write_test_diagnostics(iteration_num, loop_time,
                               ib_method_ops, mesh, equation_systems);

        if (dump_viz_data)
        {
            if (uses_visit)
            {
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            }
            if (uses_exodus)
            {
                if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                exodus_io->write_timestep(
                    exodus_filename,
                    *equation_systems, iteration_num / viz_dump_interval + 1, loop_time);
            }
        }

        // ── Main time loop ──────────────────────────────────────────────────
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;

        while (!IBTK::rel_equal_eps(loop_time, loop_time_end)
               && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time     = time_integrator->getIntegratorTime();
            const bool print_timestep_log =
                (iteration_num % s_progress_print_interval == 0);

            if (print_timestep_log)
            {
                pout << "\n+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
                pout << "At beginning of timestep # " << iteration_num << "\n";
                pout << "Simulation time is " << loop_time << "\n";
                pout << "COM = (" << xcom_tracked << ", " << ycom_tracked << ")\n";
                pout << "V_cm = (" << xcom_vel   << ", " << ycom_vel    << ")\n";
            }

            dt = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (print_timestep_log || last_step)
            {
                pout << "\nAt end of timestep # " << (iteration_num - 1) << "\n";
                pout << "Simulation time is " << loop_time << "\n";
                pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n";
            }

            write_test_diagnostics(iteration_num, loop_time,
                                   ib_method_ops, mesh, equation_systems);
            if (dump_viz_data && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                    exodus_io->write_timestep(
                        exodus_filename,
                        *equation_systems,
                        iteration_num / viz_dump_interval + 1, loop_time);
                }
            }
        }

        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];
    }
    return 0;
}

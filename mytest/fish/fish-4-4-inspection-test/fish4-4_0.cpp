// IBFE continuum fish with full physical material stresses.
//
// The passive body is a spatially varying two-dimensional continuum material
// calibrated from the target bending stiffness B(s), with objective
// Kelvin-Voigt axial damping. Active bending is represented by a
// self-equilibrated axial PK1 stress over each material section. No extracted
// centerline force or beam-specific time-step controller is used.

#include <SAMRAI_config.h>
#include <petscsys.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
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

// --- Full physical material ---
static double fluid_density = 1.0;
static double fluid_viscosity = 0.0008;
static double target_bending_B_body = 1.5e-4;
static double target_bending_B_peduncle = 7.5e-5;
static double target_bending_B_caudal = 1.0e-4;
static double material_body_transition_s = 0.60;
static double material_body_transition_w = 0.30;
static double material_caudal_transition_s = 0.85;
static double material_caudal_transition_w = 0.10;
static double material_nu_eff = 0.45;
static double section_i2_floor_ratio = 1.0e-3;
static double fiber_stiffness_ratio = 0.0;
static double structural_kv_loss_factor = 0.02;
static double structural_kv_stress_cap_over_E = 10.0;
static double active_t_max_abs = 10.0;
static double active_t_max_over_E = 0.25;
enum class ActiveCrossSectionMode { LINEAR_ETA, MUSCLE_BAND };
static ActiveCrossSectionMode active_cross_section_mode =
    ActiveCrossSectionMode::MUSCLE_BAND;
static double active_band_fraction = 0.40;
static int active_section_bins = 128;
static std::string active_section_quad_order = "FIFTH";

// --- Geometry / actuation frequency ---
static double fish_length = 1.00;
static double x_leading   = 1.00;
static double wave_frequency = 1.00;
static double wave_ramp_time = 3.0;
static double wave_omega     = 2.0 * M_PI * 1.00;
static double wave_time_sign = 1.0;

static double beta_act  = 0.0;
enum class ActiveMomentMode { TRAVELING, STATIC };
static ActiveMomentMode active_moment_mode = ActiveMomentMode::STATIC;
static double static_moment_m0 = 0.0;
static double initial_bend_amplitude = 0.01;
static double active_wavelength_over_L = 1.00;
static double active_phase0 = 0.0;
enum class ActiveKShapeMode { HALF_BELL, BELL };
static ActiveKShapeMode active_k_shape_mode = ActiveKShapeMode::BELL;
static double active_s_start  = 0.00;
static double active_s_end    = 1.00;
static double active_s_smooth = 0.0;
static double active_end_s_norm =
    std::numeric_limits<double>::quiet_NaN();
static int    reference_profile_bins = 128;
static std::string wave_head_location = "x_min";
static double reference_centerline_end_x =
    std::numeric_limits<double>::quiet_NaN(); // auto-detect fork root unless set
static bool   use_laplace_reference_parameterization = true;
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
static double ref_centerline_end_x = 1.00;
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

struct ActiveSectionNormalization
{
    bool valid = false;
    double s_mid = 0.0;
    double ds = 0.0;
    double area = 0.0;
    double eta_mean = 0.0;
    double I2 = 0.0;
    double g_mean = 0.0;
    double q_scale = 0.0;
    double q_abs_max = 0.0;
    double unit_force = 0.0;
    double unit_moment = 0.0;
};

static std::vector<CenterlineSegment> ref_centerline_segments;
static std::map<dof_id_type, ReferenceGeometrySample> ref_laplace_node_geom;
static bool ref_laplace_parameterization_built = false;
static std::vector<ActiveSectionNormalization> active_section_normalization;

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
static libMesh::Point ref_tail_tip_center_point = libMesh::Point(1.0, 0.0);

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
static double s_curvature_phase_last_power =
    std::numeric_limits<double>::quiet_NaN();
static double s_curvature_phase_last_positive_power =
    std::numeric_limits<double>::quiet_NaN();
static int s_curvature_phase_samples = 0;

// ── Body displacement amplitude envelope (per curvature station, per beat) ──
// Tracks peak-to-peak y_body at each station; resets every integer beat cycle.
// A_body_norm[k] = (max-min)/2 / Lref from the last completed beat cycle.
static std::vector<double> s_y_body_cyc_max; // running max y_body per station
static std::vector<double> s_y_body_cyc_min; // running min y_body per station
static std::vector<double> s_y_body_amp;     // half-amplitude from last completed cycle
static int s_y_body_cyc_idx = -1;            // integer cycle index for reset detection

// ── Geometry conservation diagnostics ────────────────────────────────────
static bool        s_geometry_conservation_diag_enable   = true;
static int         s_geometry_conservation_diag_interval = 100;
static std::string s_geometry_conservation_diag_filename = "geometry_conservation_diag.csv";

// -- Force decomposition diagnostics -------------------------------------
static bool        s_force_decomp_diag_enable   = false;
static int         s_force_decomp_diag_interval = 100;
static std::string s_force_decomp_diag_filename = "force_decomposition_diag.csv";
static std::string s_force_decomp_quad_order    = "FIFTH";
static double      s_force_decomp_prev_time =
    std::numeric_limits<double>::quiet_NaN();

// -- Active section equilibrium diagnostics -----------------------------
static bool        s_active_section_diag_enable   = true;
static int         s_active_section_diag_interval = 100;
static std::string s_active_section_diag_filename = "active_section_diag.csv";

// -- Material profile diagnostics ----------------------------------------
static bool        s_material_profile_diag_enable = true;
static std::string s_material_profile_diag_filename = "material_profile_diag.csv";
static int         s_material_profile_diag_samples = 201;

// ── Midline history CSV ───────────────────────────────────────────────────────
// Full per-station midline snapshot at regular step intervals. Columns include
// both body-frame deformation (y_body) and propulsion-pattern lateral motion
// (y_prop = y_lab), which preserve recoil/heave for COD and Fourier analysis.
static bool        s_midline_hist_enable      = true;
static int         s_midline_hist_interval    = 100;       // write every N timesteps
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

inline double material_body_weight(const double s_norm)
{
    const double xi = clamp01(s_norm);
    const double w_body =
        smoothstep_cosine(xi, material_body_transition_s,
                          material_body_transition_w);
    return w_body;
}

inline double material_caudal_weight(const double s_norm)
{
    const double xi = clamp01(s_norm);
    const double w_caudal =
        smoothstep_cosine(xi, material_caudal_transition_s,
                          material_caudal_transition_w);
    return w_caudal;
}

inline double get_target_bending_B_local(const double s_norm)
{
    const double w_body = material_body_weight(s_norm);
    const double w_caudal = material_caudal_weight(s_norm);
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
// Reference geometry from the true boundary centerline:
//   1. Extract the closed boundary loop directly from the mesh boundary.
//   2. Split it into upper/lower head-to-tail boundary chains.
//   3. Resample both chains with cubic-Hermite interpolation.
//   4. Reconstruct the centerline as the midpoint curve between the two
//      resampled chains, then rebuild h(s) from point-to-centerline projection.
//
// This removes the x-bin seed entirely: the centerline source is now the real
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
        TBOX_ERROR("truncate_centerline_nodes_at_x(): truncated centerline is degenerate.\n");
    }

    ref_centerline_nodes.swap(truncated);
    ref_centerline_end_x = ref_centerline_nodes.back()(0);

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
        const bool increasing_x = ref_profile_x.back() >= ref_profile_x.front();
        if (increasing_x)
        {
            if (x_query <= ref_profile_x.front()) return 0.0;
            if (x_query >= ref_profile_x.back())
                return clamp01(ref_profile_s.back() / std::max(ref_arc_length, 1.0e-12));
        }
        else
        {
            if (x_query >= ref_profile_x.front()) return 0.0;
            if (x_query <= ref_profile_x.back())
                return clamp01(ref_profile_s.back() / std::max(ref_arc_length, 1.0e-12));
        }

        const auto it = increasing_x ?
            std::lower_bound(ref_profile_x.begin(), ref_profile_x.end(), x_query) :
            std::lower_bound(ref_profile_x.begin(), ref_profile_x.end(), x_query,
                             [](const double a, const double b)
                             {
                                 return a > b;
                             });
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
        TBOX_ERROR("active_end_s_norm_effective(): reference centerline end was not "
                   "mapped through the Laplace reference field.\n");
        return 0.0;
    }
    return approximate_s_norm_from_x(ref_centerline_end_x);
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

void build_reference_centerline_from_boundary(MeshBase& mesh)
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
        TBOX_ERROR("build_reference_centerline_from_boundary(): no boundary edges found.\n");
    }

    for (const auto& entry : boundary_adjacency)
    {
        if (entry.second.size() != 2)
        {
            TBOX_ERROR("build_reference_centerline_from_boundary(): boundary is not a simple closed loop.\n");
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
        TBOX_ERROR("build_reference_centerline_from_boundary(): failed to identify head/tail boundary nodes.\n");
    }
    ref_tail_tip_center_point =
        libMesh::Point(tail_x, 0.5 * (tail_upper_y + tail_lower_y));

    const double centerline_end_x_requested =
        std::isfinite(reference_centerline_end_x) ? reference_centerline_end_x :
        (std::isfinite(detected_midline_tail_x) ? detected_midline_tail_x : tail_x);
    const double x_body_lo = std::min(head_x, tail_x);
    const double x_body_up = std::max(head_x, tail_x);
    const double centerline_end_x =
        std::min(std::max(centerline_end_x_requested, x_body_lo), x_body_up);

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

    truncate_centerline_nodes_at_x(centerline_end_x);
}

void build_reference_centerline_segments()
{
    const int n = static_cast<int>(ref_centerline_nodes.size());
    if (n < 2)
    {
        TBOX_ERROR("build_reference_centerline_segments(): need at least 2 centerline nodes.\n");
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
            TBOX_ERROR("build_reference_centerline_segments(): degenerate centerline segment.\n");
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
    if (equation_systems->has_system(REF_GEOM_SYSTEM_NAME)) return;

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
            if (node(0) > ref_centerline_end_x + x_tol) continue;
        }
        else
        {
            if (node(0) < ref_centerline_end_x - x_tol) continue;
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

    // Convert the reference centerline end from Euclidean x to harmonic s/phi once.
    const double x_cut = ref_centerline_end_x;
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
                   "found on REFERENCE_CENTERLINE_END_X = " << x_cut << ".\n");
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
    build_reference_centerline_from_boundary(mesh);
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

inline ActiveCrossSectionMode
parse_active_cross_section_mode(const std::string& mode_raw)
{
    const std::string mode = normalize_mode_string(mode_raw);
    if (mode == "LINEAR-ETA") return ActiveCrossSectionMode::LINEAR_ETA;
    if (mode == "MUSCLE-BAND") return ActiveCrossSectionMode::MUSCLE_BAND;

    TBOX_ERROR("Unknown ACTIVE_CROSS_SECTION_MODE = \"" << mode_raw
               << "\". Expected \"LINEAR_ETA\" or \"MUSCLE_BAND\".\n");
    return ActiveCrossSectionMode::MUSCLE_BAND;
}

inline const char* active_cross_section_mode_name()
{
    switch (active_cross_section_mode)
    {
    case ActiveCrossSectionMode::LINEAR_ETA:
        return "LINEAR_ETA";
    case ActiveCrossSectionMode::MUSCLE_BAND:
        return "MUSCLE_BAND";
    }
    return "MUSCLE_BAND";
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
        return "1-cos(2*pi*xi)";
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
        // Xu, Zhou & Yu (2024) Eq. (2): [1 - cos(2*pi*s/L)].
        return 1.0 - std::cos(2.0 * M_PI * xi);
    }
    return 0.5 * (1.0 - std::cos(M_PI * xi));
}

inline double muscle_moment_drive_from_s(const double s, const double time)
{
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double xi = active_xi_from_s_norm(s_norm);
    return muscle_moment_shape_from_xi(xi) *
           std::cos(active_phase_angle_from_s(s, time));
}

inline double muscle_moment_drive_amplitude_from_s(const double s)
{
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    const double xi = active_xi_from_s_norm(s_norm);
    return muscle_moment_shape_from_xi(xi);
}

inline double longitudinal_active_envelope(double s_norm);

inline double active_moment_envelope_from_s_norm(const double s_norm)
{
    return longitudinal_active_envelope(s_norm);
}

inline double active_moment_prefactor_from_sample(const double s_norm,
                                                  const double h,
                                                  const double time)
{
    const double env_s = active_moment_envelope_from_s_norm(s_norm);
    if (env_s <= 0.0) return 0.0;
    const double w_local = std::max(h, 0.0);
    return wave_ramp(time) * env_s * beta_act * w_local * w_local;
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

// =========================================================================
// Full physical material PK1 stresses
//
// Matrix:
//   W_iso = mu(s)/2 [J^(-2/d) I1 - d]
//   W_vol = K(s)/2 [ln J]^2
//
// The local effective Young modulus is calibrated by E(s) I(s) = B(s), with
// I(s) = 2 h(s)^3 / 3 for a 2D unit-depth section. The 2D shear/area moduli
// are chosen so a traction-free uniaxial test recovers E and nu.
// =========================================================================
struct MaterialProperties
{
    double B = 0.0;
    double I2 = 0.0;
    double E = 0.0;
    double mu = 0.0;
    double K = 0.0;
    double kf = 0.0;
};

inline double section_second_moment(const double halfthickness)
{
    const double h = std::max(halfthickness, 0.0);
    return (2.0 / 3.0) * h * h * h;
}

inline double section_second_moment_floor()
{
    return std::max(section_i2_floor_ratio * section_second_moment(ref_h_max),
                    1.0e-18);
}

inline int active_section_bin_from_s(const double s)
{
    const int n_bins = std::max(active_section_bins, 1);
    const double s_norm = clamp01(s / std::max(ref_arc_length, 1.0e-12));
    return std::min(static_cast<int>(s_norm * n_bins), n_bins - 1);
}

inline double active_cross_section_shape(const double eta, const double h)
{
    if (h <= 1.0e-14) return 0.0;
    const double z = std::max(-1.0, std::min(eta / h, 1.0));
    if (active_cross_section_mode == ActiveCrossSectionMode::LINEAR_ETA)
    {
        return z;
    }

    const double band_fraction =
        std::max(1.0e-6, std::min(active_band_fraction, 1.0));
    const double band_coordinate =
        (std::abs(z) - (1.0 - band_fraction)) / band_fraction;
    const double magnitude = smoothstep(band_coordinate);
    return z >= 0.0 ? magnitude : -magnitude;
}

inline bool
interpolate_active_section_normalization(
    const double s,
    ActiveSectionNormalization& normalization)
{
    if (active_section_normalization.empty()) return false;
    const int n_bins = static_cast<int>(active_section_normalization.size());
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
    const ActiveSectionNormalization& n0 =
        active_section_normalization[static_cast<std::size_t>(bin0)];
    const ActiveSectionNormalization& n1 =
        active_section_normalization[static_cast<std::size_t>(bin1)];
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
    normalization.area = lerp(n0.area, n1.area);
    normalization.eta_mean = lerp(n0.eta_mean, n1.eta_mean);
    normalization.I2 = lerp(n0.I2, n1.I2);
    normalization.g_mean = lerp(n0.g_mean, n1.g_mean);
    normalization.q_scale = lerp(n0.q_scale, n1.q_scale);
    normalization.q_abs_max =
        std::abs(normalization.q_scale) *
        std::max(std::abs(-1.0 - normalization.g_mean),
                 std::abs(1.0 - normalization.g_mean));
    normalization.valid = true;
    return true;
}

inline double active_section_q(const ReferenceGeometrySample& ref_geom)
{
    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double h = body_halfthick_from_s(s);
    ActiveSectionNormalization normalization;
    if (interpolate_active_section_normalization(s, normalization))
    {
        const double g = active_cross_section_shape(ref_geom.eta, h);
        return normalization.q_scale * (g - normalization.g_mean);
    }

    const double I2_use =
        std::max(section_second_moment(h), section_second_moment_floor());
    return -ref_geom.eta / I2_use;
}

inline double active_section_q_abs_max(const ReferenceGeometrySample& ref_geom)
{
    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    ActiveSectionNormalization normalization;
    if (interpolate_active_section_normalization(s, normalization))
        return normalization.q_abs_max;

    const double h = body_halfthick_from_s(s);
    const double I2_use =
        std::max(section_second_moment(h), section_second_moment_floor());
    return h / I2_use;
}

inline double active_stress_cap_from_s(const double s)
{
    double cap = active_t_max_abs > 0.0 ?
        active_t_max_abs : std::numeric_limits<double>::infinity();
    if (active_t_max_over_E > 0.0)
    {
        const double s_norm =
            clamp01(s / std::max(ref_arc_length, 1.0e-12));
        const double I2 =
            std::max(section_second_moment(body_halfthick_from_s(s)),
                     section_second_moment_floor());
        const double E = get_target_bending_B_local(s_norm) / I2;
        cap = std::min(cap, active_t_max_over_E * E);
    }
    return cap;
}

inline double active_uniform_cap_scale(const double moment,
                                       const double q_abs_max,
                                       const double s)
{
    if (q_abs_max <= 0.0) return 1.0;
    const double stress_cap = active_stress_cap_from_s(s);
    if (!std::isfinite(stress_cap)) return 1.0;
    if (stress_cap <= 0.0) return 0.0;
    const double ratio = std::abs(moment) * q_abs_max / stress_cap;
    if (ratio <= 1.0e-12) return 1.0;
    return std::tanh(ratio) / ratio;
}

inline double
active_section_moment_command(const ReferenceGeometrySample& ref_geom,
                              const double time)
{
    const double s_local =
        std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double s_norm =
        clamp01(s_local / std::max(ref_arc_length, 1.0e-12));
    const double h = body_halfthick_from_s(s_local);
    return active_moment_value_from_sample(s_local, s_norm, h, time);
}

inline MaterialProperties material_properties_from_s(const double s)
{
    const double Lref = std::max(ref_arc_length, 1.0e-12);
    const double s_norm = clamp01(s / Lref);
    MaterialProperties props;
    props.B = get_target_bending_B_local(s_norm);
    props.I2 = std::max(section_second_moment(body_halfthick_from_s(s)),
                        section_second_moment_floor());
    props.E = props.B / props.I2;
    // This is a two-dimensional constitutive law:
    //   sigma = 2*mu*dev_2(epsilon) + K*tr(epsilon)*I.
    // These moduli make a traction-free uniaxial test recover the configured
    // E and nu, so the small-strain section bending rigidity is E*I = B.
    props.mu = props.E / (2.0 * (1.0 + material_nu_eff));
    props.K = props.E / (2.0 * (1.0 - material_nu_eff));
    props.kf = fiber_stiffness_ratio * props.E;
    return props;
}

static void
write_material_profile_diagnostics()
{
    if (!s_material_profile_diag_enable) return;
    if (IBTK_MPI::getRank() != 0) return;

    const int n_samples = std::max(s_material_profile_diag_samples, 2);
    std::ofstream out(s_material_profile_diag_filename.c_str(), std::ios::out);
    if (!out.is_open())
    {
        TBOX_WARNING("write_material_profile_diagnostics(): cannot open "
                     << s_material_profile_diag_filename << "\n");
        return;
    }

    out << "sample,s_norm,s,h,I2_real,I2_floor,I2_used"
        << ",B_target,E_used,B_eff,B_eff_over_B_target\n";
    out.setf(std::ios::scientific);
    out.precision(10);

    const double Lref = std::max(ref_arc_length, 1.0e-12);
    const double I2_floor = section_second_moment_floor();
    for (int k = 0; k < n_samples; ++k)
    {
        const double s_norm =
            (n_samples > 1) ?
            static_cast<double>(k) / static_cast<double>(n_samples - 1) : 0.0;
        const double s = s_norm * Lref;
        const double h = body_halfthick_from_s(s);
        const double I2_real = section_second_moment(h);
        const double I2_used = std::max(I2_real, I2_floor);
        const double B_target = get_target_bending_B_local(s_norm);
        const double E_used = B_target / std::max(I2_used, 1.0e-30);
        const double B_eff = E_used * I2_real;
        const double B_eff_over_B_target =
            B_target > 1.0e-30 ?
            B_eff / B_target :
            std::numeric_limits<double>::quiet_NaN();

        out << k
            << "," << s_norm
            << "," << s
            << "," << h
            << "," << I2_real
            << "," << I2_floor
            << "," << I2_used
            << "," << B_target
            << "," << E_used
            << "," << B_eff
            << "," << B_eff_over_B_target
            << "\n";
    }
}

static void
initialize_active_section_normalization(MeshBase& mesh,
                                        EquationSystems* equation_systems)
{
    if (!equation_systems)
    {
        TBOX_ERROR("initialize_active_section_normalization(): null equation systems.\n");
    }

    const int n_bins = std::max(active_section_bins, 1);
    active_section_bins = n_bins;
    const double ds = std::max(ref_arc_length, 1.0e-12) / n_bins;
    const int n_fields = 5;
    std::vector<double> moments(static_cast<std::size_t>(n_bins * n_fields), 0.0);

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
        Utility::string_to_enum<libMesh::Order>(active_section_quad_order);
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
            const int bin = active_section_bin_from_s(s);
            const double h = body_halfthick_from_s(s);
            const double g = active_cross_section_shape(eta, h);
            // Coarea weighting converts a thin s-bin volume integral into
            // the corresponding physical iso-s section integral.
            const double grad_s_norm =
                std::sqrt(std::max(grad_s * grad_s, 0.0));
            const double weight = grad_s_norm * JxW[qp];
            const std::size_t offset =
                static_cast<std::size_t>(bin * n_fields);
            moments[offset + 0] += weight;
            moments[offset + 1] += g * weight;
            moments[offset + 2] += eta * weight;
            moments[offset + 3] += g * eta * weight;
            moments[offset + 4] += eta * eta * weight;
        }
    }

    IBTK_MPI::sumReduction(moments.data(), static_cast<int>(moments.size()));
    active_section_normalization.assign(
        static_cast<std::size_t>(n_bins), ActiveSectionNormalization());

    int invalid_bins = 0;
    double max_unit_force = 0.0;
    double max_unit_moment_error = 0.0;

    struct InvalidBinRecord
    {
        int    bin;
        double s_norm;
        double area;
        double centered_g_eta;
        const char* reason;
    };
    std::vector<InvalidBinRecord> invalid_bin_records;

    for (int bin = 0; bin < n_bins; ++bin)
    {
        const std::size_t offset = static_cast<std::size_t>(bin * n_fields);
        const double area = moments[offset + 0];
        const double g_integral = moments[offset + 1];
        const double eta_integral = moments[offset + 2];
        const double g_eta_integral = moments[offset + 3];
        const double eta2_integral = moments[offset + 4];

        ActiveSectionNormalization& normalization =
            active_section_normalization[static_cast<std::size_t>(bin)];
        normalization.s_mid = (bin + 0.5) * ds;
        normalization.ds = ds;
        normalization.area = area / ds;
        if (area <= 1.0e-18)
        {
            ++invalid_bins;
            invalid_bin_records.push_back(
                { bin,
                  normalization.s_mid / std::max(ref_arc_length, 1.0e-12),
                  area, 0.0, "empty (area=0)" });
            continue;
        }

        normalization.g_mean = g_integral / area;
        normalization.eta_mean = eta_integral / area;
        normalization.I2 =
            (eta2_integral - eta_integral * eta_integral / area) / ds;
        const double centered_g_eta =
            g_eta_integral - normalization.g_mean * eta_integral;
        if (std::abs(centered_g_eta) <= 1.0e-18)
        {
            ++invalid_bins;
            invalid_bin_records.push_back(
                { bin,
                  normalization.s_mid / std::max(ref_arc_length, 1.0e-12),
                  area, centered_g_eta, "degenerate (centered_g_eta=0)" });
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
            invalid_bin_records.push_back(
                { bin,
                  normalization.s_mid / std::max(ref_arc_length, 1.0e-12),
                  area, centered_g_eta, "non-finite q_scale" });
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
        std::ostringstream oss;
        oss << "Active section FE normalization has " << invalid_bins
            << " invalid/empty bins; those bins use the analytic -eta/I2 fallback.\n"
            << "  Invalid bin details (bin, s_norm, area, centered_g_eta, reason):\n";
        for (const InvalidBinRecord& rec : invalid_bin_records)
        {
            oss << "    bin=" << rec.bin
                << "  s_norm=" << rec.s_norm
                << "  area=" << rec.area
                << "  centered_g_eta=" << rec.centered_g_eta
                << "  [" << rec.reason << "]\n";
        }
        TBOX_WARNING(oss.str());
    }
    pout << "  active section FE normalization: bins=" << n_bins
         << ", invalid=" << invalid_bins
         << ", max |unit force|=" << max_unit_force
         << ", max |unit moment-1|=" << max_unit_moment_error << "\n";
}

inline void check_deformation_jacobian(const TensorValue<double>& FF,
                                       const libMesh::Point& X_ref,
                                       const char* caller)
{
    const double J = FF.det();
    if (!(J > 1.0e-14) || !std::isfinite(J))
    {
        TBOX_ERROR(caller << ": non-positive or non-finite det(F) = " << J
                   << " at X_ref=(" << X_ref(0) << ", " << X_ref(1)
                   << "). Mesh is excessively distorted.\n");
    }
}

static void
compute_matrix_PK1_stress_impl(TensorValue<double>& PP,
                               const TensorValue<double>& FF,
                               const libMesh::Point& X_ref,
                               const ReferenceGeometrySample& ref_geom)
{
    check_deformation_jacobian(FF, X_ref, "compute_matrix_PK1_stress_impl()");
    const MaterialProperties props = material_properties_from_s(ref_geom.s);
    const double J = FF.det();
    const double d = static_cast<double>(NDIM);
    const TensorValue<double> FinvT = tensor_inverse_transpose(FF, NDIM);
    double I1 = 0.0;
    for (unsigned int i = 0; i < NDIM; ++i)
        for (unsigned int j = 0; j < NDIM; ++j)
            I1 += FF(i, j) * FF(i, j);

    PP = props.mu * std::pow(J, -2.0 / d) *
             (FF - (I1 / d) * FinvT) +
         props.K * std::log(J) * FinvT;
}

static void
compute_fiber_PK1_stress_impl(TensorValue<double>& PP,
                              const TensorValue<double>& FF,
                              const libMesh::Point& X_ref,
                              const ReferenceGeometrySample& ref_geom)
{
    PP = 0.0;
    if (fiber_stiffness_ratio <= 0.0) return;
    check_deformation_jacobian(FF, X_ref, "compute_fiber_PK1_stress_impl()");

    const VectorValue<double> f0 = ref_geom.t_hat;
    const VectorValue<double> Ff0 = FF * f0;
    const double I4_minus_1 = Ff0 * Ff0 - 1.0;
    if (I4_minus_1 <= 0.0) return;

    const MaterialProperties props = material_properties_from_s(ref_geom.s);
    TensorValue<double> f0_f0;
    outer_product(f0_f0, f0, f0);
    PP = 2.0 * props.kf * I4_minus_1 * FF * f0_f0;
}

static void
compute_structural_damping_PK1_stress_impl(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point& X_ref,
    const ReferenceGeometrySample& ref_geom,
    const TensorValue<double>& F_dot)
{
    PP = 0.0;
    if (structural_kv_loss_factor <= 0.0) return;
    check_deformation_jacobian(
        FF, X_ref, "compute_structural_damping_PK1_stress_impl()");

    const double J = FF.det();
    const TensorValue<double> F_inv_trans =
        tensor_inverse_transpose(FF, NDIM);
    const TensorValue<double> F_inv = F_inv_trans.transpose();
    const TensorValue<double> L = F_dot * F_inv;
    const TensorValue<double> D = 0.5 * (L + L.transpose());

    const VectorValue<double> a = FF * ref_geom.t_hat;
    const double a_norm = std::sqrt(std::max(a * a, 0.0));
    if (a_norm <= 1.0e-14) return;
    const VectorValue<double> a_hat = a / a_norm;

    const double axial_strain_rate = a_hat * (D * a_hat);
    const MaterialProperties props = material_properties_from_s(ref_geom.s);
    const double omega_ref = std::max(std::abs(wave_omega), 1.0e-12);
    // eta_s = loss_factor*E/omega gives D_bend =
    // loss_factor*B/omega in the small-strain beam limit.
    const double eta_s =
        structural_kv_loss_factor * props.E / omega_ref;
    const double T_raw = eta_s * axial_strain_rate;
    const double T_cap = structural_kv_stress_cap_over_E > 0.0 ?
        structural_kv_stress_cap_over_E * props.E :
        std::numeric_limits<double>::infinity();
    const double T_damping = std::isfinite(T_cap) ?
        std::max(-T_cap, std::min(T_cap, T_raw)) : T_raw;

    TensorValue<double> aa;
    outer_product(aa, a_hat, a_hat);
    PP = J * (T_damping * aa) * F_inv_trans;
}

static void
compute_active_PK1_stress_impl(TensorValue<double>& PP,
                               const TensorValue<double>& FF,
                               const libMesh::Point& X_ref,
                               const ReferenceGeometrySample& ref_geom,
                               const double time)
{
    PP = 0.0;
    if (active_moment_mode == ActiveMomentMode::TRAVELING && beta_act <= 0.0) return;
    if (active_moment_mode == ActiveMomentMode::STATIC &&
        std::abs(static_moment_m0) <= 1.0e-30) return;
    check_deformation_jacobian(FF, X_ref, "compute_active_PK1_stress_impl()");

    const double s = std::max(0.0, std::min(ref_geom.s, ref_arc_length));
    const double h = body_halfthick_from_s(s);
    if (h <= 1.0e-12) return;

    const double Ma = active_section_moment_command(ref_geom, time);
    if (std::abs(Ma) <= 1.0e-30) return;

    const double q = active_section_q(ref_geom);
    const double cap_scale =
        active_uniform_cap_scale(Ma, active_section_q_abs_max(ref_geom), s);
    const double T_active = cap_scale * Ma * q;

    const VectorValue<double> f0 = ref_geom.t_hat;
    TensorValue<double> f0_f0;
    outer_product(f0_f0, f0, f0);
    PP = T_active * FF * f0_f0;
}

void PK1_matrix_stress_function(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point&,
    const libMesh::Point& X_ref,
    Elem* const,
    const std::vector<const std::vector<double>*>& system_var_data,
    const std::vector<const std::vector<VectorValue<double> >*>&,
    double,
    void*)
{
    const ReferenceGeometrySample ref_geom =
        reference_geometry_from_system_data(system_var_data);
    compute_matrix_PK1_stress_impl(PP, FF, X_ref, ref_geom);
}

void PK1_fiber_stress_function(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point&,
    const libMesh::Point& X_ref,
    Elem* const,
    const std::vector<const std::vector<double>*>& system_var_data,
    const std::vector<const std::vector<VectorValue<double> >*>&,
    double,
    void*)
{
    const ReferenceGeometrySample ref_geom =
        reference_geometry_from_system_data(system_var_data);
    compute_fiber_PK1_stress_impl(PP, FF, X_ref, ref_geom);
}

void PK1_structural_damping_stress_function(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point&,
    const libMesh::Point& X_ref,
    Elem* const,
    const std::vector<const std::vector<double>*>& system_var_data,
    const std::vector<const std::vector<VectorValue<double> >*>&
        system_grad_var_data,
    double,
    void*)
{
    PP = 0.0;
    if (structural_kv_loss_factor <= 0.0) return;
    if (system_grad_var_data.size() < 2 ||
        system_grad_var_data[1] == nullptr ||
        system_grad_var_data[1]->size() < NDIM)
    {
        TBOX_ERROR("PK1_structural_damping_stress_function(): "
                   "velocity-gradient data are unavailable.\n");
    }

    const ReferenceGeometrySample ref_geom =
        reference_geometry_from_system_data(system_var_data);
    const std::vector<VectorValue<double> >& grad_U =
        *system_grad_var_data[1];
    TensorValue<double> F_dot;
    F_dot.zero();
    for (unsigned int i = 0; i < NDIM; ++i)
        for (unsigned int j = 0; j < NDIM; ++j)
            F_dot(i, j) = grad_U[i](j);
    compute_structural_damping_PK1_stress_impl(
        PP, FF, X_ref, ref_geom, F_dot);
}

void PK1_active_stress_function(
    TensorValue<double>& PP,
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
    if (!equation_systems)
    {
        x_cm_out = xcom_tracked;
        y_cm_out = ycom_tracked;
        if (area_out) *area_out = 0.0;
        return;
    }

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
        boundary_x_end : ref_centerline_end_x;
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

static std::vector<double>
compute_body_midline_curvature_tangent_angle(const std::vector<MidlineSample>& samples)
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

static bool
solve_small_linear_system(std::vector<double> A,
                          std::vector<double> b,
                          std::vector<double>& x)
{
    const int n = static_cast<int>(b.size());
    if (A.size() != static_cast<std::size_t>(n * n)) return false;

    for (int col = 0; col < n; ++col)
    {
        int pivot = col;
        double pivot_abs = std::abs(A[static_cast<std::size_t>(col * n + col)]);
        for (int row = col + 1; row < n; ++row)
        {
            const double value_abs =
                std::abs(A[static_cast<std::size_t>(row * n + col)]);
            if (value_abs > pivot_abs)
            {
                pivot = row;
                pivot_abs = value_abs;
            }
        }
        if (pivot_abs <= 1.0e-18 || !std::isfinite(pivot_abs)) return false;

        if (pivot != col)
        {
            for (int j = col; j < n; ++j)
            {
                std::swap(A[static_cast<std::size_t>(col * n + j)],
                          A[static_cast<std::size_t>(pivot * n + j)]);
            }
            std::swap(b[static_cast<std::size_t>(col)],
                      b[static_cast<std::size_t>(pivot)]);
        }

        const double diag = A[static_cast<std::size_t>(col * n + col)];
        for (int row = col + 1; row < n; ++row)
        {
            const double factor =
                A[static_cast<std::size_t>(row * n + col)] / diag;
            A[static_cast<std::size_t>(row * n + col)] = 0.0;
            for (int j = col + 1; j < n; ++j)
            {
                A[static_cast<std::size_t>(row * n + j)] -=
                    factor * A[static_cast<std::size_t>(col * n + j)];
            }
            b[static_cast<std::size_t>(row)] -=
                factor * b[static_cast<std::size_t>(col)];
        }
    }

    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int row = n - 1; row >= 0; --row)
    {
        double rhs = b[static_cast<std::size_t>(row)];
        for (int j = row + 1; j < n; ++j)
        {
            rhs -= A[static_cast<std::size_t>(row * n + j)] *
                   x[static_cast<std::size_t>(j)];
        }
        const double diag = A[static_cast<std::size_t>(row * n + row)];
        if (std::abs(diag) <= 1.0e-18 || !std::isfinite(diag)) return false;
        x[static_cast<std::size_t>(row)] = rhs / diag;
    }
    return true;
}

static bool
fit_local_polynomial_coefficients(const std::vector<double>& z,
                                  const std::vector<double>& values,
                                  const int                  order,
                                  std::vector<double>&       coeffs)
{
    const int m = static_cast<int>(z.size());
    if (m != static_cast<int>(values.size()) || m <= order || order < 2)
        return false;

    const int n = order + 1;
    std::vector<double> A(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> b(static_cast<std::size_t>(n), 0.0);
    std::vector<double> powers(static_cast<std::size_t>(2 * order + 1), 1.0);

    for (int row = 0; row < m; ++row)
    {
        powers[0] = 1.0;
        for (int p = 1; p <= 2 * order; ++p)
        {
            powers[static_cast<std::size_t>(p)] =
                powers[static_cast<std::size_t>(p - 1)] *
                z[static_cast<std::size_t>(row)];
        }
        for (int p = 0; p <= order; ++p)
        {
            b[static_cast<std::size_t>(p)] +=
                values[static_cast<std::size_t>(row)] *
                powers[static_cast<std::size_t>(p)];
            for (int q = 0; q <= order; ++q)
            {
                A[static_cast<std::size_t>(p * n + q)] +=
                    powers[static_cast<std::size_t>(p + q)];
            }
        }
    }

    return solve_small_linear_system(A, b, coeffs);
}

static std::vector<double>
compute_body_midline_curvature_local_poly(const std::vector<MidlineSample>& samples)
{
    const int n = static_cast<int>(samples.size());
    std::vector<double> kappa =
        compute_body_midline_curvature_tangent_angle(samples);
    if (n < 3) return kappa;

    const int window = 7;
    const int order_requested = 3;

    for (int i = 0; i < n; ++i)
    {
        const int use_window = std::min(window, n);
        int left = i - use_window / 2;
        left = std::max(0, std::min(left, n - use_window));
        const int right = left + use_window - 1;
        const int m = right - left + 1;
        const int order = std::min(order_requested, m - 1);
        if (order < 2) continue;

        const double s_center = samples[static_cast<std::size_t>(i)].s;
        double scale = 0.0;
        for (int j = left; j <= right; ++j)
        {
            scale = std::max(scale,
                             std::abs(samples[static_cast<std::size_t>(j)].s -
                                      s_center));
        }
        if (!(scale > 1.0e-14) || !std::isfinite(scale)) continue;

        std::vector<double> z;
        std::vector<double> x_values;
        std::vector<double> y_values;
        z.reserve(static_cast<std::size_t>(m));
        x_values.reserve(static_cast<std::size_t>(m));
        y_values.reserve(static_cast<std::size_t>(m));
        for (int j = left; j <= right; ++j)
        {
            const MidlineSample& sample = samples[static_cast<std::size_t>(j)];
            z.push_back((sample.s - s_center) / scale);
            x_values.push_back(sample.x_body);
            y_values.push_back(sample.y_body);
        }

        std::vector<double> x_coeffs;
        std::vector<double> y_coeffs;
        if (!fit_local_polynomial_coefficients(z, x_values, order, x_coeffs) ||
            !fit_local_polynomial_coefficients(z, y_values, order, y_coeffs))
        {
            continue;
        }

        const double dx_ds = x_coeffs[1] / scale;
        const double dy_ds = y_coeffs[1] / scale;
        const double d2x_ds2 = 2.0 * x_coeffs[2] / (scale * scale);
        const double d2y_ds2 = 2.0 * y_coeffs[2] / (scale * scale);
        const double speed2 = dx_ds * dx_ds + dy_ds * dy_ds;
        const double denom = speed2 * std::sqrt(std::max(speed2, 0.0));
        if (denom <= 1.0e-30 || !std::isfinite(denom)) continue;

        const double value =
            (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denom;
        if (std::isfinite(value))
        {
            kappa[static_cast<std::size_t>(i)] = value;
        }
    }

    return kappa;
}

std::vector<double> compute_body_midline_curvature(const std::vector<MidlineSample>& samples)
{
    return compute_body_midline_curvature_local_poly(samples);
}

static std::vector<double>
compute_reference_midline_curvature_at_s(const std::vector<double>& s_values)
{
    std::vector<MidlineSample> ref_samples(s_values.size());
    for (std::size_t k = 0; k < s_values.size(); ++k)
    {
        const ReferenceFrame frame = reference_frame_at_s(s_values[k]);
        MidlineSample& sample = ref_samples[k];
        sample.s = s_values[k];
        sample.s_norm = clamp01(s_values[k] / std::max(ref_arc_length, 1.0e-12));
        sample.ref = frame;
        sample.x_body = frame.X(0);
        sample.y_body = frame.X(1);
    }

    std::vector<double> kappa_ref =
        compute_body_midline_curvature(ref_samples);
    for (double& value : kappa_ref)
    {
        if (!std::isfinite(value)) value = 0.0;
    }
    return kappa_ref;
}

// =========================================================================
// Force decomposition diagnostic
// =========================================================================
struct ForceDecompAccumulator
{
    double Fx = 0.0;
    double Fy = 0.0;
    double L1 = 0.0;
    double abs_x = 0.0;
    double abs_y = 0.0;
    double P = 0.0;
    double P_abs = 0.0;
    double P_weak = 0.0;
    double P_weak_abs = 0.0;
};

enum ForceDecompComponent
{
    FORCE_MATRIX = 0,
    FORCE_FIBER = 1,
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
    case FORCE_MATRIX: return "matrix";
    case FORCE_FIBER:  return "fiber";
    case FORCE_DAMPING: return "damping";
    case FORCE_ACTIVE: return "active";
    case FORCE_SUM:    return "sum";
    default:           return "unknown";
    }
}

static void
write_force_decomposition_diagnostics(const int            iteration_num,
                                      const double         loop_time,
                                      Pointer<IBFEMethod>  ib_method_ops,
                                      MeshBase&            mesh,
                                      EquationSystems*     equation_systems,
                                      const double         x_cm,
                                      const double         y_cm,
                                      const double         vcm_x,
                                      const double         vcm_y)
{
    if (!s_force_decomp_diag_enable) return;
    if (s_force_decomp_diag_interval > 1 &&
        (iteration_num % s_force_decomp_diag_interval != 0)) return;
    if (!equation_systems) return;

    const unsigned int dim = mesh.mesh_dimension();

    System& X_sys = equation_systems->get_system(
        ib_method_ops->getCurrentCoordinatesSystemName());
    NumericVector<double>* X_vec = X_sys.solution.get();
    NumericVector<double>* X_ghost_vec = X_sys.current_local_solution.get();
    X_vec->close();
    copy_and_synch(*X_vec, *X_ghost_vec);

    System& U_sys = equation_systems->get_system(
        ib_method_ops->getVelocitySystemName());
    NumericVector<double>* U_vec = U_sys.solution.get();
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

    auto accumulate_power =
        [&](const int c, const double p)
        {
            acc[c].P += p;
            acc[c].P_abs += std::abs(p);
            acc[FORCE_SUM].P += p;
            acc[FORCE_SUM].P_abs += std::abs(p);
        };

    auto add_node_load =
        [&](const int c,
            const unsigned int k,
            const VectorValue<double>& f,
            std::vector<std::vector<VectorValue<double> > >& elem_loads)
        {
            elem_loads[c][k] += f;
            elem_loads[FORCE_SUM][k] += f;
        };

    auto accumulate_stress_qp =
        [&](const int c,
            const TensorValue<double>& PP,
            const TensorValue<double>& F_dot,
            const double JxW_qp,
            const unsigned int n_nodes,
            const unsigned int qp,
            std::vector<std::vector<VectorValue<double> > >& elem_loads)
        {
            double p = 0.0;
            for (unsigned int i = 0; i < NDIM; ++i)
                for (unsigned int j = 0; j < NDIM; ++j)
                    p += PP(i, j) * F_dot(i, j);
            accumulate_power(c, p * JxW_qp);

            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                VectorValue<double> f_node(0.0, 0.0);
                for (unsigned int i = 0; i < NDIM; ++i)
                {
                    double f_i = 0.0;
                    for (unsigned int j = 0; j < NDIM; ++j)
                        f_i -= PP(i, j) * dphi[k][qp](j) * JxW_qp;
                    f_node(i) = f_i;
                }
                add_node_load(c, k, f_node, elem_loads);
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
        std::vector<std::vector<VectorValue<double> > > elem_loads(
            FORCE_N_COMPONENTS);
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
                        F_dot(i, j) += u_ki * dphi[k][qp](j);
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
                        phi[k][qp] *
                        (*ref_geom_ghost_vec)(ref_geom_dof_indices[v][k]);
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

            TensorValue<double> PP_matrix(0.0), PP_fiber(0.0);
            TensorValue<double> PP_damping(0.0), PP_active(0.0);
            compute_matrix_PK1_stress_impl(PP_matrix, FF, X_ref_qp, ref_geom);
            compute_fiber_PK1_stress_impl(PP_fiber, FF, X_ref_qp, ref_geom);
            compute_structural_damping_PK1_stress_impl(
                PP_damping, FF, X_ref_qp, ref_geom, F_dot);
            compute_active_PK1_stress_impl(PP_active, FF, X_ref_qp,
                                           ref_geom, loop_time);

            accumulate_stress_qp(FORCE_MATRIX, PP_matrix, F_dot, JxW[qp],
                                 n_nodes, qp, elem_loads);
            accumulate_stress_qp(FORCE_FIBER, PP_fiber, F_dot, JxW[qp],
                                 n_nodes, qp, elem_loads);
            accumulate_stress_qp(FORCE_DAMPING, PP_damping, F_dot, JxW[qp],
                                 n_nodes, qp, elem_loads);
            accumulate_stress_qp(FORCE_ACTIVE, PP_active, F_dot, JxW[qp],
                                 n_nodes, qp, elem_loads);
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
                acc[c].abs_x += std::abs(f(0));
                acc[c].abs_y += std::abs(f(1));
                acc[c].P_weak += p_weak;
                acc[c].P_weak_abs += std::abs(p_weak);
            }
        }
    }

    std::vector<double> reduced;
    reduced.reserve(FORCE_N_COMPONENTS * 9);
    for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
    {
        reduced.push_back(acc[c].Fx);
        reduced.push_back(acc[c].Fy);
        reduced.push_back(acc[c].L1);
        reduced.push_back(acc[c].abs_x);
        reduced.push_back(acc[c].abs_y);
        reduced.push_back(acc[c].P);
        reduced.push_back(acc[c].P_abs);
        reduced.push_back(acc[c].P_weak);
        reduced.push_back(acc[c].P_weak_abs);
    }
    IBTK_MPI::sumReduction(reduced.data(), static_cast<int>(reduced.size()));

    std::size_t r = 0;
    for (unsigned int c = 0; c < FORCE_N_COMPONENTS; ++c)
    {
        acc[c].Fx = reduced[r++];
        acc[c].Fy = reduced[r++];
        acc[c].L1 = reduced[r++];
        acc[c].abs_x = reduced[r++];
        acc[c].abs_y = reduced[r++];
        acc[c].P = reduced[r++];
        acc[c].P_abs = reduced[r++];
        acc[c].P_weak = reduced[r++];
        acc[c].P_weak_abs = reduced[r++];
    }

    const double dt_eff = std::isfinite(s_force_decomp_prev_time) ?
        loop_time - s_force_decomp_prev_time :
        std::numeric_limits<double>::quiet_NaN();
    s_force_decomp_prev_time = loop_time;

    double component_sum_x = 0.0;
    double component_sum_y = 0.0;
    for (int c = FORCE_MATRIX; c <= FORCE_ACTIVE; ++c)
    {
        component_sum_x += acc[c].Fx;
        component_sum_y += acc[c].Fy;
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
        out << "step,time,dt_eff,x_cm,y_cm,vcm_x,vcm_y";
        for (int c = FORCE_MATRIX; c <= FORCE_SUM; ++c)
        {
            const char* name = force_decomp_component_name(c);
            out << ",F_" << name << "_x"
                << ",F_" << name << "_y"
                << ",F_L1_" << name
                << ",F_abs_x_" << name
                << ",F_abs_y_" << name
                << ",P_density_" << name
                << ",P_density_abs_" << name
                << ",P_weak_" << name
                << ",P_weak_abs_" << name;
        }
        out << ",sum_check_error_x,sum_check_error_y\n";
        out.flush();
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    out << iteration_num
        << "," << loop_time
        << "," << dt_eff
        << "," << x_cm
        << "," << y_cm
        << "," << vcm_x
        << "," << vcm_y;
    for (int c = FORCE_MATRIX; c <= FORCE_SUM; ++c)
    {
        out << "," << acc[c].Fx
            << "," << acc[c].Fy
            << "," << acc[c].L1
            << "," << acc[c].abs_x
            << "," << acc[c].abs_y
            << "," << acc[c].P
            << "," << acc[c].P_abs
            << "," << acc[c].P_weak
            << "," << acc[c].P_weak_abs;
    }
    out << "," << (acc[FORCE_SUM].Fx - component_sum_x)
        << "," << (acc[FORCE_SUM].Fy - component_sum_y)
        << "\n";
    out.flush();
}

static void
write_active_section_diagnostics(const int        iteration_num,
                                 const double     loop_time,
                                 MeshBase&        mesh,
                                 EquationSystems* equation_systems)
{
    if (!s_active_section_diag_enable) return;
    if (s_active_section_diag_interval > 1 &&
        (iteration_num % s_active_section_diag_interval != 0)) return;
    if (!equation_systems || active_section_normalization.empty()) return;

    const int n_bins = static_cast<int>(active_section_normalization.size());
    const int n_fields = 5;
    std::vector<double> values(static_cast<std::size_t>(n_bins * n_fields), 0.0);

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
        Utility::string_to_enum<libMesh::Order>(active_section_quad_order);
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
            ReferenceGeometrySample ref_geom;
            RealGradient grad_s;
            grad_s.zero();
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                const double s_node =
                    (*ref_geom_ghost_vec)(dof_indices[REF_GEOM_S][k]);
                ref_geom.s += phi[k][qp] * s_node;
                ref_geom.eta +=
                    phi[k][qp] *
                    (*ref_geom_ghost_vec)(dof_indices[REF_GEOM_ETA][k]);
                grad_s.add_scaled(dphi[k][qp], s_node);
            }
            ref_geom.s =
                std::max(0.0, std::min(ref_geom.s, ref_arc_length));
            const int bin = active_section_bin_from_s(ref_geom.s);
            const double moment =
                active_section_moment_command(ref_geom, loop_time);
            const double cap_scale =
                active_uniform_cap_scale(moment,
                                         active_section_q_abs_max(ref_geom),
                                         ref_geom.s);
            const double applied_moment = cap_scale * moment;
            const double stress = applied_moment * active_section_q(ref_geom);
            const double grad_s_norm =
                std::sqrt(std::max(grad_s * grad_s, 0.0));
            const double weight = grad_s_norm * JxW[qp];
            const std::size_t offset =
                static_cast<std::size_t>(bin * n_fields);
            values[offset + 0] += weight;
            values[offset + 1] += stress * weight;
            values[offset + 2] += -stress * ref_geom.eta * weight;
            values[offset + 3] += moment * weight;
            values[offset + 4] += applied_moment * weight;
        }
    }

    IBTK_MPI::sumReduction(values.data(), static_cast<int>(values.size()));
    if (IBTK_MPI::getRank() != 0) return;

    static std::ofstream out;
    if (!out.is_open())
    {
        out.open(s_active_section_diag_filename.c_str(), std::ios::out);
        if (!out.is_open())
        {
            TBOX_WARNING("write_active_section_diagnostics(): cannot open "
                         << s_active_section_diag_filename << "\n");
            return;
        }
        out << "step,time,bin,s_norm,valid,A_section,eta_bar,I2_section"
            << ",g_bar,q_scale,q_abs_max,unit_force,unit_moment"
            << ",N_active,M_active,M_command,M_applied_target"
            << ",M_ratio,N_relative\n";
    }

    out.setf(std::ios::scientific);
    out.precision(10);
    for (int bin = 0; bin < n_bins; ++bin)
    {
        const ActiveSectionNormalization& normalization =
            active_section_normalization[static_cast<std::size_t>(bin)];
        const std::size_t offset = static_cast<std::size_t>(bin * n_fields);
        const double area_volume = values[offset + 0];
        const double ds = std::max(normalization.ds, 1.0e-30);
        const double N_active = values[offset + 1] / ds;
        const double M_active = values[offset + 2] / ds;
        const double M_command = area_volume > 1.0e-30 ?
            values[offset + 3] / area_volume : 0.0;
        const double M_applied = area_volume > 1.0e-30 ?
            values[offset + 4] / area_volume : 0.0;
        const double M_ratio = std::abs(M_applied) > 1.0e-30 ?
            M_active / M_applied :
            std::numeric_limits<double>::quiet_NaN();
        const double h = body_halfthick_from_s(normalization.s_mid);
        const double force_scale =
            std::abs(M_applied) / std::max(h, 1.0e-30);
        const double N_relative = force_scale > 1.0e-30 ?
            std::abs(N_active) / force_scale :
            std::numeric_limits<double>::quiet_NaN();

        out << iteration_num
            << "," << loop_time
            << "," << bin
            << "," << normalization.s_mid /
                         std::max(ref_arc_length, 1.0e-12)
            << "," << (normalization.valid ? 1 : 0)
            << "," << normalization.area
            << "," << normalization.eta_mean
            << "," << normalization.I2
            << "," << normalization.g_mean
            << "," << normalization.q_scale
            << "," << normalization.q_abs_max
            << "," << normalization.unit_force
            << "," << normalization.unit_moment
            << "," << N_active
            << "," << M_active
            << "," << M_command
            << "," << M_applied
            << "," << M_ratio
            << "," << N_relative
            << "\n";
    }
    out.flush();
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
        s_curvature_phase_last_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_last_positive_power = std::numeric_limits<double>::quiet_NaN();
        s_curvature_phase_samples = 0;
    }

    double signed_power = std::numeric_limits<double>::quiet_NaN();
    double positive_power = std::numeric_limits<double>::quiet_NaN();
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
                if (std::isfinite(kappa_body[k]))
                {
                    s_curvature_cos_accum[k] += kappa_body[k] * c_t * dt_eff;
                    s_curvature_sin_accum[k] += kappa_body[k] * s_t * dt_eff;
                }
                if (std::abs(Mm) > 1.0e-30)
                {
                    s_activation_cos_accum[k] += Mm * c_t * dt_eff;
                    s_activation_sin_accum[k] += Mm * s_t * dt_eff;
                }

                if (std::isfinite(kappa_body[k]) &&
                    std::isfinite(s_prev_curvature_body[k]))
                {
                    const double kappa_dot =
                        (kappa_body[k] - s_prev_curvature_body[k]) / dt_eff;
                    const double p_density = Mm * kappa_dot;
                    signed_power += p_density * ds_station;
                    positive_power += std::max(p_density, 0.0) * ds_station;
                }
            }

            s_curvature_phase_accum_time += dt_eff;
            s_curvature_phase_signed_work_accum += signed_power * dt_eff;
            s_curvature_phase_positive_work_accum += positive_power * dt_eff;
            s_curvature_phase_last_power = signed_power;
            s_curvature_phase_last_positive_power = positive_power;
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
            << ",active_moment"
            << ",curvature_phase,activation_phase,phase_lag"
            << ",curvature_phase_unwrapped,curvature_phase_slope"
            << ",active_phase_slope_abs,active_phase_slope_expected"
            << ",traveling_wave_index,signed_traveling_wave_index,drive_following_index"
            << ",theta_body,x_cm,y_cm,y_cm_relative"
            << ",active_power,active_positive_power,active_signed_work,active_positive_work"
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
            << "\n";
    }
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
    std::vector<double> s_values(samples.size(), 0.0);
    for (std::size_t k = 0; k < samples.size(); ++k)
        s_values[k] = samples[k].s;
    std::vector<double> kappa_ref_body =
        compute_reference_midline_curvature_at_s(s_values);

    if (IBTK_MPI::getRank() != 0) return;

    const double ph = phase01(loop_time);

    std::ofstream f(s_midline_hist_filename,
                    s_midline_hist_header_done ? std::ios::app : std::ios::out);
    if (!f.is_open()) return;

    if (!s_midline_hist_header_done)
    {
        f << "time,cycle_phase,station,s_norm"
          << ",s"
          << ",x_lab,y_lab,x_cm,y_cm,theta_body"
          << ",x_body,y_body"   // deformation frame (remove translation + rotation)
          << ",y_prop"          // propulsion pattern: lab-frame lateral motion
          << ",curvature,curvature_ref,curvature_rel\n";
        s_midline_hist_header_done = true;
    }

    f.setf(std::ios::scientific);
    f.precision(8);
    for (int k = 0; k < static_cast<int>(samples.size()); ++k)
    {
        const double kv = (k < static_cast<int>(kappa_body.size()))
            ? kappa_body[static_cast<std::size_t>(k)]
            : std::numeric_limits<double>::quiet_NaN();
        const double kref = (k < static_cast<int>(kappa_ref_body.size()))
            ? kappa_ref_body[static_cast<std::size_t>(k)]
            : std::numeric_limits<double>::quiet_NaN();
        const double krel = (std::isfinite(kv) && std::isfinite(kref))
            ? kv - kref
            : std::numeric_limits<double>::quiet_NaN();
        // Keep lateral recoil/heave in the propulsion pattern; only forward
        // translation is irrelevant for the Y(s,t) matrix used downstream.
        const double y_prop = samples[k].y_lab;
        f << loop_time
          << "," << ph
          << "," << k
          << "," << (samples[k].s / Lref)
          << "," << samples[k].s
          << "," << samples[k].x_lab
          << "," << samples[k].y_lab
          << "," << x_cm
          << "," << y_cm
          << "," << theta_body
          << "," << samples[k].x_body
          << "," << samples[k].y_body
          << "," << y_prop
          << "," << kv
          << "," << kref
          << "," << krel
          << "\n";
    }
}

// =========================================================================
// Retained diagnostics for COM, J positivity, curvature phase, and midline history.
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
    const bool geometry_conservation_due =
        s_geometry_conservation_diag_enable &&
        (s_geometry_conservation_diag_interval <= 1 ||
         (iteration_num % s_geometry_conservation_diag_interval == 0));
    const bool force_decomp_due =
        s_force_decomp_diag_enable &&
        (s_force_decomp_diag_interval <= 1 ||
         (iteration_num % s_force_decomp_diag_interval == 0));
    const bool active_section_due =
        s_active_section_diag_enable &&
        (s_active_section_diag_interval <= 1 ||
         (iteration_num % s_active_section_diag_interval == 0));
    const bool midline_hist_due =
        s_midline_hist_enable &&
        (s_midline_hist_interval <= 1 ||
         (iteration_num % s_midline_hist_interval == 0));
    if (!curvature_phase_due && !geometry_conservation_due &&
        !force_decomp_due && !active_section_due && !midline_hist_due) return;
    if (!equation_systems) return;

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
                                              ib_method_ops, mesh,
                                              equation_systems,
                                              x_cm_new, y_cm_new,
                                              vcm_x, vcm_y);
    }

    if (active_section_due)
    {
        write_active_section_diagnostics(iteration_num, loop_time,
                                         mesh, equation_systems);
    }

    if (geometry_conservation_due)
    {
        write_geometry_conservation_diagnostics(iteration_num, loop_time,
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

        // ── Read full physical material parameters ─────────────────────────
        target_bending_B_body =
            input_db->getDoubleWithDefault("B_BODY", target_bending_B_body);
        target_bending_B_peduncle =
            input_db->getDoubleWithDefault("B_PEDUNCLE", target_bending_B_peduncle);
        target_bending_B_caudal =
            input_db->getDoubleWithDefault("B_CAUDAL", target_bending_B_caudal);
        material_body_transition_s =
            input_db->getDoubleWithDefault("MATERIAL_BODY_TRANSITION_S",
                                           material_body_transition_s);
        material_body_transition_w =
            input_db->getDoubleWithDefault("MATERIAL_BODY_TRANSITION_W",
                                           material_body_transition_w);
        material_caudal_transition_s =
            input_db->getDoubleWithDefault("MATERIAL_CAUDAL_TRANSITION_S",
                                           material_caudal_transition_s);
        material_caudal_transition_w =
            input_db->getDoubleWithDefault("MATERIAL_CAUDAL_TRANSITION_W",
                                           material_caudal_transition_w);
        material_nu_eff =
            input_db->getDoubleWithDefault("MATERIAL_NU_EFF", material_nu_eff);
        section_i2_floor_ratio =
            input_db->getDoubleWithDefault("SECTION_I2_FLOOR_RATIO",
                                           section_i2_floor_ratio);
        fiber_stiffness_ratio =
            input_db->getDoubleWithDefault("FIBER_STIFFNESS_RATIO",
                                           fiber_stiffness_ratio);
        structural_kv_loss_factor =
            input_db->getDoubleWithDefault("STRUCTURAL_KV_LOSS_FACTOR",
                                           structural_kv_loss_factor);
        structural_kv_stress_cap_over_E =
            input_db->getDoubleWithDefault("STRUCTURAL_KV_STRESS_CAP_OVER_E",
                                           structural_kv_stress_cap_over_E);
        active_t_max_abs =
            input_db->getDoubleWithDefault("ACTIVE_T_MAX_ABS",
                                           active_t_max_abs);
        active_t_max_over_E =
            input_db->getDoubleWithDefault("ACTIVE_T_MAX_OVER_E",
                                           active_t_max_over_E);
        active_cross_section_mode =
            parse_active_cross_section_mode(input_db->getStringWithDefault(
                "ACTIVE_CROSS_SECTION_MODE",
                active_cross_section_mode_name()));
        active_band_fraction =
            input_db->getDoubleWithDefault("ACTIVE_BAND_FRACTION",
                                           active_band_fraction);
        active_section_bins =
            input_db->getIntegerWithDefault("ACTIVE_SECTION_BINS",
                                            active_section_bins);
        active_section_quad_order =
            input_db->getStringWithDefault("ACTIVE_SECTION_QUAD_ORDER",
                                           active_section_quad_order);
        s_material_profile_diag_enable =
            input_db->getBoolWithDefault("MATERIAL_PROFILE_DIAG_ENABLE",
                                         s_material_profile_diag_enable);
        s_material_profile_diag_filename =
            input_db->getStringWithDefault("MATERIAL_PROFILE_DIAG_FILENAME",
                                           s_material_profile_diag_filename);
        s_material_profile_diag_samples =
            input_db->getIntegerWithDefault("MATERIAL_PROFILE_DIAG_SAMPLES",
                                            s_material_profile_diag_samples);

        target_bending_B_body = std::max(0.0, target_bending_B_body);
        target_bending_B_peduncle = std::max(0.0, target_bending_B_peduncle);
        target_bending_B_caudal = std::max(0.0, target_bending_B_caudal);
        material_body_transition_s = clamp01(material_body_transition_s);
        material_body_transition_w =
            std::max(material_body_transition_w, 1.0e-12);
        material_caudal_transition_s = clamp01(material_caudal_transition_s);
        material_caudal_transition_w =
            std::max(material_caudal_transition_w, 1.0e-12);
        material_nu_eff =
            std::max(0.0, std::min(material_nu_eff, 0.499));
        section_i2_floor_ratio = std::max(0.0, section_i2_floor_ratio);
        fiber_stiffness_ratio = std::max(0.0, fiber_stiffness_ratio);
        structural_kv_loss_factor =
            std::max(0.0, structural_kv_loss_factor);
        structural_kv_stress_cap_over_E =
            std::max(0.0, structural_kv_stress_cap_over_E);
        active_t_max_abs = std::max(0.0, active_t_max_abs);
        // Negative active_t_max_over_E disables the E-relative cap; do not clamp
        // to zero so the user's sentinel value (-1 = disabled) appears in the log.
        active_band_fraction =
            std::max(1.0e-6, std::min(active_band_fraction, 1.0));
        active_section_bins = std::max(1, active_section_bins);
        s_material_profile_diag_samples =
            std::max(2, s_material_profile_diag_samples);

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

        beta_act   = input_db->getDoubleWithDefault("BETA_ACT",   beta_act);
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
        active_k_shape_mode =
            parse_active_k_shape_mode(input_db->getStringWithDefault(
                "K_SHAPE_MODE", active_k_shape_mode_name()));
        active_s_start  = input_db->getDoubleWithDefault("ACTIVE_S_START",  active_s_start);
        active_s_end    = input_db->getDoubleWithDefault("ACTIVE_S_END",    active_s_end);
        active_s_smooth = input_db->getDoubleWithDefault("ACTIVE_S_SMOOTH", active_s_smooth);
        reference_profile_bins =
            input_db->getIntegerWithDefault("REFERENCE_PROFILE_BINS", reference_profile_bins);
        reference_centerline_end_x =
            input_db->getDoubleWithDefault("REFERENCE_CENTERLINE_END_X", reference_centerline_end_x);
        use_laplace_reference_parameterization =
            input_db->getBoolWithDefault("USE_LAPLACE_REFERENCE_PARAMETERIZATION",
                                         use_laplace_reference_parameterization);
        laplace_head_bc_width_over_L =
            input_db->getDoubleWithDefault("LAPLACE_HEAD_BC_WIDTH_OVER_L",
                                           laplace_head_bc_width_over_L);
        laplace_tail_bc_width_over_L =
            input_db->getDoubleWithDefault("LAPLACE_TAIL_BC_WIDTH_OVER_L",
                                           laplace_tail_bc_width_over_L);
        active_s_start = clamp01(active_s_start);
        if (active_s_end >= 0.0) active_s_end = clamp01(active_s_end);
        active_s_smooth = std::max(0.0, active_s_smooth);
        active_wavelength_over_L = std::max(active_wavelength_over_L, 1.0e-12);
        reference_profile_bins = std::max(8, reference_profile_bins);
        laplace_head_bc_width_over_L = std::max(0.0, laplace_head_bc_width_over_L);
        laplace_tail_bc_width_over_L = std::max(0.0, laplace_tail_bc_width_over_L);

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
        write_material_profile_diagnostics();
        if (active_s_span_norm_effective() <= 1.0e-12)
        {
            TBOX_ERROR("Effective active-body span is zero: check ACTIVE_S_START "
                       "and ACTIVE_S_END/REFERENCE_CENTERLINE_END_X.\n");
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

        s_geometry_conservation_diag_enable = input_db->getBoolWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_ENABLE", s_geometry_conservation_diag_enable);
        s_geometry_conservation_diag_interval = std::max(1, input_db->getIntegerWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_INTERVAL", s_geometry_conservation_diag_interval));
        s_geometry_conservation_diag_filename = input_db->getStringWithDefault(
            "GEOMETRY_CONSERVATION_DIAG_FILENAME", s_geometry_conservation_diag_filename);

        s_force_decomp_diag_enable = input_db->getBoolWithDefault(
            "FORCE_DECOMP_DIAG_ENABLE", s_force_decomp_diag_enable);
        s_force_decomp_diag_interval = std::max(1, input_db->getIntegerWithDefault(
            "FORCE_DECOMP_DIAG_INTERVAL", s_force_decomp_diag_interval));
        s_force_decomp_diag_filename = input_db->getStringWithDefault(
            "FORCE_DECOMP_DIAG_FILENAME", s_force_decomp_diag_filename);
        s_force_decomp_quad_order = input_db->getStringWithDefault(
            "FORCE_DECOMP_QUAD_ORDER", s_force_decomp_quad_order);

        s_active_section_diag_enable = input_db->getBoolWithDefault(
            "ACTIVE_SECTION_DIAG_ENABLE", s_active_section_diag_enable);
        s_active_section_diag_interval = std::max(
            1, input_db->getIntegerWithDefault(
                   "ACTIVE_SECTION_DIAG_INTERVAL",
                   s_active_section_diag_interval));
        s_active_section_diag_filename = input_db->getStringWithDefault(
            "ACTIVE_SECTION_DIAG_FILENAME",
            s_active_section_diag_filename);

        // ── Print startup summary ─────────────────────────────────────────
        const double L_fish = std::max(fish_length, 1.0e-12);
        const double lambda_phase = active_phase_wavelength_dimensional();
        const double active_s0 = active_s_start_norm_effective();
        const double active_s1 = active_s_end_norm_effective();
        const double active_s_span = active_s_span_norm_effective();
        const double U_act   = wave_frequency * lambda_phase;
        const double Re_act  = fluid_density * U_act * L_fish /
            std::max(fluid_viscosity, 1.0e-30);
        pout << "\n=== IBFE continuum fish: full physical material ===\n";
        pout << "  fish length = " << fish_length
             << ", reference arc length = " << ref_arc_length << "\n";
        pout << "  active s_norm range = [" << active_s0 << ", " << active_s1
             << "], span = " << active_s_span << "\n";
        pout << "  lambda_act = " << lambda_phase
             << ", U_act = " << U_act
             << ", Re_act = " << Re_act << "\n";
        pout << "  PK1 material = 2D matrix + optional tension-only fiber"
                " + Kelvin-Voigt damping + active\n";
        pout << "  target B body/peduncle/caudal = "
             << target_bending_B_body << " / "
             << target_bending_B_peduncle << " / "
             << target_bending_B_caudal << "\n";
        pout << "  matrix calibration: E=B/I, I=2*h^3/3, 2D K=E/[2(1-nu)], nu_eff="
             << material_nu_eff
             << ", section I2 floor ratio=" << section_i2_floor_ratio << "\n";
        pout << "  fiber stiffness ratio kf/E = " << fiber_stiffness_ratio
             << ", structural KV loss factor = "
             << structural_kv_loss_factor
             << ", damping stress cap/E = "
             << structural_kv_stress_cap_over_E
             << ", active stress absolute cap = " << active_t_max_abs
             << ", active stress cap/E = "
             << (active_t_max_over_E > 0.0 ?
                 std::to_string(active_t_max_over_E) : "disabled")
             << "\n";
        if (fiber_stiffness_ratio > 0.0)
        {
            pout << "  WARNING: passive tension-only fiber is enabled; "
                    "effective bending rigidity exceeds target B(s) and must "
                    "be calibrated independently.\n";
        }
        pout << "  active cross-section mode = "
             << active_cross_section_mode_name()
             << ", muscle band fraction = " << active_band_fraction
             << ", FE section bins = " << active_section_bins << "\n";
        pout << "  active mapping: FE-normalized zero-force/unit-moment q(eta), "
                "PK1 stress only\n";
        pout << "  active moment mode = " << active_moment_mode_name()
             << ", beta_act = " << beta_act
             << ", static M0 = " << static_moment_m0
             << ", initial bend amplitude = " << initial_bend_amplitude << "\n";
        pout << "  beta_act = " << beta_act
             << ", K_shape = " << active_k_shape_formula_string()
             << "\n";
        pout << "  active moment = beta_act*w(s)^2*K_shape(xi)*cos(active phase)\n";
        pout << "  active zone request s/L = [" << active_s_start << ", ";
        if (active_s_end < 0.0)
            pout << "AUTO_REFERENCE_CENTERLINE_END";
        else
            pout << active_s_end;
        pout << "], taper = " << active_s_smooth << "\n";
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
        pout << "  reference centerline end x = " << ref_centerline_end_x << "\n";
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
             << s_force_decomp_diag_filename << " (force decomposition), "
             << s_active_section_diag_filename << " (active section balance)\n";
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
        pout << "  geometry conservation diagnostics = "
             << (s_geometry_conservation_diag_enable ? "on" : "off")
             << ", interval = " << s_geometry_conservation_diag_interval
             << ", file = " << s_geometry_conservation_diag_filename << "\n";
        pout << "  force decomposition diagnostics = "
             << (s_force_decomp_diag_enable ? "on" : "off")
             << ", interval = " << s_force_decomp_diag_interval
             << ", quad_order = " << s_force_decomp_quad_order
             << ", file = " << s_force_decomp_diag_filename << "\n";
        pout << "  active section diagnostics = "
             << (s_active_section_diag_enable ? "on" : "off")
             << ", interval = " << s_active_section_diag_interval
             << ", quad_order = " << active_section_quad_order
             << ", file = " << s_active_section_diag_filename << "\n";
        pout << "  material profile diagnostics = "
             << (s_material_profile_diag_enable ? "on" : "off")
             << ", samples = " << s_material_profile_diag_samples
             << ", file = " << s_material_profile_diag_filename << "\n";

        pout << "  effective active-zone length = "
             << active_s_span * ref_arc_length;
        if (active_s_end < 0.0)
        {
            pout << " (auto endpoint from REFERENCE_CENTERLINE_END_X)\n";
        }
        else
        {
            pout << " (requested unclipped length = "
                 << (active_s_end - active_s_start) * ref_arc_length << ")\n";
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
        IBFEMethod::PK1StressFcnData PK1_matrix_data(
            PK1_matrix_stress_function, ref_geom_sys_data);
        IBFEMethod::PK1StressFcnData PK1_fiber_data(
            PK1_fiber_stress_function, ref_geom_sys_data);
        std::vector<int> velocity_vars(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) velocity_vars[d] = d;
        std::vector<SystemData> ref_geom_velocity_grad_sys_data =
            ref_geom_sys_data;
        ref_geom_velocity_grad_sys_data.push_back(
            SystemData(ib_method_ops->getVelocitySystemName(),
                       std::vector<int>(), velocity_vars));
        IBFEMethod::PK1StressFcnData PK1_damping_data(
            PK1_structural_damping_stress_function,
            ref_geom_velocity_grad_sys_data);
        IBFEMethod::PK1StressFcnData PK1_active_data(
            PK1_active_stress_function, ref_geom_sys_data);

        PK1_matrix_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_MATRIX_QUAD_ORDER", "FIFTH"));
        PK1_fiber_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_FIBER_QUAD_ORDER", "FIFTH"));
        PK1_damping_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_DAMP_QUAD_ORDER", "FIFTH"));
        PK1_active_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(
                input_db->getStringWithDefault("PK1_ACTIVE_QUAD_ORDER", "FIFTH"));

        ib_method_ops->registerPK1StressFunction(PK1_matrix_data);
        ib_method_ops->registerPK1StressFunction(PK1_fiber_data);
        ib_method_ops->registerPK1StressFunction(PK1_damping_data);
        ib_method_ops->registerPK1StressFunction(PK1_active_data);
        ib_method_ops->registerInitialCoordinateMappingFunction(
            coordinate_mapping_function);

        ib_method_ops->initializeFEEquationSystems();
        EquationSystems* equation_systems =
            ib_method_ops->getFEDataManager()->getEquationSystems();
        // IBAMR creates EquationSystems above; external systems must be added
        // before initializeFEData(), which initializes all systems together.
        add_reference_geometry_system(equation_systems);

        // ── Post-processor ─────────────────────────────────────────────────
        Pointer<IBFEPostProcessor> ib_post_processor =
            new IBFECentroidPostProcessor("IBFEPostProcessor",
                                          ib_method_ops->getFEDataManager());
        ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT,
                                                   IBFEPostProcessor::FF_fcn);

        IBFEMethod::PK1StressFcnData pk1_matrix_post_data(
            PK1_matrix_stress_function, ref_geom_sys_data);
        IBFEMethod::PK1StressFcnData pk1_fiber_post_data(
            PK1_fiber_stress_function, ref_geom_sys_data);
        IBFEMethod::PK1StressFcnData pk1_damping_post_data(
            PK1_structural_damping_stress_function,
            ref_geom_velocity_grad_sys_data);
        IBFEMethod::PK1StressFcnData pk1_active_post_data(
            PK1_active_stress_function, ref_geom_sys_data);
        ib_post_processor->registerTensorVariable(
            "sigma_matrix", MONOMIAL, CONSTANT,
            IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
            ref_geom_sys_data, &pk1_matrix_post_data);
        ib_post_processor->registerTensorVariable(
            "sigma_fiber", MONOMIAL, CONSTANT,
            IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
            ref_geom_sys_data, &pk1_fiber_post_data);
        ib_post_processor->registerTensorVariable(
            "sigma_damping", MONOMIAL, CONSTANT,
            IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
            ref_geom_velocity_grad_sys_data, &pk1_damping_post_data);
        ib_post_processor->registerTensorVariable(
            "sigma_active", MONOMIAL, CONSTANT,
            IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
            ref_geom_sys_data, &pk1_active_post_data);

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
        initialize_active_section_normalization(mesh, equation_systems);

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

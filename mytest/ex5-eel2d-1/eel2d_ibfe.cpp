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
    double A;          // amplitude coefficient, e.g. 0.125
    double s_shift;    // 0.03125
    double omega;      // 0.785 / 0.125
    double k_wave;     // 2*pi

    // geometry
    double x_leading;  // reference head x
    double y_center0;  // reference body center y

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

    Eel2DData(Pointer<Database> input_db)
      : c1_s(input_db->getDouble("C1_S")),
        kappa_s_body(input_db->getDouble("KAPPA_S_BODY")),
        eta_s_body(input_db->getDouble("ETA_S_BODY")),
        kappa_s_surface(input_db->getDouble("KAPPA_S_SURFACE")),
        eta_s_surface(input_db->getDouble("ETA_S_SURFACE")),
        L(input_db->getDouble("EEL_LENGTH")),
        A(input_db->getDouble("EEL_A")),
        s_shift(input_db->getDouble("EEL_S_SHIFT")),
        omega(input_db->getDouble("EEL_OMEGA")),
        k_wave(input_db->getDouble("EEL_KWAVE")),
        x_leading(input_db->getDouble("EEL_X_LEADING")),
        y_center0(input_db->getDouble("EEL_Y_CENTER0")),
        xcom_cur(0.0), ycom_cur(0.0), theta_cur(0.0),
        xcom_ref(0.0), ycom_ref(0.0), theta_ref(0.0),
        area_ref(0.0), polar_moment_ref(0.0),
        cached_mode_time(0.0), cached_mode_valid(false),
        disp_tx(0.0), disp_ty(0.0), disp_rot(0.0),
        vel_tx(0.0), vel_ty(0.0), vel_rot(0.0)
    {}
};

static Eel2DData* s_eel_data_state = nullptr;

inline double body_half_width(double s, const Eel2DData& d)
{
    const double width_head = 0.04 * d.L;
    const double length_head = 0.04 * d.L;

    if (s < 0.0 || s > d.L) return 0.0;
    if (s <= length_head)
        return std::sqrt(std::max(0.0, 2.0 * width_head * s - s * s));
    else
        return width_head * (d.L - s) / (d.L - length_head);
}
inline double eel_centerline_y(double s, double t, const Eel2DData& d)
{
    return d.A * ((s + d.s_shift) / (d.L + d.s_shift))
         * std::sin(d.k_wave * s - d.omega * t);
}
inline double eel_centerline_v(double s, double t, const Eel2DData& d)
{
    return -d.A * ((s + d.s_shift) / (d.L + d.s_shift))
         * d.omega * std::cos(d.k_wave * s - d.omega * t);
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
    disp_y = eel_centerline_y(s, time, d);
    vel_x = 0.0;
    vel_y = eel_centerline_v(s, time, d);
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
    const double dx_lead = d.x_leading - d.xcom_ref;
    const double dy_lead = d.y_center0 - d.ycom_ref;
    const double xhat_leading = cr * dx_lead + sr * dy_lead;

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
            const double s = xhat - xhat_leading;
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

    // ------------------------------------------------------------
    // 2) 参考头部点（leading point）也转换到 reference body frame
    //    用它来定义真正一致的轴向坐标 s
    // ------------------------------------------------------------
    const double dx_lead = d.x_leading - d.xcom_ref;
    const double dy_lead = d.y_center0 - d.ycom_ref;

    const double xhat_leading =  cr * dx_lead + sr * dy_lead;
    // const double yhat_leading = -sr * dx_lead + cr * dy_lead; // 若后续需要可保留

    // body-frame axial coordinate measured from the leading point
    const double s = xhat_ref - xhat_leading;

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

    xtar = d.xcom_cur + ct * xhat_tar - st * yhat_tar;
    ytar = d.ycom_cur + st * xhat_tar + ct * yhat_tar;

    // ------------------------------------------------------------
    // 5) 目标速度：使用去除刚体模态后的 body-frame 速度，
    //    再按当前自由位姿映射回实验室坐标。
    // ------------------------------------------------------------
    utar_x = ct * vel_x - st * vel_y;
    utar_y = st * vel_x + ct * vel_y;
}

// Tether (penalty) stress function.
void
PK1_stress_function(TensorValue<double>& PP,
                    const TensorValue<double>& FF,
                    const libMesh::Point& /*x*/,
                    const libMesh::Point& /*X*/,
                    Elem* const /*elem*/,
                    const vector<const vector<double>*>& /*var_data*/,
                    const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                    double /*time*/,
                    void* ctx)
{
    const Eel2DData* const d = reinterpret_cast<Eel2DData*>(ctx);

    PP = 2.0 * d->c1_s * (FF - tensor_inverse_transpose(FF, NDIM));
    return;
} // PK1_stress_function

// Tether (penalty) force functions.
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
    Eel2DData* d = reinterpret_cast<Eel2DData*>(ctx);
    const std::vector<double>& U = *var_data[0];

    double xtar, ytar, utar_x, utar_y;
    compute_eel_target(X, time, *d, xtar, ytar, utar_x, utar_y);

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

    // 简化版本中关闭表面附加力，仅保留体力罚项驱动摆动。
    F(0) = 0.0;
    F(1) = 0.0;
    return;
} // eel_surface_force_function

} // namespace ModelData
using namespace ModelData;

// Function prototypes
static ofstream drag_stream, lift_stream, U_L1_norm_stream, U_L2_norm_stream, U_max_norm_stream, pose_stream;
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
        const double ds = input_db->getDouble("MFAC") * dx;
        std::string elem_type = input_db->getString("ELEM_TYPE");

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
        PK1_stress_data.quad_order =
            Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault("PK1_QUAD_ORDER", "THIRD"));
        ibfe_ops->registerPK1StressFunction(PK1_stress_data);

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
        initialize_reference_projection_data(equation_systems, coords_system_name, mesh, eel_data);

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

            drag_stream.precision(10);
            lift_stream.precision(10);
            U_L1_norm_stream.precision(10);
            U_L2_norm_stream.precision(10);
            U_max_norm_stream.precision(10);
            pose_stream.precision(10);
            pose_stream << "# iter time x_com y_com theta tail_y" << endl;
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

            const double theta_prev = eel_data.theta_cur;
            compute_com_and_rigid_pose(equation_systems,
                                       coords_system_name,
                                       mesh,
                                       eel_data,
                                       eel_data.xcom_cur,
                                       eel_data.ycom_cur,
                                       eel_data.theta_cur,
                                       theta_prev);
            pout << "Pose(COM/theta): (" << eel_data.xcom_cur << ", " << eel_data.ycom_cur << ", " << eel_data.theta_cur
                 << ")\n";

            time_integrator->advanceHierarchy(dt);
            loop_time += dt;

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
    void* const eel_data_ptr = reinterpret_cast<void*>(eel_data);
    bool use_boundary_mesh = input_db->getBoolWithDefault("USE_BOUNDARY_MESH", false);
    const unsigned int dim = mesh.mesh_dimension();
    double F_integral[NDIM];
    for (unsigned int d = 0; d < NDIM; ++d) F_integral[d] = 0.0;

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

    const auto el_begin = mesh.active_local_elements_begin();
    const auto el_end = mesh.active_local_elements_end();
    if (!use_boundary_mesh)
    {
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

            const unsigned int n_qp = qrule->n_points();
            for (unsigned int qp = 0; qp < n_qp; ++qp)
            {
                interpolate(x, qp, X_node, phi);
                jacobian(FF, qp, X_node, dphi);
                interpolate(U, qp, U_node, phi);
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    U_qp_vec[d] = U(d);
                }
                eel_body_force_function(F, FF, x, q_point[qp], elem, var_data, grad_var_data, loop_time, eel_data_ptr);
                for (int d = 0; d < NDIM; ++d)
                {
                    F_integral[d] += F(d) * JxW[qp];
                }
            }
            for (unsigned short int side = 0; side < elem->n_sides(); ++side)
            {
                if (elem->neighbor_ptr(side)) continue;
                fe_face->reinit(elem, side);
                const unsigned int n_qp_face = qrule_face->n_points();
                for (unsigned int qp = 0; qp < n_qp_face; ++qp)
                {
                    interpolate(x, qp, X_node, phi_face);
                    jacobian(FF, qp, X_node, dphi_face);
                    interpolate(U, qp, U_node, phi_face);
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        U_qp_vec[d] = U(d);
                    }
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
                                          loop_time,
                                          eel_data_ptr);
                    for (int d = 0; d < NDIM; ++d)
                    {
                        F_integral[d] += F(d) * JxW_face[qp];
                    }
                }
            }
        }
    }
    else
    {
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

            for (unsigned short int side = 0; side < elem->n_sides(); ++side)
            {
                fe_face->reinit(elem, side);
                const unsigned int n_qp_face = qrule_face->n_points();
                for (unsigned int qp = 0; qp < n_qp_face; ++qp)
                {
                    interpolate(x, qp, X_node, phi_face);
                    jacobian(FF, qp, X_node, dphi_face);
                    interpolate(U, qp, U_node, phi_face);
                    for (unsigned int d = 0; d < NDIM; ++d)
                    {
                        U_qp_vec[d] = U(d);
                    }
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
                                          loop_time,
                                          eel_data_ptr);
                    for (int d = 0; d < NDIM; ++d)
                    {
                        F_integral[d] += F(d) * JxW_face[qp];
                    }
                }
            }
        }
    }
    IBTK_MPI::sumReduction(F_integral, NDIM);

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
    const double tail_y = compute_tail_y(equation_systems, coords_system_name, mesh, *eel_data);

    static const double rho = 1.0;
    static const double U_max = 1.0;
    static const double D = 1.0;
    if (IBTK_MPI::getRank() == 0)
    {
        drag_stream << loop_time << " " << -F_integral[0] / (0.5 * rho * U_max * U_max * D) << endl;
        lift_stream << loop_time << " " << -F_integral[1] / (0.5 * rho * U_max * U_max * D) << endl;
        pose_stream << iteration_num << " " << loop_time << " " << xcom << " " << ycom << " " << theta << " " << tail_y << endl;
    }
    return;
} // postprocess_data

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

    // rigid-body state (updated every step)
    double xcom_cur;
    double ycom_cur;
    double theta_cur;

    // reference body-frame origin
    double xcom_ref;
    double ycom_ref;
    double theta_ref;

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
        xcom_ref(0.0), ycom_ref(0.0), theta_ref(0.0)
    {}
};

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
void compute_com_and_orientation(
    EquationSystems* equation_systems,
    const std::string& coords_system_name,
    MeshBase& mesh,
    double& xcom,
    double& ycom,
    double& theta);

inline void compute_eel_target(
    const libMesh::Point& X,
    double time,
    const Eel2DData& d,
    double& xtar,
    double& ytar,
    double& utar_x,
    double& utar_y);

void
compute_com_and_orientation(EquationSystems* equation_systems,
                            const std::string& coords_system_name,
                            MeshBase& mesh,
                            double& xcom,
                            double& ycom,
                            double& theta)
{
    System& X_system = equation_systems->get_system<System>(coords_system_name);
    NumericVector<double>* X_ghost_vec = X_system.current_local_solution.get();
    const DofMap& dof_map = X_system.get_dof_map();

    std::set<dof_id_type> visited_nodes;
    double xsum = 0.0, ysum = 0.0;
    int npts = 0;

    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        for (unsigned int k = 0; k < elem->n_nodes(); ++k)
        {
            const Node* node = elem->node_ptr(k);
            if (!visited_nodes.insert(node->id()).second) continue;
            std::vector<dof_id_type> dof_idx_x, dof_idx_y;
            dof_map.dof_indices(node, dof_idx_x, 0);
            dof_map.dof_indices(node, dof_idx_y, 1);
            xsum += (*X_ghost_vec)(dof_idx_x[0]);
            ysum += (*X_ghost_vec)(dof_idx_y[0]);
            ++npts;
        }
    }

    double local[3] = { xsum, ysum, static_cast<double>(npts) };
    IBTK_MPI::sumReduction(local, 3);
    xcom = local[0] / std::max(1.0, local[2]);
    ycom = local[1] / std::max(1.0, local[2]);

    double cxx = 0.0, cyy = 0.0, cxy = 0.0;
    visited_nodes.clear();
    for (auto el_it = mesh.active_local_elements_begin(); el_it != mesh.active_local_elements_end(); ++el_it)
    {
        const Elem* elem = *el_it;
        for (unsigned int k = 0; k < elem->n_nodes(); ++k)
        {
            const Node* node = elem->node_ptr(k);
            if (!visited_nodes.insert(node->id()).second) continue;
            std::vector<dof_id_type> dof_idx_x, dof_idx_y;
            dof_map.dof_indices(node, dof_idx_x, 0);
            dof_map.dof_indices(node, dof_idx_y, 1);
            const double dx = (*X_ghost_vec)(dof_idx_x[0]) - xcom;
            const double dy = (*X_ghost_vec)(dof_idx_y[0]) - ycom;
            cxx += dx * dx;
            cxy += dx * dy;
            cyy += dy * dy;
        }
    }

    double local_cov[3] = { cxx, cyy, cxy };
    IBTK_MPI::sumReduction(local_cov, 3);
    theta = 0.5 * std::atan2(2.0 * local_cov[2], local_cov[0] - local_cov[1]);
}

inline void
compute_eel_target(const libMesh::Point& X,
                   double time,
                   const Eel2DData& d,
                   double& xtar,
                   double& ytar,
                   double& utar_x,
                   double& utar_y)
{
    const double s = X(0) - d.x_leading;
    const double yc = eel_centerline_y(s, time, d);
    const double vc = eel_centerline_v(s, time, d);

    const double xb = X(0) - d.xcom_ref;
    const double yb = (X(1) - d.ycom_ref) + yc;

    const double ct = std::cos(d.theta_cur);
    const double st = std::sin(d.theta_cur);
    xtar = d.xcom_cur + ct * xb - st * yb;
    ytar = d.ycom_cur + st * xb + ct * yb;

    utar_x = -st * vc;
    utar_y = ct * vc;
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
    const Eel2DData* d = reinterpret_cast<Eel2DData*>(ctx);
    const std::vector<double>& U = *var_data[0];

    const double s = X(0) - d->x_leading;   // material arc-wise coordinate

    const double yc = eel_centerline_y(s, time, *d);
    const double vc = eel_centerline_v(s, time, *d);

    // target in body frame, relative to reference COM
    const double xb = X(0) - d->xcom_ref;
    const double yb = (X(1) - d->ycom_ref) + yc;

    const double ct = std::cos(d->theta_cur);
    const double st = std::sin(d->theta_cur);

    const double xtar = d->xcom_cur + ct * xb - st * yb;
    const double ytar = d->ycom_cur + st * xb + ct * yb;

    // first version: only deformational target velocity
    const double utar_x = -st * vc;
    const double utar_y =  ct * vc;

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

    double xtar, ytar, utar_x, utar_y;
    compute_eel_target(X, time, *d, xtar, ytar, utar_x, utar_y);

    F(0) = d->kappa_s_surface * (xtar - x(0)) + d->eta_s_surface * (utar_x - U(0));
    F(1) = d->kappa_s_surface * (ytar - x(1)) + d->eta_s_surface * (utar_y - U(1));
    return;
} // eel_surface_force_function

} // namespace ModelData
using namespace ModelData;

// Function prototypes
static ofstream drag_stream, lift_stream, U_L1_norm_stream, U_L2_norm_stream, U_max_norm_stream;
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
        solid_mesh.read("IBFE_Mesh2D_128.msh");
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
        Eel2DData eel_data(input_db);
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

        IBFEMethod::LagSurfaceForceFcnData surface_fcn_data(eel_surface_force_function, sys_data, eel_data_ptr);
        ibfe_ops->registerLagSurfaceForceFunction(surface_fcn_data);

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

        compute_com_and_orientation(equation_systems, coords_system_name, mesh, eel_data.xcom_cur, eel_data.ycom_cur, eel_data.theta_cur);
        eel_data.xcom_ref = eel_data.xcom_cur;
        eel_data.ycom_ref = eel_data.ycom_cur;
        eel_data.theta_ref = eel_data.theta_cur;

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

            drag_stream.precision(10);
            lift_stream.precision(10);
            U_L1_norm_stream.precision(10);
            U_L2_norm_stream.precision(10);
            U_max_norm_stream.precision(10);
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

            compute_com_and_orientation(
                equation_systems, coords_system_name, mesh, eel_data.xcom_cur, eel_data.ycom_cur, eel_data.theta_cur);

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
                 const int /*iteration_num*/,
                 const double loop_time,
                 const string& /*data_dump_dirname*/)
{
    Eel2DData eel_data(input_db);
    void* const eel_data_ptr = reinterpret_cast<void*>(&eel_data);
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
                        pout << F(d) << endl;
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
    static const double rho = 1.0;
    static const double U_max = 1.0;
    static const double D = 1.0;
    if (IBTK_MPI::getRank() == 0)
    {
        drag_stream << loop_time << " " << -F_integral[0] / (0.5 * rho * U_max * U_max * D) << endl;
        lift_stream << loop_time << " " << -F_integral[1] / (0.5 * rho * U_max * U_max * D) << endl;
    }
    return;
} // postprocess_data
